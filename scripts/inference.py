import argparse, os, json, numpy as np
from datasets import Dataset, Features, Value, Sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments

from src.data_utils import load_jsonl, tokenize_and_align_labels, SPECIAL_STEP, SPECIAL_Q

# Максимальная длина последовательности для токенизации
MAX_LEN = 4096

def parse_args():
    """Парсинг аргументов командной строки для инференса"""
    p = argparse.ArgumentParser(description="Скрипт инференса для классификации токенов")
    
    p.add_argument("--model_dir", type=str, required=True)  # Путь к обученной модели
    p.add_argument("--data_file", type=str, required=True)  # Путь к JSONL файлу с данными
    p.add_argument("--out_file", type=str, required=True)   # Путь для сохранения предсказаний
    p.add_argument("--batch_size", type=int, default=4)     # Размер батча для инференса
    
    return p.parse_args()

def main():
    """Основная функция для выполнения инференса"""
    args = parse_args()
    
    # Загрузка и предобработка данных
    rows = load_jsonl(args.data_file)
    for r in rows:
        r["id"] = str(r.get("id", ""))  # Нормализация ID
        if "labels" not in r:
            r["labels"] = []  # Добавление пустых меток если их нет

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_STEP, SPECIAL_Q]})  # Добавление специальных токенов
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    # Создание датасета с правильной схемой
    features = Features({
        "id": Value("string"),
        "question": Value("string"),
        "steps": Sequence(Value("string")),
        "labels": Sequence(Value("int64"))
    })
    ds = Dataset.from_list(rows, features=features)
    
    # Токенизация и выравнивание меток
    tokenized = ds.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer=tokenizer, max_length=MAX_LEN),
        batched=True, 
        remove_columns=ds.column_names
    )

    # Настройка коллатора и тренера для инференса
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    targs = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "pred_tmp"),  # Временная директория
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False,
        report_to="none"  # Отключение логирования
    )
    trainer = Trainer(model=model, args=targs, tokenizer=tokenizer, data_collator=collator)
    
    # Получение предсказаний модели
    preds_output = trainer.predict(tokenized)
    logits = preds_output.predictions
    
    # Вычисление вероятностей через softmax
    exps = np.exp(logits - logits.max(axis=-1, keepdims=True))  # Стабилизация для избежания overflow
    probs = exps / exps.sum(axis=-1, keepdims=True)

    # Поиск позиций токенов шагов и обработка предсказаний
    step_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_STEP)
    input_ids = tokenized["input_ids"]
    outputs = []

    for ex_idx, row in enumerate(rows):
        ids = input_ids[ex_idx]
        step_positions = [k for k, t in enumerate(ids) if t == step_token_id]  # Позиции токенов [STEP]
        step_probs = [float(probs[ex_idx, pos, 1]) for pos in step_positions]  # Вероятности ошибки для каждого шага
        step_preds = [1 if p >= 0.5 else 0 for p in step_probs]  # Бинарные предсказания
        
        # Поиск первого шага с ошибкой
        try:
            fei = step_preds.index(1)  # Индекс первого шага с ошибкой
            has_error = True
        except ValueError:
            fei = -1  # Ошибок нет
            has_error = False

        outputs.append({
            "id": row["id"], 
            "pred_first_error_index": int(fei), 
            "pred_has_error": bool(has_error), 
            "step_probs": step_probs  # Сохранение вероятностей для анализа
        })

    # Сохранение результатов в JSONL формате
    with open(args.out_file, "w", encoding="utf-8") as out:
        for r in outputs:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"Predictions saved to {args.out_file}")

if __name__ == "__main__":
    main()