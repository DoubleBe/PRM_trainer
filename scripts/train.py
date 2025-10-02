import argparse, os, numpy as np

from typing import Dict
from datasets import DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer, set_seed,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score

from clearml import Task

from src.data_utils import make_hf_dataset, tokenize_and_align_labels, SPECIAL_STEP, SPECIAL_Q
from src.clearml_utils import ClearMLEvalCallback

# Максимальная длина последовательности для токенизации
MAX_LEN = 4096

# Настройка конфигурации ClearML для логирования экспериментов
os.environ['CLEARML_CONFIG_FILE'] = '/home/jovyan/.mlspace/envs/garaev_hallu/MathClass/clearml.conf'

def parse_args():
    """Парсинг аргументов командной строки для обучения модели"""
    p = argparse.ArgumentParser()
    
    # Основные параметры модели и данных
    p.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base", 
                   help="Название предобученной модели для загрузки из HuggingFace Hub")
    p.add_argument("--train_file", type=str, required=True, 
                   help="Путь к JSONL файлу с обучающими данными")
    p.add_argument("--eval_file", type=str, required=True, 
                   help="Путь к JSONL файлу с валидационными данными")
    p.add_argument("--output_dir", type=str, required=True, 
                   help="Директория для сохранения обученной модели и чекпоинтов")
    
    # Параметры обучения
    p.add_argument("--batch_size", type=int, default=16, 
                   help="Размер батча для обучения и валидации")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, 
                   help="Количество шагов накопления градиентов")
    p.add_argument("--num_train_epochs", type=int, default=5, 
                   help="Количество эпох обучения")
    p.add_argument("--learning_rate", type=float, default=2e-5, 
                   help="Скорость обучения")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", 
                   help="Тип планировщика скорости обучения")
    p.add_argument("--warmup_ratio", type=float, default=0.03, 
                   help="Доля шагов для warmup")
    
    # Параметры логирования и сохранения
    p.add_argument("--logging_steps", type=int, default=20, 
                   help="Частота логирования метрик (каждые N шагов)")
    p.add_argument("--eval_strategy", type=str, default="steps", 
                   help="Стратегия оценки модели (steps, epoch)")
    p.add_argument("--eval_steps", type=int, default=200, 
                   help="Частота оценки модели (каждые N шагов)")
    p.add_argument("--save_steps", type=int, default=200, 
                   help="Частота сохранения чекпоинтов (каждые N шагов)")
    p.add_argument("--save_total_limit", type=int, default=2, 
                   help="Максимальное количество сохраняемых чекпоинтов")
    
    # Параметры воспроизводимости и оптимизации
    p.add_argument("--seed", type=int, default=42, 
                   help="Случайное зерно для воспроизводимости")
    p.add_argument("--bf16", type=lambda x: str(x).lower()=='true', default=False, 
                   help="Использовать bfloat16 для ускорения обучения")
    p.add_argument("--fp16", type=lambda x: str(x).lower()=='true', default=False, 
                   help="Использовать float16 для ускорения обучения")
    p.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower()=='true', default=False, 
                   help="Использовать checkpointing для экономии памяти")
    
    # Параметры ранней остановки
    p.add_argument("--early_stopping_patience", type=int, default=3, 
                   help="Терпение для ранней остановки (количество оценок без улучшения). 0 = отключить раннюю остановку")
    p.add_argument("--early_stopping_threshold", type=float, default=0.001, 
                   help="Минимальное улучшение для ранней остановки")
    
    # Параметры эксперимента и метрик
    p.add_argument("--run_name", type=str, default=None, 
                   help="Название эксперимента в ClearML")
    p.add_argument("--metric_for_best", type=str, default=None, 
                   help="Метрика для выбора лучшей модели (processbench_f1, f1, etc.)")
    p.add_argument("--error_threshold", type=float, default=0.5,
                   help="Порог вероятности для классификации ошибки")
    p.add_argument("--project", type=str, default="prm")
    
    return p.parse_args()

def main():
    """Основная функция для обучения модели классификации токенов"""
    args = parse_args()
    set_seed(args.seed)  # Установка случайного зерна для воспроизводимости

    # Инициализация ClearML для логирования эксперимента
    task = Task.init(
        project_name=args.project,
        task_name=args.run_name or "run",
        output_uri=None
    )
    logger = task.get_logger()

    # Загрузка и настройка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    special_tokens = {"additional_special_tokens": [SPECIAL_Q, SPECIAL_STEP]}  # Добавление специальных токенов
    tokenizer.add_special_tokens(special_tokens)

    # Загрузка и токенизация данных
    dsd_raw: DatasetDict = make_hf_dataset(args.train_file, args.eval_file)
    dsd_tok = dsd_raw.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, max_length=MAX_LEN),
        batched=True,
        remove_columns=dsd_raw["train"].column_names
    )

    # Загрузка модели и изменение размера эмбеддингов для новых токенов
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        """Вычисление метрик качества модели на валидационных данных"""
        logits, labels = eval_pred
        
        # Обработка логгитов в зависимости от их размерности
        if logits.ndim == 3:
            # Стандартная обработка для классификации токенов
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))  # Стабилизация softmax
            probs = probs / probs.sum(axis=-1, keepdims=True)
            prob1 = probs[..., 1]  # Вероятности класса "ошибка"
            preds = np.argmax(logits, axis=-1)
        else:
            # Обработка для бинарной классификации
            preds = (logits > args.error_threshold).astype(int)
            prob1 = logits

        # Фильтрация игнорируемых токенов (label = -100)
        mask = labels != -100
        y_true = labels[mask].astype(int)
        y_pred = preds[mask].astype(int)

        # Вычисление базовых метрик
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1_token, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        # Вычисление PR-AUC с обработкой ошибок
        try:
            pr_auc = float(average_precision_score(y_true, prob1[mask]))
        except Exception:
            pr_auc = float("nan")

        # Вычисление метрик на уровне шагов (ProcessBench метрики)
        acc_err = acc_correct = 0.0
        hits_err = hits_correct = num_err = num_correct = 0
        step_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_STEP)

        # Анализ каждого примера для поиска первого ошибочного шага
        for i in range(labels.shape[0]):
            lab = labels[i]
            prd = preds[i]
            if i >= len(dsd_tok["validation"]):
                break
            input_ids = dsd_tok["validation"][i]["input_ids"]
            step_positions = [pos for pos, tok in enumerate(input_ids) if tok == step_token_id]

            # Поиск индекса первого ошибочного шага в истинных метках
            true_idx = -1
            pred_idx = -1
            for j, pos in enumerate(step_positions):
                if lab[pos] == 1:
                    true_idx = j
                    break
            # Поиск индекса первого ошибочного шага в предсказаниях
            for j, pos in enumerate(step_positions):
                if prd[pos] == 1:
                    pred_idx = j
                    break
            
            # Подсчет точности для примеров с ошибками и без ошибок
            if true_idx != -1:
                num_err += 1
                if pred_idx == true_idx:
                    hits_err += 1
            else:
                num_correct += 1
                if pred_idx == -1:
                    hits_correct += 1

        # Вычисление финальных метрик
        if num_err > 0: acc_err = hits_err / num_err
        if num_correct > 0: acc_correct = hits_correct / num_correct
        f1_pb = 0.0
        if (acc_err + acc_correct) > 0:
            f1_pb = 2 * acc_err * acc_correct / (acc_err + acc_correct)  # Гармоническое среднее

        return {
            "step_accuracy": float(acc),      # Точность на уровне токенов
            "step_precision": float(prec),    # Точность на уровне токенов
            "step_recall": float(rec),        # Полнота на уровне токенов
            "f1": float(f1_pb),               # Основная F1 метрика ProcessBench
            "step_f1": float(f1_token),       # F1 на уровне токенов
            "acc_err": float(acc_err),        # Точность для примеров с ошибками
            "acc_correct": float(acc_correct), # Точность для примеров без ошибок
            "pr_auc": float(pr_auc),          # PR-AUC
        }

    # Настройка параметров обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,  # Загрузка лучшей модели в конце
        metric_for_best_model=args.metric_for_best or "f1",
        greater_is_better=True,
        run_name=args.run_name,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Создание тренера с коллбэками
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsd_tok["train"],
        eval_dataset=dsd_tok["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )] if args.early_stopping_patience > 0 else None,  # Условное добавление ранней остановки
    )

    # Добавление кастомного коллбэка для ClearML логирования
    model.trainer = trainer
    trainer.add_callback(ClearMLEvalCallback(logger, dsd_raw["validation"], tokenizer, step_token=SPECIAL_STEP))

    # Запуск обучения и сохранение модели
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()