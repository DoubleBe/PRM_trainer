from typing import List, Dict, Any, Optional
import json
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from transformers import PreTrainedTokenizerBase

# Специальные токены для структурирования математических задач
SPECIAL_STEP = "[STEP]"  # Токен для обозначения шага решения
SPECIAL_Q = "[Q]"        # Токен для обозначения вопроса

def load_jsonl(path: str):
    """Загрузка данных из JSONL файла"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Пропуск пустых строк
                data.append(json.loads(line))
    return data

def build_text(question: str, steps: List[str]) -> str:
    """Построение текста из вопроса и шагов решения с использованием специальных токенов"""
    parts = [f"{SPECIAL_Q} {question.strip()}"]  # Добавление вопроса с токеном [Q]
    for idx, s in enumerate(steps, start=1):
        parts.append(f"{SPECIAL_STEP} {idx}. {s.strip()}")  # Добавление шагов с токеном [STEP]
    return "\n".join(parts)

def make_hf_dataset(train_file: Optional[str]=None,
                    eval_file: Optional[str]=None,
                    test_file: Optional[str]=None) -> DatasetDict:
    """Создание HuggingFace DatasetDict из JSONL файлов с валидацией данных"""
    
    def _to_records(path: str, split: str):
        """Загрузка и валидация данных из одного файла"""
        rows = load_jsonl(path)
        for r in rows:
            r["id"] = str(r.get("id", f"{split}_{len(rows)}"))  # Нормализация ID
            
            # Валидация обязательных полей
            if "steps" not in r or not isinstance(r["steps"], list):
                raise ValueError(f"`steps` must be a list in {path}")
            if "question" not in r:
                raise ValueError(f"`question` is missing in {path}")
            if "labels" in r and r["labels"] is not None:
                if len(r["labels"]) != len(r["steps"]):
                    raise ValueError(f"len(labels) != len(steps) for id={r['id']} in {path}")
        return rows

    # Загрузка данных из всех предоставленных файлов
    d = {}
    if train_file: d["train"] = _to_records(train_file, "train")
    if eval_file: d["validation"] = _to_records(eval_file, "validation")
    if test_file: d["test"] = _to_records(test_file, "test")

    # Определение схемы данных для HuggingFace Dataset
    features = Features({
        "id": Value("string"),                    # Уникальный идентификатор
        "question": Value("string"),              # Текст вопроса
        "steps": Sequence(Value("string")),       # Список шагов решения
        "labels": Sequence(Value("int64"))        # Метки ошибок для каждого шага
    })
    return DatasetDict({k: Dataset.from_list(v, features=features) for k,v in d.items()})

def tokenize_and_align_labels(examples: Dict[str, Any],
                              tokenizer: PreTrainedTokenizerBase,
                              max_length: int) -> Dict[str, Any]:
    """Токенизация текста и выравнивание меток с токенами"""
    
    # Построение текстов из вопросов и шагов
    texts = [build_text(q, s) for q, s in zip(examples["question"], examples["steps"])]
    
    # Токенизация текстов
    enc = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=False,
        add_special_tokens=True,
    )
    step_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_STEP)

    # Выравнивание меток с токенами
    all_labels = []
    for i, input_ids in enumerate(enc["input_ids"]):
        L = [-100] * len(input_ids)  # Инициализация меток как игнорируемых
        ex_labels = examples["labels"][i] if ("labels" in examples and examples["labels"] is not None) else None
        step_positions = [pos for pos, tok in enumerate(input_ids) if tok == step_token_id]  # Позиции токенов [STEP]
        
        # Присвоение меток только токенам шагов
        if ex_labels is not None:
            n = min(len(step_positions), len(ex_labels))  # Ограничение по количеству шагов
            for j in range(n):
                L[step_positions[j]] = int(ex_labels[j])  # Присвоение метки токену шага
        all_labels.append(L)
    
    enc["labels"] = all_labels
    return enc