from typing import List
import pandas as pd
import numpy as np
from clearml import Logger
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

class ClearMLEvalCallback(TrainerCallback):
    """Кастомный коллбэк для логирования метрик и ошибок в ClearML"""
    
    def __init__(self, logger: Logger, raw_eval_ds, tokenizer, step_token: str="[STEP]"):
        self.logger = logger          # Логгер ClearML для отправки метрик
        self.raw_eval_ds = raw_eval_ds  # Исходные данные для анализа ошибок
        self.tokenizer = tokenizer    # Токенизатор для работы с токенами
        self.step_token = step_token  # Специальный токен для обозначения шагов

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Логирование метрик в ClearML при каждом шаге логирования"""
        if not logs:
            return
        step = int(state.global_step)
        
        # Отправка всех eval метрик в ClearML
        for k, v in logs.items():
            if isinstance(v, (int, float)) and k.startswith("eval_"):
                name = k.replace("eval_", "")  # Убираем префикс "eval_"
                self.logger.report_scalar(title=name,
                                        series="eval",
                                        value=float(v), iteration=step)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Анализ ошибок и построение PR-кривой при каждой оценке модели"""
        trainer = getattr(kwargs.get("model"), "trainer", None)
        if trainer is None:
            return
        
        # Получение предсказаний на валидационных данных
        pred_out = trainer.predict(trainer.eval_dataset)
        logits = pred_out.predictions
        labels = pred_out.label_ids

        # Подготовка данных для анализа
        mask = labels != -100  # Исключение игнорируемых токенов
        prob1 = logits[..., 1] if logits.ndim == 3 else logits  # Вероятности класса "ошибка"
        y_true = labels[mask].astype(int)
        y_scores = prob1[mask]
        
        # Построение и отправка PR-кривой в ClearML
        if y_true.size > 0:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            self.logger.report_scatter2d(
                title="PR Curve",
                series="Precision-Recall",
                scatter=[[recall[i], precision[i]] for i in range(len(precision))],
                iteration=int(state.global_step)
            )

        # Анализ ошибок на уровне шагов
        preds01 = (prob1 > 0.5).astype(int)  # Бинарные предсказания
        rows: List[List] = []
        step_token_id = self.tokenizer.convert_tokens_to_ids(self.step_token)

        # Поиск примеров с неправильными предсказаниями
        for i in range(labels.shape[0]):
            y = labels[i]  # Истинные метки
            p = preds01[i]  # Предсказания
            input_ids = trainer.eval_dataset[i]["input_ids"]
            step_positions = [pos for pos, tok in enumerate(input_ids) if tok == step_token_id]

            # Поиск индексов первого ошибочного шага
            true_idx = next((j for j,pos in enumerate(step_positions) if y[pos] == 1), -1)
            pred_idx = next((j for j,pos in enumerate(step_positions) if p[pos] == 1), -1)

            # Сохранение только неправильных предсказаний
            if true_idx != pred_idx:
                ex = self.raw_eval_ds[i]
                steps = ex.get("steps", [])
                def get_step(idx): return steps[idx] if 0 <= idx < len(steps) else ""
                rows.append([
                    ex.get("id", str(i)),
                    ex.get("question",""),
                    int(true_idx),
                    int(pred_idx),
                    get_step(true_idx),  # Текст истинного ошибочного шага
                    get_step(pred_idx),  # Текст предсказанного ошибочного шага
                ])

        # Отправка таблицы ошибок в ClearML
        if rows:
            df = pd.DataFrame(rows, columns=[
                "id","question","true_first_error_index","pred_first_error_index","true_step_text","pred_step_text"
            ])
            self.logger.report_table(
                title="validation_mistakes",
                series="errors",
                iteration=int(state.global_step),
                table_plot=df
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Завершение обучения - в данном случае ничего не делаем"""
        pass