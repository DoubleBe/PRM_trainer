
## Установка
```bash
pip install -r requirements.txt
```

### Полный пример с настройками
```bash
nohup python scripts/train.py  --model_name answerdotai/ModernBERT-base  --train_file data/preproc_prm800k/train.jsonl  --eval_file data/preproc_prm800k/validation.jsonl  --output_dir outputs/model  --batch_size 16  --gradient_accumulation_steps 2  --num_train_epochs 5  --learning_rate 2e-5  --lr_scheduler_type cosine  --warmup_ratio 0.03  --logging_steps 20  --eval_strategy steps  --eval_steps 200  --save_steps 200  --save_total_limit 2  --seed 42  --bf16 True  --gradient_checkpointing True  --early_stopping_patience 3  --early_stopping_threshold 0.001  --metric_for_best f1  --error_threshold 0.5  --run_name my_experiment  --project processbench > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python -m scripts.train  --model_name answerdotai/ModernBERT-base  --train_file data/preproc_prm800k/train.jsonl  --eval_file data/preproc_prm800k/validation.jsonl  --output_dir outputs/model  --bf16 True  --gradient_checkpointing True --metric_for_best f1  --error_threshold 0.5  --run_name mBERT-base_PRM800K  --project PRM > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

## Инференс
```bash
python scripts/inference.py \
  --model_dir outputs/model \
  --data_file data/validation.jsonl \
  --out_file outputs/preds.jsonl \
  --batch_size 8
```

## ClearML логирование

Скрипт автоматически настраивает ClearML и логирует метрики через callback.


### Логируемые метрики
- **Скалярные метрики**: `accuracy`, `precision`, `recall`, `token_f1`, `acc_err`, `acc_correct`, `f1` (ProcessBench-F1), `pr_auc`
- **PR кривая** (precision–recall) для валидации
- **Таблица ошибок** `validation_mistakes` с примерами неправильно предсказанных индексов первых ошибок
