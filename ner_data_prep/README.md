# NER SFT Data Preparation

Скрипты для подготовки данных для обучения моделей NER (Supervised Fine-Tuning).

## Шаги подготовки

1. **Сбор и разбиение**  
   `python scripts/01_collect_and_split.py data datasplits`
2. **Очистка и валидация**  
   `python scripts/02_clean_jsonl.py --in datasplits/train.jsonl --out datasplits/clean_train.jsonl --max_tokens 12000`
3. **Преобразование в формат диалога**  
   `python scripts/03.py datasplits/clean_train.jsonl datasplits/clean_train_converted.jsonl`
4. **Формирование HuggingFace Dataset**  
   `python scripts/04.py datasplits/clean_train_converted.jsonl datasplits/clean_val_converted.jsonl datasplits/clean_test_converted.jsonl hf_dataset`

## Зависимости

```
pip install -r requirements.txt
```

## Структура данных

- `prompt`: исходный текст
- `completion`: разметка/ответ
- `domain`: домен задачи

## Лицензия

MIT 