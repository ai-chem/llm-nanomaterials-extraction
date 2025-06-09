#!/usr/bin/env python3
"""
Формирование финального HuggingFace DatasetDict для SFT.
"""
from datasets import Dataset, DatasetDict
from utils import load_jsonl, fix_content
import sys
import os

train_path = sys.argv[1] if len(sys.argv) > 1 else "../datasplits/clean_train_converted.jsonl"
val_path = sys.argv[2] if len(sys.argv) > 2 else "../datasplits/clean_val_converted.jsonl"
test_path = sys.argv[3] if len(sys.argv) > 3 else "../datasplits/clean_test_converted.jsonl"
out_path = sys.argv[4] if len(sys.argv) > 4 else "../hf_dataset"

train_data = [fix_content(x) for x in load_jsonl(train_path)]
val_data = [fix_content(x) for x in load_jsonl(val_path)]
test_data = [fix_content(x) for x in load_jsonl(test_path)]

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data)
})

os.makedirs(out_path, exist_ok=True)
dataset.save_to_disk(out_path)
print(f"Готово! Датасет сохранён в {out_path}") 