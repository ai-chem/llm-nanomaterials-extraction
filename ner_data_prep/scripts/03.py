#!/usr/bin/env python3
"""
Преобразование очищенного jsonl в формат conversation для SFT.
"""
from utils import load_jsonl
import json
import sys

input_path = sys.argv[1] if len(sys.argv) > 1 else "../datasplits/clean_test.jsonl"
output_path = sys.argv[2] if len(sys.argv) > 2 else "../datasplits/clean_test_converted.jsonl"

def convert_line(data):
    conversation = [
        {"role": "user", "content": data["prompt"]},
        {"role": "assistant", "content": data["completion"]},
        {"domain": data["domain"]}
    ]
    return {"conversation": conversation, "fname": data.get("fname", "")}

if __name__ == "__main__":
    data = load_jsonl(input_path)
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in data:
            converted = convert_line(item)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
    print(f"Конвертация завершена! Результат: {output_path}") 