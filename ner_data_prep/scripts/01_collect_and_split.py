#!/usr/bin/env python3
"""
Сбор всех json-файлов по доменам, разбиение на train/val/test, сохранение в datasplits.
"""
import os
import json
import random
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "datasplits"
VAL_FRAC = 0.05
TEST_FRAC = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

all_samples = []

for domain in os.listdir(ROOT):
    print(domain)
    domain_dir = os.path.join(ROOT, domain)
    if not os.path.isdir(domain_dir):
        continue
    for fname in os.listdir(domain_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(domain_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Ошибка чтения {fpath}: {e}")
                continue
            # Если в файле список — перебираем, иначе один объект
            if isinstance(data, list):
                for item in data:
                    if isinstance(item.get("completion"), list):
                        for comp in item["completion"]:
                            all_samples.append({
                                "prompt": item.get("prompt", ""),
                                "completion": comp,
                                "domain": domain 
                            })
                    else:
                        all_samples.append({
                            "prompt": item.get("prompt", ""),
                            "completion": item.get("completion", ""),
                            "domain": domain 
                        })
            else:
                if isinstance(data.get("completion"), list):
                    for comp in data["completion"]:
                        all_samples.append({
                            "prompt": data.get("prompt", ""),
                            "completion": comp,
                            "domain": domain 
                        })
                else:
                    all_samples.append({
                        "prompt": data.get("prompt", ""),
                        "completion": data.get("completion", ""),
                        "domain": domain 
                    })

random.shuffle(all_samples)
n = len(all_samples)
n_val = int(n * VAL_FRAC)
n_test = int(n * TEST_FRAC)

val = all_samples[:n_val]
test = all_samples[n_val:n_val + n_test]
train = all_samples[n_val + n_test:]

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(f"{OUT_DIR}/train.jsonl", train)
write_jsonl(f"{OUT_DIR}/val.jsonl", val)
write_jsonl(f"{OUT_DIR}/test.jsonl", test)

print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}") 