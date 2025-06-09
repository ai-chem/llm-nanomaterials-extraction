#!/usr/bin/env python3
"""
Очистка и валидация JSONL-файла с помощью Pydantic, pandas, tiktoken, rich и typer.

Пример запуска:
python scripts/02_clean_jsonl.py --in datasplits/train.jsonl --out datasplits/clean_train.jsonl --max_tokens 12000

- Стримово читает строки, валидирует через Pydantic-схемы (Context, Measurement, Record).
- Числовые поля в completion.context и measurements[i] приводятся к float через pandas.to_numeric, NaN→None.
- Удаляет служебные строки и email из prompt, нормализует Unicode, схлопывает пробелы.
- Если prompt > max_tokens (tiktoken), строка пропускается (логируется причина).
- Дедупликация по SHA256(prompt+domain).
- Сохраняет чистый JSONL с сохранением ключей и порядка.
- В конце — статистика: сколько строк оставлено, сколько дубликатов, сколько невалидных.

Зависимости: pandas, pydantic>=2, typer, rich, tiktoken.
"""
import json
import re
import unicodedata
import hashlib
from typing import Optional, List, Dict, Any
from collections import OrderedDict

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, create_model
import typer
from rich.progress import track
from rich.console import Console
import tiktoken
from langdetect import detect

console = Console()

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b", re.I)
DOI_RE = re.compile(r"\b10\.\d+\/[\w.]+\b")
ORCID_RE = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]")
COPYRIGHT_RE = re.compile(r"©|copyright", re.I)
SUPPORT_RE = re.compile(r"Supporting Information", re.I)
BOILERPLATE_RE = re.compile(r"^(View the article online|Original content|This content was downloaded).*", re.I)
WHITESPACE_RE = re.compile(r"\s+")

# ==== Pydantic Schemas ====
class BaseRecord(BaseModel):
    prompt: str
    completion: dict
    domain: str

domain_fields = {
    "magnetics": {"magnetic_field": (float, None), "temperature": (float, None)},
    "nanozymes": {"enzyme_type": (str, None)},
    "synergy": {"drug": (str, None), "effect": (str, None)},
}

def get_domain_model(domain):
    if domain in domain_fields:
        return create_model(f"{domain.capitalize()}Record", __base__=BaseRecord, **domain_fields[domain])
    return BaseRecord

class Context(BaseModel):
    CellAge: Optional[str] = None
    CellSource: Optional[str] = None
    CellTissue: Optional[str] = None
    CellType: Optional[str] = None
    CoatFunctionalGroup: Optional[str] = None
    HumanAnimal: Optional[str] = None
    Material: Optional[str] = None
    Shape: Optional[str] = None
    SizeInMediumNm: Optional[float] = None
    SurfaceCharge: Optional[str] = None
    # Можно добавить другие поля по необходимости

class Measurement(BaseModel):
    h: Optional[float] = None
    mgL: Optional[float] = None
    viab: Optional[float] = None
    # Можно добавить другие поля по необходимости

class Record(BaseModel):
    prompt: str
    response: Optional[str] = None
    domain: Optional[str] = None
    completion: Optional[Dict[str, Any]] = None

# ==== Cleaning Functions ====
def clean_prompt(text):
    lines = text.splitlines()
    lines = [l for l in lines if not BOILERPLATE_RE.match(l)]
    text = "\n".join(lines)
    text = EMAIL_RE.sub("", text)
    text = DOI_RE.sub("", text)
    text = ORCID_RE.sub("", text)
    text = COPYRIGHT_RE.sub("", text)
    text = SUPPORT_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()

def is_lang_ok(text, allowed_langs):
    try:
        lang = detect(text)
        return lang in allowed_langs
    except Exception:
        return False

def clean_floats(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float, str)):
                v2 = pd.to_numeric(v, errors="coerce")
                obj[k] = None if pd.isna(v2) else float(v2)
            elif isinstance(v, dict) or isinstance(v, list):
                obj[k] = clean_floats(v)
    elif isinstance(obj, list):
        return [clean_floats(x) for x in obj]
    return obj

def get_token_len(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_dedup_hash(prompt, domain, measurements):
    key = prompt + domain + json.dumps(measurements, sort_keys=True)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

# ==== Main ==== 
def main(
    in_: str = typer.Option(..., "--in", help="Путь к исходному JSONL"),
    out: str = typer.Option(..., "--out", help="Путь для сохранения очищенного JSONL"),
    max_tokens: int = typer.Option(12000, help="Максимум токенов в prompt")
):
    seen_hashes = set()
    kept, total, dups, invalid = 0, 0, 0, 0
    with open(in_, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        lines = fin.readlines()
        for line in track(lines, description="Cleaning JSONL..."):
            total += 1
            try:
                raw = json.loads(line)
            except Exception as e:
                invalid += 1
                console.log(f"[red]Invalid JSON: {e}")
                continue
            # Валидация и очистка
            try:
                rec = Record.model_validate(raw)
            except ValidationError as e:
                invalid += 1
                console.log(f"[red]Pydantic validation error: {e}")
                continue
            # Очистка prompt
            prompt = clean_prompt(rec.prompt)
            # Проверка длины prompt
            try:
                n_tokens = get_token_len(prompt)
            except Exception as e:
                invalid += 1
                console.log(f"[red]Tiktoken error: {e}")
                continue
            if n_tokens > max_tokens:
                invalid += 1
                console.log(f"[yellow]Skip: prompt tokens {n_tokens} > {max_tokens}")
                continue
            # Дедупликация
            domain = rec.domain or ""
            hashval = sha256(prompt + domain)
            if hashval in seen_hashes:
                dups += 1
                continue
            seen_hashes.add(hashval)
            # Очистка completion
            comp = rec.completion
            if comp:
                if "context" in comp:
                    comp["context"] = clean_floats(comp["context"])
                if "measurements" in comp and isinstance(comp["measurements"], list):
                    comp["measurements"] = [clean_floats(m) for m in comp["measurements"]]
            # Сборка результата с сохранением порядка
            out_obj = OrderedDict()
            for k in raw:
                if k == "prompt":
                    out_obj[k] = prompt
                elif k == "completion" and comp:
                    out_obj[k] = comp
                else:
                    out_obj[k] = raw[k]
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1
    console.rule("[bold green]Cleaning complete")
    console.print(f"[green]Kept: {kept} / {total}")
    console.print(f"[yellow]Duplicates: {dups}")
    console.print(f"[red]Invalid/skipped: {invalid}")

def export_jsonl(out, samples):
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"prompt": s["prompt"], "completion": s["completion"]}, ensure_ascii=False) + "\n")

def export_alpaca(out, samples):
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"instruction": s["prompt"], "input": "", "output": s["completion"]}, ensure_ascii=False) + "\n")

def export_openchat(out, samples):
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"USER: {s['prompt']}\nASSISTANT: {s['completion']}\n")

if __name__ == "__main__":
    import typer; typer.run(main) 