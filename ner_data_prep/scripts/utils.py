import json

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def fix_content(item):
    for msg in item.get("conversation", []):
        if "content" not in msg:
            print("Нет content в сообщении:", msg)
        elif not isinstance(msg["content"], str):
            msg["content"] = json.dumps(msg["content"], ensure_ascii=False)
    return item

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n") 