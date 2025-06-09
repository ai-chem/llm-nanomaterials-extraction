from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
import json

# === Модели для инференса ===
model_paths = [
    "zjkarina/nanoMINER_sft-Qwen1.5-7B-unsloth-full",
    "zjkarina/nanoMINER_sft-Llama-3.1-8B-unsloth-full"
]
dataset_path = "zjkarina/nanoMINER_test"
generation_kwargs = dict(
    max_new_tokens=1024,
    temperature=0.3,
    do_sample=False,
    top_p=0.95,
)

def run_inference(model_path, dataset_path):
    print(f"\n=== MODEL: {model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).cuda().eval()

    dataset = load_dataset(dataset_path)

    for split_name, split_data in dataset.items():
        print(f"\n=== Split: {split_name} ===")
        results = []

        for sample in tqdm(split_data):
            conversation = sample.get("conversation", [])
            if isinstance(conversation, list) and conversation:
                prompt_text = conversation[0].get("content", "")
            else:
                prompt_text = ""

            prompt = f"Extract nanozyme data from the article. <doc> {prompt_text} </doc> ###"
            message = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(message, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs,
                    attention_mask=(inputs != tokenizer.pad_token_id).long(),
                    **generation_kwargs
                )

            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Обрезаем до ответа ассистента, если есть
            if "assistant" in decoded:
                clean_response = decoded.split("assistant", 1)[1].strip()
            else:
                clean_response = decoded.strip()

            results.append({
                "domain": split_name,
                "fname": sample.get("fname"),
                "input_text": conversation[-1] if conversation else "",
                "output_text": clean_response
            })

        fname = f"nanoMINER_{model_path.split('/')[-1]}_{split_name}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f">>> Saved {len(results)} results to: {fname}")

if __name__ == "__main__":
    for model_path in model_paths:
        run_inference(model_path, dataset_path) 