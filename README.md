# Evaluation of Large Language Models for Nanomaterials Data Extraction

## Abstract
Large language models can automate knowledge extraction, but generic systems struggle with the complexities of nanomaterials research. We intro005 duce MINION, a specialized framework for extracting information from full-text papers in this field. Combining vision and natural language pro008 cessing with a dedicated knowledge base, MIN009 ION converts papers into structured records. It effectively links figures to text, resolves complex chemical entities, and captures critical experimen012 tal details often overlooked by general models. A qualitative evaluation on the CHEMX dataset shows that MINION achieves significantly more accurate and comprehensive extractions, underscoring the need for domain-specific architectures in scientific information mining.

## Project Overview

**MINION** is a multi-agent system for extracting structured data from nanomaterials research articles. The system includes:
- **Vision agent**: Extracts and analyzes figures, tables, and images from PDFs (YOLO + GPT-4o).
- **Main LLM agent**: Extracts parameters from article text (LLM inference, structured output).
- **NER agent**: Extracts entities and parameters using a specialized SFT NER agent.

The project supports the full pipeline: data preparation, training, inference, and model comparison.

## Repository Structure
- `graph_processing/` — Vision agent: PDF processing, figure/table extraction, YOLO & OpenAI integration.
- `structured_output/` — Main LLM agent: structured extraction from text.
- `ner_data_prep/` — NER agent: data preparation, training, and inference (SFT).
- `data/` — Results, benchmarks, and auxiliary files.

## NER Agent: Data Preparation, Training, and Inference

### Data Preparation

1. Collect and split data by domain:
   ```bash
   python ner_data_prep/scripts/01_collect_and_split.py ner_data_prep/data ner_data_prep/datasplits
   ```
2. Clean and validate data:
   ```bash
   python ner_data_prep/scripts/02_clean_jsonl.py --in ner_data_prep/datasplits/train.jsonl --out ner_data_prep/datasplits/clean_train.jsonl --max_tokens 12000
   # Repeat for val/test
   ```
3. Convert to dialogue format:
   ```bash
   python ner_data_prep/scripts/03.py ner_data_prep/datasplits/clean_train.jsonl ner_data_prep/datasplits/clean_train_converted.jsonl
   # Repeat for val/test
   ```
4. Create HuggingFace Dataset:
   ```bash
   python ner_data_prep/scripts/04.py ner_data_prep/datasplits/clean_train_converted.jsonl ner_data_prep/datasplits/clean_val_converted.jsonl ner_data_prep/datasplits/clean_test_converted.jsonl ner_data_prep/hf_dataset
   ```

### Training

For training, use [effective_llm_alignment](https://github.com/VikhrModels/effective_llm_alignment) — configure your YAML and specify the path to your HF Dataset.

### Inference

To run inference with trained NER models:
```bash
python ner_data_prep/scripts/infer_ner.py
```

The script runs both model variants (Qwen1.5-7B and Llama-3.1-8B) on all splits of the `zjkarina/nanoMINER_test` dataset and saves results to separate `.jsonl` files.

## Vision Agent (`graph_processing/`)
- Extracts images from PDFs, detects figures/tables (YOLO), analyzes with GPT-4o.
- See `graph_processing/README.md` for details.

## Main LLM Agent (`structured_output/`)
- Extracts nanomaterial parameters from article text.
- Structured output, supports various models (Llama, Qwen, Mistral, etc.).
- Examples and instructions in `structured_output/`.

## Requirements
- Python 3.10+
- torch, transformers, datasets, ultralytics, fitz, pillow, rich, typer, tiktoken, and more.
- For NER agent: see `ner_data_prep/requirements.txt`
- For vision agent: see `graph_processing/README.md`

## License & Contact
- License: MIT
- Questions: [github.com/zjkarina/nanoMINER](https://github.com/zjkarina/nanoMINER)