"""LLM-based caption matching for images and tables in OCR output."""

import os
import json
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def extract_tables(markdown: str) -> list:
    """Extract markdown tables with positions."""
    tables = []
    lines = markdown.split('\n')
    i = 0
    
    while i < len(lines):
        if '|' in lines[i] and i + 1 < len(lines) and '---' in lines[i + 1]:
            start = i
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            
            tables.append({
                'id': f'table-{len(tables)}',
                'markdown': '\n'.join(table_lines),
                'line_start': start,
                'line_end': i - 1
            })
        else:
            i += 1
    
    return tables


def match_captions(page_data: dict, client: OpenAI) -> dict:
    """Match images and tables with their captions using LLM."""
    images = page_data.get("images", [])
    markdown = page_data.get("markdown", "")
    tables = extract_tables(markdown)
    
    if not images and not tables:
        return {"image_groups": [], "image_individual": {}, "table_captions": {}, "tables": []}
    
    image_ids = [img["id"] for img in images]
    table_ids = [tbl["id"] for tbl in tables]
    
    table_contexts = []
    lines = markdown.split('\n')
    for tbl in tables:
        start = max(0, tbl['line_start'] - 5)
        end = min(len(lines), tbl['line_end'] + 5)
        context = '\n'.join(lines[start:end])[:500]
        table_contexts.append(f"{tbl['id']}: {context}")
    
    prompt = f"""Analyze images and tables with their captions on this scientific paper page.

Images: {', '.join(image_ids) if image_ids else 'none'}
Images in text are marked as: ![img-X.jpeg](img-X.jpeg)

Tables with context: {', '.join(table_ids) if table_ids else 'none'}
{chr(10).join(table_contexts) if table_contexts else ''}

Full page text:
{markdown}

TASK:

For IMAGES:
- GROUPED: Multiple images with labels (a), (b), (c) and ONE common description
- INDIVIDUAL: Each image has its own separate caption

For TABLES:
- Find caption near each table (search text before AND after the table)
- Caption usually: "TABLE X:", "Table X:", where X is number
- Copy FULL caption including everything after the colon
- If multiple sentences, include all of them

CRITICAL: 
- Copy FULL text WITHOUT summarization
- Search both before and after table markdown
- Empty string "" only if truly no caption exists

Return JSON:
{{
  "image_groups": [
    {{
      "images": ["img-id"],
      "common_caption": "text",
      "individual_captions": {{"img-id": "label"}}
    }}
  ],
  "image_individual": {{"img-id": "text"}},
  "table_captions": {{"table-id": "text"}}
}}"""

    completion = client.chat.completions.create(
        model=os.getenv("CAPTION_MATCHER_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    result = json.loads(completion.choices[0].message.content)
    result['tables'] = tables
    return result


def process_ocr_result(ocr_result: dict, api_key: Optional[str] = None) -> dict:
    """
    Enrich OCR result with matched captions for images and tables.
    
    Args:
        ocr_result: Raw Mistral OCR output
        api_key: OpenRouter API key (optional, uses env var if not provided)
        
    Returns:
        Enriched OCR result with captions
    """
    client = OpenAI(
        api_key=api_key or os.getenv("OPENROUTER_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    )
    
    enriched = ocr_result.copy()
    
    for page in enriched.get("pages", []):
        markdown = page.get("markdown", "")
        tables = extract_tables(markdown)
        
        if not page.get("images") and not tables:
            continue
        
        try:
            result = match_captions(page, client)
            
            for group in result.get("image_groups", []):
                common = group.get("common_caption", "")
                for img in page.get("images", []):
                    if img["id"] in group.get("images", []):
                        img["caption"] = group.get("individual_captions", {}).get(img["id"], "")
                        img["figure_caption"] = common
            
            for img in page.get("images", []):
                if img["id"] in result.get("image_individual", {}):
                    img["caption"] = result["image_individual"][img["id"]]
            
            page["tables"] = []
            for tbl in result.get("tables", []):
                if tbl["id"] in result.get("table_captions", {}):
                    tbl["caption"] = result["table_captions"][tbl["id"]]
                page["tables"].append(tbl)
        except Exception:
            continue
    
    return enriched
