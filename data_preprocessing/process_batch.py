"""Batch processing of PDFs with Mistral OCR and caption matching."""

import os
import sys
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral, DocumentURLChunk

try:
    from data_preprocessing.caption_matcher import process_ocr_result
except ModuleNotFoundError:
    from caption_matcher import process_ocr_result

load_dotenv()

DEFAULT_MISTRAL_MODEL = "mistral-ocr-latest"


def process_pdf(pdf_path: str, output_dir: str, api_key: Optional[str] = None) -> Path:
    """
    Process single PDF and save enriched JSON result.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for JSON files
        api_key: Mistral API key (optional)
        
    Returns:
        Path to output JSON file
    """
    mistral_key = api_key or os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=mistral_key)
    pdf_file = Path(pdf_path)
    
    uploaded_file = client.files.upload(
        file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
        purpose="ocr"
    )
    
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    
    response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model=os.getenv("MISTRAL_OCR_MODEL", DEFAULT_MISTRAL_MODEL),
        include_image_base64=False
    )
    
    ocr_result = json.loads(response.model_dump_json())
    enriched = process_ocr_result(ocr_result)
    
    output_path = Path(output_dir) / f"{pdf_file.stem}_with_captions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    
    pdf_dir = sys.argv[1]
    output_dir = os.path.join(pdf_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    for pdf_file in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            process_pdf(str(pdf_file), output_dir)
        except Exception:
            continue


if __name__ == "__main__":
    main()
