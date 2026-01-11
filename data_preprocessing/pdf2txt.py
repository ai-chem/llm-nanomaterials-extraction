"""PDF to text extraction using Mistral OCR with caption matching."""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral, DocumentURLChunk

try:
    from data_preprocessing.caption_matcher import process_ocr_result
    from data_preprocessing.ocr_formatter import format_to_markdown, remove_references
except ModuleNotFoundError:
    from caption_matcher import process_ocr_result
    from ocr_formatter import format_to_markdown, remove_references

load_dotenv()

DEFAULT_MISTRAL_MODEL = "mistral-ocr-latest"


def extract_text_from_pdf(pdf_path: str, api_key: Optional[str] = None) -> str:
    """
    Extract text from PDF using Mistral OCR with caption matching.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Mistral API key (optional, uses env var if not provided)
        
    Returns:
        Clean markdown text with image and table captions
    """
    mistral_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("MISTRAL_API_KEY not found")
    
    client = Mistral(api_key=mistral_key)
    pdf_file = Path(pdf_path)
    
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
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
    enriched = process_ocr_result(ocr_result, api_key=os.getenv("OPENROUTER_KEY"))
    text = format_to_markdown(enriched)
    
    return remove_references(text)


def process_directory(input_dir: str, output_dir: str) -> tuple[list, list]:
    """
    Process all PDFs in directory.
    
    Args:
        input_dir: Directory with PDF files
        output_dir: Directory for output text files
        
    Returns:
        Tuple of (successful_files, failed_files)
    """
    success, failed = [], []
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".pdf"):
            continue
        
        try:
            text = extract_text_from_pdf(f"{input_dir}/{filename}")
            output_file = f"{output_dir}/{filename[:-4]}.txt"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            success.append(filename)
        except Exception as e:
            failed.append((filename, str(e)))
    
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs using Mistral OCR")
    parser.add_argument("--input_directory_path", type=str, required=True)
    parser.add_argument("--output_directory_path", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_directory_path, exist_ok=True)
    process_directory(args.input_directory_path, args.output_directory_path)


if __name__ == "__main__":
    main()
