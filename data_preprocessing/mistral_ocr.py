"""Mistral OCR API wrapper for PDF text extraction."""

import os
import json
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral, DocumentURLChunk

load_dotenv()

DEFAULT_MISTRAL_MODEL = "mistral-ocr-latest"


def run_ocr(pdf_path: Path, api_key: Optional[str] = None) -> dict:
    """
    Process PDF file with Mistral OCR API.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Mistral API key (optional, uses env var if not provided)
        
    Returns:
        OCR result as dictionary with pages, images, and markdown text
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
    
    return json.loads(response.model_dump_json())


def extract_text_with_markers(ocr_result: dict) -> str:
    """
    Extract text from OCR result with image markers.
    
    Args:
        ocr_result: Mistral OCR result dictionary
        
    Returns:
        Text with image markers in format [IMAGE: id at (x1,y1)-(x2,y2)]
    """
    pages = []
    
    for page in ocr_result.get("pages", []):
        markdown = page.get("markdown", "")
        
        for img in page.get("images", []):
            img_id = img.get("id", "")
            coords = f"({img['top_left_x']},{img['top_left_y']})-({img['bottom_right_x']},{img['bottom_right_y']})"
            pattern = f"!\\[{img_id}\\]\\({img_id}\\)"
            markdown = re.sub(pattern, f"[IMAGE: {img_id} at {coords}]", markdown)
        
        pages.append(markdown)
    
    return "\n\n".join(pages)
