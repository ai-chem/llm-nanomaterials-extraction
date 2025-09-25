#!/usr/bin/env python3
"""
Qwen2.5-VL-72B API Endpoint for Scientific Parameter Extraction
With updated prompts supporting multiple experiments per PDF
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import base64
import json
from pathlib import Path
import fitz
from PIL import Image
import io
from openai import OpenAI
from datetime import datetime
import uvicorn

from vision_agent_prompts import DATASET_PROMPTS, DATASET_PARAMETERS
import os

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "qwen/qwen2.5-vl-72b-instruct"  

app = FastAPI(
    title="Qwen Scientific Extraction API",
    description="Extract parameters from scientific PDFs using Qwen2.5-VL-72B vision model",
    version="2.0"
)

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class ExtractionRequest(BaseModel):
    dataset_type: str
    max_pages: Optional[int] = 10
    page_numbers: Optional[List[int]] = None

class ExtractionResponse(BaseModel):
    status: str
    dataset: str
    pdf_name: str
    total_pages: int
    pages_processed: int
    experiments_found: int
    all_extracted_data: List[Dict[str, Any]]
    page_results: List[Dict[str, Any]]
    timestamp: str

def pdf_to_images(pdf_bytes: bytes, max_pages: Optional[int] = None,
                  page_numbers: Optional[List[int]] = None) -> List[Image.Image]:
    """Convert PDF bytes to images"""
    images = []
    pdf_stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    if page_numbers:
        # Process specific pages
        pages_to_process = [p - 1 for p in page_numbers if 0 <= p - 1 < len(doc)]
    else:
        # Process all or up to max_pages
        num_pages = len(doc)
        pages_to_process = range(min(num_pages, max_pages) if max_pages else num_pages)

    for page_num in pages_to_process:
        page = doc[page_num]
        # High quality: 2.5x resolution
        mat = fitz.Matrix(2.5, 2.5)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    doc.close()
    return images, len(doc)

def optimize_image(image: Image.Image, max_size: int = 2048) -> str:
    """Optimize and encode image to base64"""
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Encode to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", optimize=True, quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_from_image(image: Image.Image, dataset_type: str,
                       page_num: int, total_pages: int) -> Dict:
    """Extract parameters from a single page"""
    prompt = DATASET_PROMPTS.get(dataset_type, DATASET_PROMPTS['cytox'])

    # Add page context
    page_context = f"\n\nPAGE {page_num + 1} of {total_pages}"
    if page_num == 0:
        page_context += " (FIRST PAGE - likely contains title/abstract)"
    elif page_num == total_pages - 1:
        page_context += " (LAST PAGE - may contain conclusions/references)"

    full_prompt = prompt + page_context + """

CRITICAL: Return ONLY the JSON array, no explanations.
Extract ALL experiments/samples separately.
Return empty array [] if no relevant data on this page.
"""

    base64_image = optimize_image(image)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            temperature=0.1,
            max_tokens=1000,
        )

        result = response.choices[0].message.content

        # Parse response
        extracted = []
        if result:
            # Try to extract JSON array
            if '[' in result and ']' in result:
                json_str = result[result.find('['):result.rfind(']')+1]
                try:
                    extracted = json.loads(json_str)
                    if not isinstance(extracted, list):
                        extracted = [extracted]
                except:
                    pass
            # Fallback: try single object
            elif '{' in result and '}' in result:
                json_str = result[result.find('{'):result.rfind('}')+1]
                try:
                    obj = json.loads(json_str)
                    extracted = [obj]
                except:
                    pass

        return {
            "page": page_num + 1,
            "extracted": extracted,
            "raw_response": result[:500] if result else None,
            "success": True
        }

    except Exception as e:
        return {
            "page": page_num + 1,
            "extracted": [],
            "error": str(e)[:200],
            "success": False
        }

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Qwen Scientific Extraction API",
        "version": "2.0",
        "model": MODEL,
        "supported_datasets": list(DATASET_PROMPTS.keys()),
        "endpoints": {
            "/extract": "POST - Extract parameters from PDF",
            "/datasets": "GET - List supported datasets with expected parameters",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/datasets")
async def list_datasets():
    """List supported datasets and their parameters"""
    return {
        dataset: {
            "parameters": params,
            "description": DATASET_PROMPTS[dataset][:100] + "..."
        }
        for dataset, params in DATASET_PARAMETERS.items()
    }

@app.post("/extract", response_model=ExtractionResponse)
async def extract_parameters(
    file: UploadFile = File(...),
    dataset_type: str = "cytox",
    max_pages: Optional[int] = 10,
    page_numbers: Optional[str] = None
):
    """
    Extract scientific parameters from PDF

    Args:
        file: PDF file to process
        dataset_type: Type of dataset (cytox, synergy, magnetic, nanozymes, seltox)
        max_pages: Maximum number of pages to process (default: 10)
        page_numbers: Comma-separated page numbers to process (e.g., "1,3,5")

    Returns:
        Extracted parameters organized by page and experiment
    """

    # Validate dataset type
    if dataset_type not in DATASET_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset_type. Must be one of: {list(DATASET_PROMPTS.keys())}"
        )

    # Parse page numbers if provided
    pages_to_process = None
    if page_numbers:
        try:
            pages_to_process = [int(p.strip()) for p in page_numbers.split(',')]
        except:
            raise HTTPException(status_code=400, detail="Invalid page_numbers format")

    # Read PDF
    pdf_bytes = await file.read()

    try:
        # Convert to images
        images, total_pages = pdf_to_images(pdf_bytes, max_pages, pages_to_process)

        # Extract from each page
        page_results = []
        all_extracted = []

        for i, image in enumerate(images):
            page_num = pages_to_process[i] - 1 if pages_to_process else i
            result = extract_from_image(image, dataset_type, page_num, total_pages)
            page_results.append(result)

            # Aggregate all experiments
            if result.get("extracted"):
                for exp in result["extracted"]:
                    if isinstance(exp, dict):
                        # Add page reference
                        exp_with_page = exp.copy()
                        exp_with_page["source_page"] = page_num + 1
                        all_extracted.append(exp_with_page)

        # Count experiments with actual data
        experiments_with_data = 0
        for exp in all_extracted:
            # Check if has any non-null values besides metadata
            has_data = any(
                v is not None
                for k, v in exp.items()
                if k not in ['source_page', 'sample_id', 'substrate']
            )
            if has_data:
                experiments_with_data += 1

        return ExtractionResponse(
            status="success",
            dataset=dataset_type,
            pdf_name=file.filename,
            total_pages=total_pages,
            pages_processed=len(images),
            experiments_found=experiments_with_data,
            all_extracted_data=all_extracted,
            page_results=page_results,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QWEN SCIENTIFIC EXTRACTION API v2.0")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Supported datasets: {list(DATASET_PROMPTS.keys())}")
    print("\nFeatures:")
    print("✅ Multiple experiments per PDF")
    print("✅ Array responses for all datasets")
    print("✅ Concentration ranges (c_min, c_max) for nanozymes")
    print("✅ All missing parameters added")
    print("\nStarting server on http://localhost:8002")
    print("API docs available at: http://localhost:8002/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8002)