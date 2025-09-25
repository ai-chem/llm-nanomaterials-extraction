"""
Vision Agent API for parameter extraction from scientific PDFs
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import base64
import json
import io
import logging
from pathlib import Path
import tempfile
import fitz  # PyMuPDF
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import vision agent components
from vision_agent_prompts import (
    get_prompt_for_dataset,
    get_parameters_for_dataset,
    validate_parameter_value
)

# Import VLM configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from vlm_config import VLMProvider, vlm_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vision Agent API",
    description="Extract parameters from scientific PDFs using Qwen2.5-VL",
    version="1.0.0"
)

class ExtractionRequest(BaseModel):
    """Request model for parameter extraction"""
    dataset_type: str
    pdf_path: Optional[str] = None
    max_pages: Optional[int] = None
    
class ExtractionResponse(BaseModel):
    """Response model for extraction results - supports multiple experiments"""
    success: bool
    dataset_type: str
    extracted_values: Dict[str, Any]  # Deprecated - for backward compatibility
    extracted_experiments: List[Dict[str, Any]]  # New: array of experiments
    processing_time: float
    pages_processed: int
    total_experiments: int
    errors: List[str] = []

class VisionAgent:
    """Vision agent for parameter extraction"""
    
    def __init__(self):
        """Initialize the vision agent with VLM client"""
        try:
            self.vlm_client, self.config = vlm_manager.get_client(VLMProvider.QWEN_2_5_VL)
            logger.info("Vision Agent initialized with Qwen2.5-VL")
        except Exception as e:
            logger.error(f"Failed to initialize VLM client: {e}")
            raise
    
    def pdf_to_images(self, pdf_path: Path, max_pages: Optional[int] = None) -> List[Image.Image]:
        """Convert PDF pages to images"""
        images = []
        try:
            pdf = fitz.open(str(pdf_path))
            num_pages = min(len(pdf), max_pages) if max_pages else len(pdf)
            
            for page_num in range(num_pages):
                page = pdf[page_num]
                mat = fitz.Matrix(3.0, 3.0)  # High resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
                
            pdf.close()
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def encode_image(self, image: Image.Image) -> str:
        """Encode image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def extract_from_image(self, image: Image.Image, dataset_type: str, page_num: int, total_pages: int) -> List[Dict[str, Any]]:
        """Extract parameters from a single image - returns array of experiments"""
        try:
            # Get prompt for dataset type
            prompt = get_prompt_for_dataset(dataset_type)

            # Add page context
            page_context = f"\n\nPAGE {page_num + 1} of {total_pages}"
            if page_num == 0:
                page_context += " (FIRST PAGE - likely contains title/abstract)"
            elif page_num == total_pages - 1:
                page_context += " (LAST PAGE - may contain conclusions/references)"

            full_prompt = prompt + page_context + """\n\nCRITICAL: Return ONLY the JSON array, no explanations.
Extract ALL experiments/samples separately.
Return empty array [] if no relevant data on this page."""

            # Encode image
            image_base64 = self.encode_image(image)

            # Create message for VLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            # Call VLM with more tokens for array responses
            response = self.vlm_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )

            # Parse response
            content = response.choices[0].message.content
            extracted = []

            try:
                # Try to extract JSON array
                if '[' in content and ']' in content:
                    json_str = content[content.index('['):content.rindex(']')+1]
                    extracted = json.loads(json_str)
                    if not isinstance(extracted, list):
                        extracted = [extracted]
                # Fallback: try single object
                elif '{' in content and '}' in content:
                    json_str = content[content.index('{'):content.rindex('}')+1]
                    obj = json.loads(json_str)
                    extracted = [obj]
            except:
                logger.warning(f"Failed to parse JSON from response: {content[:200]}")

            return extracted

        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            return []
    
    def extract_from_pdf(self, pdf_path: Path, dataset_type: str, max_pages: Optional[int] = None) -> Dict[str, Any]:
        """Extract parameters from entire PDF - now returns multiple experiments"""
        start_time = time.time()

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, max_pages)
        total_pages = len(images)

        if not images:
            return {
                'success': False,
                'error': 'Failed to convert PDF to images',
                'extracted_values': {},  # Keep for backward compatibility
                'extracted_experiments': [],
                'pages_processed': 0,
                'total_experiments': 0
            }

        # Get expected parameters for dataset
        expected_params = get_parameters_for_dataset(dataset_type)
        all_experiments = []
        errors = []

        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_page = {
                executor.submit(self.extract_from_image, img, dataset_type, i, total_pages): i
                for i, img in enumerate(images)
            }

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    experiments = future.result()

                    # Add page reference and collect all experiments
                    for exp in experiments:
                        if isinstance(exp, dict):
                            # Add source page info
                            exp_with_page = exp.copy()
                            exp_with_page['source_page'] = page_num + 1

                            # Check if has meaningful data (not all null)
                            has_data = any(
                                v is not None
                                for k, v in exp.items()
                                if k not in ['source_page', 'sample_id', 'substrate']
                            )

                            if has_data:
                                all_experiments.append(exp_with_page)

                except Exception as e:
                    errors.append(f"Page {page_num + 1}: {str(e)}")

        # Create backward-compatible single value dict (from first experiment)
        extracted_values = {}
        if all_experiments:
            # Aggregate values from all experiments for backward compatibility
            for param in expected_params:
                values = [exp.get(param) for exp in all_experiments if exp.get(param) is not None]
                if values:
                    if all(isinstance(v, (int, float)) for v in values):
                        # For numeric values, take the median
                        extracted_values[param] = sorted(values)[len(values)//2]
                    else:
                        # For text values, take the most common
                        from collections import Counter
                        extracted_values[param] = Counter(values).most_common(1)[0][0]
                else:
                    extracted_values[param] = None
        else:
            extracted_values = {param: None for param in expected_params}

        processing_time = time.time() - start_time

        return {
            'success': True,
            'dataset_type': dataset_type,
            'extracted_values': extracted_values,  # Backward compatibility
            'extracted_experiments': all_experiments,  # New: all experiments
            'processing_time': processing_time,
            'pages_processed': len(images),
            'total_experiments': len(all_experiments),
            'errors': errors
        }

# Create global vision agent instance
vision_agent = VisionAgent()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vision Agent API is running",
        "endpoints": {
            "/extract": "POST - Extract parameters from PDF",
            "/extract_file": "POST - Upload and extract from PDF file",
            "/datasets": "GET - List available dataset types",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "Qwen2.5-VL-72B"}

@app.get("/datasets")
async def list_datasets():
    """List available dataset types and their parameters"""
    from vision_agent_prompts import DATASET_PARAMETERS
    return {
        "datasets": DATASET_PARAMETERS,
        "total": len(DATASET_PARAMETERS)
    }

@app.post("/extract", response_model=ExtractionResponse)
async def extract_parameters(request: ExtractionRequest):
    """Extract parameters from a PDF file"""
    try:
        # Validate dataset type
        if request.dataset_type not in ['synergy', 'cytox', 'magnetic', 'nanozymes', 'seltox']:
            raise HTTPException(status_code=400, detail=f"Invalid dataset type: {request.dataset_type}")
        
        # Check if PDF path exists
        if not request.pdf_path:
            raise HTTPException(status_code=400, detail="PDF path is required")
        
        pdf_path = Path(request.pdf_path)
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")
        
        # Extract parameters
        result = vision_agent.extract_from_pdf(
            pdf_path,
            request.dataset_type,
            request.max_pages
        )
        
        return ExtractionResponse(
            success=result['success'],
            dataset_type=result['dataset_type'],
            extracted_values=result['extracted_values'],  # Backward compatibility
            extracted_experiments=result.get('extracted_experiments', []),
            processing_time=result['processing_time'],
            pages_processed=result['pages_processed'],
            total_experiments=result.get('total_experiments', 0),
            errors=result.get('errors', [])
        )
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_file")
async def extract_from_upload(
    file: UploadFile = File(...),
    dataset_type: str = Form(...),
    max_pages: Optional[int] = Form(None)
):
    """Upload a PDF file and extract parameters"""
    try:
        # Validate dataset type
        if dataset_type not in ['synergy', 'cytox', 'magnetic', 'nanozymes', 'seltox']:
            raise HTTPException(status_code=400, detail=f"Invalid dataset type: {dataset_type}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Extract parameters
            result = vision_agent.extract_from_pdf(
                tmp_path,
                dataset_type,
                max_pages
            )
            
            return JSONResponse(content={
                "success": result['success'],
                "dataset_type": result['dataset_type'],
                "extracted_values": result['extracted_values'],  # Backward compatibility
                "extracted_experiments": result.get('extracted_experiments', []),
                "processing_time": result['processing_time'],
                "pages_processed": result['pages_processed'],
                "total_experiments": result.get('total_experiments', 0),
                "errors": result.get('errors', []),
                "filename": file.filename
            })
            
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Extraction from upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)