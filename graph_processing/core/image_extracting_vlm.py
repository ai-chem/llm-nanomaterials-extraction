"""
Enhanced image extraction module with VLM and dataset-specific support
Uses new VLM providers (Gemma3-27b, GLM 4.1) with dataset-specific prompts
"""

import os
import fitz
import torch
from PIL import Image
from typing import List, Dict, Union, Optional
from ultralytics import YOLO
import logging
from dotenv import load_dotenv
from pathlib import Path

# Import the dataset-specific reasoning module
from image_reasoning_dataset import analyze_image_for_dataset, extract_table_dataset_specific
from dataset_prompts import map_folder_to_dataset_type
from vlm_config import VLMProvider

load_dotenv()
logger = logging.getLogger(__name__)


def extract_image_pages(pdf_path: str) -> List[int]:
    """Extract page numbers that contain images from a PDF."""
    pdf_document = fitz.open(pdf_path)
    image_pages = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images()
        if image_list:
            image_pages.append(page_num + 1)  # Convert to 1-indexed
    
    return image_pages


def process_images_with_yolo(images: List[Image.Image], model_path: str) -> tuple[List[Image.Image], List[Image.Image]]:
    """
    Process images with YOLO model to detect and classify graphs and tables.
    
    Returns:
        Tuple of (concentration_graphs, tables)
    """
    try:
        # Load YOLO model
        model = YOLO(model_path)
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Run inference on all images
        results = model(images, device=device)
        
        concentration_graphs = []
        tables = []
        
        # Process each result
        for idx, result in enumerate(results):
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the original image
                orig_img = images[idx]
                
                # Process each detection
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Crop the image
                    cropped = orig_img.crop((int(x1), int(y1), int(x2), int(y2)))
                    
                    # Get class (assuming class 1 is table, others are graphs)
                    cls = int(box.cls[0])
                    
                    if cls == 1:  # Table class
                        tables.append(cropped)
                    else:  # Graph classes
                        concentration_graphs.append(cropped)
        
        logger.info(f"Detected {len(concentration_graphs)} graphs and {len(tables)} tables")
        return concentration_graphs, tables
        
    except Exception as e:
        logger.error(f"Error in YOLO processing: {str(e)}")
        return [], []


def pdf_analysis(pdf_path: str, yolo_model_path: str = None, vlm_provider: Optional[VLMProvider] = None, dataset_name: str = None) -> Dict:
    """
    Analyze PDF to extract dataset-specific information from images using VLM.
    
    Args:
        pdf_path: Path to PDF file
        yolo_model_path: Path to YOLO model (optional, uses env var if not provided)
        vlm_provider: Optional VLM provider to use (defaults to Gemma3-27b)
        dataset_name: Name of dataset for specialized extraction (cytox, magnetic, nanozymes, etc.)
    
    Returns:
        Dictionary containing analyses and tables
    """
    if yolo_model_path is None:
        yolo_model_path = os.getenv('YOLO_PATH', 'graph_processing/best.pt')
    
    logger.info(f"Starting PDF analysis: {pdf_path}")
    logger.info(f"Using YOLO model: {yolo_model_path}")
    
    # Try to infer dataset name from path if not provided
    if dataset_name is None:
        pdf_path_obj = Path(pdf_path)
        # Try to find dataset name in path
        for part in pdf_path_obj.parts:
            if any(ds in part.lower() for ds in ['cytox', 'magnetic', 'nanozyme', 'seltox', 'synergy']):
                dataset_name = part.lower()
                break
        
        if dataset_name is None:
            dataset_name = 'nanozymes'  # Default
            logger.warning(f"Could not infer dataset type, using default: {dataset_name}")
        else:
            logger.info(f"Inferred dataset type: {dataset_name}")
    
    # Map to dataset type
    dataset_type = map_folder_to_dataset_type(dataset_name)
    logger.info(f"Using dataset type: {dataset_type.value}")
    
    # Extract pages with images
    image_pages = extract_image_pages(pdf_path)
    if not image_pages:
        logger.warning(f"No images found in PDF: {pdf_path}")
        return {"analyses": [], "tables": [], "dataset_type": dataset_type.value}
    
    logger.info(f"Found images on {len(image_pages)} pages")
    
    # Convert PDF pages to images
    pdf_document = fitz.open(pdf_path)
    images = []
    
    for page_num in image_pages:
        page = pdf_document[page_num - 1]  # Convert to 0-indexed
        pix = page.get_pixmap(dpi=300)  # High resolution for better detection
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    # Process with YOLO
    concentration_graphs, table_images = process_images_with_yolo(images, yolo_model_path)
    
    # Analyze graphs with dataset-specific VLM
    analyses = []
    for graph in concentration_graphs:
        try:
            # Use dataset-specific analysis
            analysis = analyze_image_for_dataset(
                graph, 
                dataset_name=dataset_name,
                provider=vlm_provider
            )
            
            # Check if we got valid data
            if analysis and 'description' in analysis:
                analyses.append(analysis)
                logger.info(f"✓ Extracted {dataset_type.value} data: {analysis.get('description', '')[:100]}")
            
        except Exception as e:
            logger.error(f"Error analyzing graph: {str(e)}")
    
    # Convert tables to markdown with dataset-specific focus
    tables = []
    for table_img in table_images:
        try:
            markdown = extract_table_dataset_specific(
                table_img, 
                dataset_type=dataset_type,
                provider=vlm_provider
            )
            if markdown:
                tables.append(markdown)
                logger.info(f"✓ Converted table to markdown ({len(markdown)} chars)")
        except Exception as e:
            logger.error(f"Error converting table: {str(e)}")
    
    result = {
        "analyses": analyses,
        "tables": tables,
        "dataset_type": dataset_type.value,
        "extraction_stats": {
            "pages_with_images": len(image_pages),
            "graphs_detected": len(concentration_graphs),
            "tables_detected": len(table_images),
            "graphs_analyzed": len(analyses),
            "tables_converted": len(tables)
        }
    }
    
    logger.info(f"Analysis complete: {len(analyses)} graphs, {len(tables)} tables")
    return result


# Maintain backward compatibility
def pdf_analysis_legacy(pdf_path: str, yolo_model_path: str = None) -> Dict:
    """Legacy function for backward compatibility"""
    return pdf_analysis(pdf_path, yolo_model_path)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample PDF
    import sys
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
        result = pdf_analysis(test_pdf, vlm_provider=VLMProvider.GEMMA3_27B)
        print(f"Results: {len(result['analyses'])} analyses, {len(result['tables'])} tables")