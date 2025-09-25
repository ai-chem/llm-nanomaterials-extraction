"""
Dataset-specific image reasoning module with VLM support
Uses specialized prompts and models for each dataset type
"""

import base64
import io
import os
from typing import Dict, Optional, List, Union, Any
import fitz
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import json
import logging
import re

try:
    from graph_processing.vlm_config import VLMProvider, vlm_manager
    from graph_processing.dataset_prompts import (
        DatasetType, 
        get_dataset_prompt, 
        get_dataset_model,
        map_folder_to_dataset_type,
        NanozymesAnalysis,
        CytotoxicityData,
        MagneticData,
        SelectiveToxicityData,
        SynergyData
    )
except ImportError:
    from vlm_config import VLMProvider, vlm_manager
    from dataset_prompts import (
        DatasetType, 
        get_dataset_prompt, 
        get_dataset_model,
        map_folder_to_dataset_type,
        NanozymesAnalysis,
        CytotoxicityData,
        MagneticData,
        SelectiveToxicityData,
        SynergyData
    )

load_dotenv(override=True)
logger = logging.getLogger(__name__)


def pdf_page_to_base64(pdf_path: str, page_number: int) -> str:
    """Convert PDF page to base64 string."""
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_dataset_specific_data(
    image: Union[str, Image.Image],
    dataset_type: DatasetType,
    provider: Optional[VLMProvider] = None
) -> BaseModel:
    """
    Extract dataset-specific information from an image using VLM.
    
    Args:
        image: Either a PDF path string or PIL Image object
        dataset_type: Type of dataset (determines prompt and model)
        provider: Optional VLM provider to use
    
    Returns:
        Dataset-specific Pydantic model with extracted information
    """
    
    # Get dataset-specific configuration
    prompt = get_dataset_prompt(dataset_type)
    model_class = get_dataset_model(dataset_type)
    
    # Convert image to base64
    if isinstance(image, str):
        base64_image = pdf_page_to_base64(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Build message
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f"Analyze this image and extract {dataset_type.value} parameters. Return as JSON."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]
    
    # Get VLM client
    if provider:
        client, config = vlm_manager.get_client(provider)
        return _extract_with_client(client, config, messages, model_class, dataset_type)
    
    # Use fallback strategy
    def analyze_with_client(client, config):
        return _extract_with_client(client, config, messages, model_class, dataset_type)
    
    try:
        return vlm_manager.execute_with_fallback(analyze_with_client)
    except Exception as e:
        logger.error(f"All VLM providers failed: {str(e)}")
        # Return empty model with error description
        return model_class(
            description=f"Failed to analyze image: {str(e)}"
        )


def _extract_with_client(
    client: OpenAI,
    config: Any,
    messages: List[Dict],
    model_class: BaseModel,
    dataset_type: DatasetType
) -> BaseModel:
    """
    Core extraction function with specific VLM client.
    """
    try:
        if config.supports_structured_output:
            # Use structured output for providers that support it
            completion = client.beta.chat.completions.parse(
                model=config.model_name,
                messages=messages,
                response_format=model_class,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            return completion.choices[0].message.parsed
        else:
            # Use regular completion and parse JSON manually
            response = client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object in the response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content
            
            try:
                data = json.loads(json_str)
                return model_class(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON for {dataset_type.value}: {e}")
                # Try to extract key values using regex as fallback
                return _extract_values_fallback(content, model_class, dataset_type)
                
    except Exception as e:
        logger.error(f"Error in VLM analysis for {dataset_type.value}: {str(e)}")
        raise


def _extract_values_fallback(content: str, model_class: BaseModel, dataset_type: DatasetType) -> BaseModel:
    """
    Fallback extraction using regex patterns for common parameters.
    """
    extracted = {}
    
    # Common patterns for different parameter types
    patterns = {
        # Nanozymes patterns
        'c_min': r'c_min[:\s]*([0-9.]+)',
        'c_max': r'c_max[:\s]*([0-9.]+)',
        'km_value': r'[Kk]m[:\s]*([0-9.]+)',
        'vmax_value': r'[Vv]max[:\s]*([0-9.]+(?:[eE][+-]?[0-9]+)?)',
        
        # Cytotoxicity patterns
        'size_in_medium_nm': r'size.*?([0-9.]+)\s*nm',
        'zeta_in_medium_mv': r'zeta.*?([+-]?[0-9.]+)\s*m[Vv]',
        'ic50_value': r'IC50[:\s]*([0-9.]+)',
        
        # Magnetic patterns
        'squid_temperature': r'temperature[:\s]*([0-9.]+)\s*[Kk]',
        'hc_kOe': r'[Hh]c[:\s]*([0-9.]+)\s*k[Oo]e',
        'ms_emu_g': r'[Mm]s[:\s]*([0-9.]+)\s*emu',
        'mri_r1': r'r1[:\s]*([0-9.]+)',
        'mri_r2': r'r2[:\s]*([0-9.]+)',
        
        # Selective toxicity patterns
        'np_size_avg_nm': r'size.*?([0-9.]+)\s*nm',
        'zeta_potential_mV': r'zeta.*?([+-]?[0-9.]+)\s*m[Vv]',
        'selectivity_index': r'selectivity.*?([0-9.]+)',
        
        # Synergy patterns
        'NP_size_avg_nm': r'size.*?([0-9.]+)\s*nm',
        'combination_index': r'CI[:\s]*([0-9.]+)',
        'synergy_score': r'synergy.*?([0-9.]+)'
    }
    
    # Extract values using patterns
    for field, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                extracted[field] = value
            except ValueError:
                pass
    
    # Add description
    extracted['description'] = f"Extracted from text (fallback): {content[:200]}"
    
    # Create model instance with extracted values
    try:
        return model_class(**extracted)
    except Exception as e:
        logger.warning(f"Fallback extraction failed: {e}")
        return model_class(description="Failed to extract parameters")


def extract_table_dataset_specific(
    image: Union[str, Image.Image],
    dataset_type: DatasetType,
    provider: Optional[VLMProvider] = None
) -> Optional[str]:
    """
    Convert table image to markdown with dataset-specific focus.
    """
    # Get dataset-specific table prompt
    table_prompt = f"""
    Convert this table to markdown format.
    
    Focus on {dataset_type.value} related parameters:
    {_get_table_focus(dataset_type)}
    
    - Create proper markdown tables with aligned columns
    - Preserve all headers and data exactly as shown
    - Include units in headers or cells
    - If not a relevant table, return "NOT_A_TABLE"
    """
    
    # Convert image to base64
    if isinstance(image, str):
        base64_image = pdf_page_to_base64(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    messages = [
        {"role": "system", "content": table_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Convert this table to markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    # Get VLM client and process
    if provider:
        client, config = vlm_manager.get_client(provider)
        try:
            response = client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                temperature=0,
                max_tokens=2000
            )
            result = response.choices[0].message.content
            return None if result == "NOT_A_TABLE" else result
        except Exception as e:
            logger.error(f"Failed to convert table: {e}")
            return None
    
    # Use fallback strategy
    def convert_with_client(client, config):
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=0,
            max_tokens=2000
        )
        result = response.choices[0].message.content
        return None if result == "NOT_A_TABLE" else result
    
    try:
        return vlm_manager.execute_with_fallback(convert_with_client)
    except Exception as e:
        logger.error(f"All VLM providers failed for table conversion: {e}")
        return None


def _get_table_focus(dataset_type: DatasetType) -> str:
    """Get table extraction focus for each dataset type."""
    focus_map = {
        DatasetType.NANOZYMES: "Km, Vmax, kcat values with units",
        DatasetType.CYTOTOXICITY: "IC50, cell viability, size, zeta potential",
        DatasetType.MAGNETIC: "Ms, Hc, temperature, r1/r2 relaxivity",
        DatasetType.SELECTIVE_TOXICITY: "IC50 for cancer/normal cells, selectivity index",
        DatasetType.SYNERGY: "Combination index, drug concentrations, synergy scores"
    }
    return focus_map.get(dataset_type, "all relevant parameters")


# Convenience function for pipeline integration
def analyze_image_for_dataset(
    image: Union[str, Image.Image],
    dataset_name: str,
    provider: Optional[VLMProvider] = None
) -> Dict:
    """
    Main function to analyze image based on dataset name.
    
    Args:
        image: Image to analyze
        dataset_name: Name of dataset (cytox, magnetic, nanozymes, etc.)
        provider: Optional VLM provider
    
    Returns:
        Dictionary with extracted data
    """
    # Map dataset name to type
    dataset_type = map_folder_to_dataset_type(dataset_name)
    
    # Extract data
    extracted_data = extract_dataset_specific_data(image, dataset_type, provider)
    
    # Convert to dict
    result = extracted_data.dict()
    result['dataset_type'] = dataset_type.value
    
    return result


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    print("Dataset-specific image reasoning module loaded successfully")
    
    # Show available dataset types
    for dt in DatasetType:
        model = get_dataset_model(dt)
        print(f"\n{dt.value}:")
        print(f"  Model: {model.__name__}")
        print(f"  Fields: {list(model.__fields__.keys())}")