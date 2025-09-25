"""
Enhanced image reasoning module with support for multiple VLM providers
Supports Gemma3-27b, GLM 4.1, and GPT-4o with automatic fallback
"""

import base64
import io
import os
from typing import Dict, Tuple, Optional, List, Union
import fitz
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import json
import logging
from vlm_config import VLMProvider, vlm_manager

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

class ConcentrationData(BaseModel):
    reaction_type: str = Field(description="Type of reaction (e.g. TMB+H2O2, H2O2+TMB)")
    c_min: float = Field(description="Minimum concentration value in mM")
    c_max: float = Field(description="Maximum concentration value in mM")
    co_substrate_concentration: Optional[float] = Field(None, description="Concentration of co-substrate in mM if specified")

class KineticParameters(BaseModel):
    km: Optional[float] = Field(None, description="Michaelis constant Km in mM")
    vmax: Optional[float] = Field(None, description="Maximum reaction rate Vmax in mM/s")
    kcat: Optional[float] = Field(None, description="Turnover number kcat in s^-1")

class NanozymeProperties(BaseModel):
    formula: Optional[str] = Field(None, description="Chemical formula of the nanozyme")
    activity: Optional[str] = Field(None, description="Type of activity (peroxidase, oxidase, etc.)")
    syngony: Optional[str] = Field(None, description="Crystal system")
    size: Optional[Dict[str, float]] = Field(None, description="Size parameters in nm (length, width, depth or diameter)")
    surface_chemistry: Optional[str] = Field(None, description="Surface modification")

class ImageAnalysis(BaseModel):
    image_type: str = Field(description="Type of image (concentration_graph)")
    nanozyme_properties: Optional[NanozymeProperties] = Field(None, description="Properties of nanozyme if mentioned")
    concentration_data: Optional[List[ConcentrationData]] = Field(None, description="Concentration data if present")
    kinetic_parameters: Optional[List[KineticParameters]] = Field(None, description="Kinetic parameters if present")
    description: str = Field(description="Brief description of what was found in the image")


def extract_concentration_range_with_vlm(client, config, base64_image: str, example_base64: str) -> ImageAnalysis:
    """
    Core function to analyze image with specific VLM client.
    Handles both structured and non-structured output based on provider capabilities.
    """
    system_prompt = f"""
    Analyze the image and extract information about nanozyme properties, concentrations and kinetic parameters.
    
    Here is an example of a concentration graph that you should look for:
    <image>data:image/jpeg;base64,{example_base64}</image>
    
    Pay attention to:
    1. Type of image (concentration_graph)
    2. For concentration graphs:
       - y-axis is velocity (v)
       - x-axis is concentration (C), units are mM, µM, nM, etc.
       - Look for actual experimental points (dots/squares with error bars)
       - Find leftmost (C_min) and rightmost (C_max) points on concentration axis
       - Identify reaction type and co-substrate concentration
       - Check if points at x=0 are present
    
    IGNORE OTHER TYPES OF GRAPHS!
    Ignore also:
    - Lineweaver-Burk plots (1/v vs 1/[S])
    - Non-kinetic data
    - Images without nanozyme-related information
    
    Return the analysis as a JSON object with the following structure:
    {{
        "image_type": "concentration_graph or other",
        "nanozyme_properties": {{
            "formula": "chemical formula or null",
            "activity": "activity type or null",
            "syngony": "crystal system or null",
            "size": {{"diameter": value}} or null,
            "surface_chemistry": "surface modification or null"
        }} or null,
        "concentration_data": [
            {{
                "reaction_type": "reaction type",
                "c_min": minimum_concentration,
                "c_max": maximum_concentration,
                "co_substrate_concentration": value or null
            }}
        ] or null,
        "kinetic_parameters": [
            {{
                "km": value or null,
                "vmax": value or null,
                "kcat": value or null
            }}
        ] or null,
        "description": "brief description of findings"
    }}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Analyze this image and extract all relevant information about nanozymes. Return as JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    try:
        if config.supports_structured_output:
            # Use structured output for providers that support it
            completion = client.beta.chat.completions.parse(
                model=config.model_name,
                messages=messages,
                response_format=ImageAnalysis,
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
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response (sometimes wrapped in markdown code blocks)
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to parse the entire content as JSON
                    json_str = content
                
                data = json.loads(json_str)
                
                # Convert dict to ImageAnalysis object
                return ImageAnalysis(**data)
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                # Return basic analysis with the raw response
                return ImageAnalysis(
                    image_type="unparsed",
                    description=f"Raw response: {content[:500]}"
                )
                
    except Exception as e:
        logger.error(f"Error in VLM analysis: {str(e)}")
        raise


def extract_concentration_range(image, provider: Optional[VLMProvider] = None) -> ImageAnalysis:
    """
    Analyze image and extract structured information about nanozyme properties, concentrations and kinetic parameters.
    
    Args:
        image: Either a PDF path string or PIL Image object
        provider: Optional VLM provider to use. If None, uses fallback strategy.
    
    Returns:
        ImageAnalysis object with extracted information
    """
    
    # Load and encode example image
    example_path = "graph_processing/conc_example.jpg"
    if not os.path.exists(example_path):
        # Try alternative path
        example_path = "conc_example.jpg"
    
    try:
        with open(example_path, "rb") as image_file:
            example_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.warning(f"Example image not found at {example_path}, proceeding without example")
        example_base64 = ""
    
    # Convert image to base64
    if isinstance(image, str):
        base64_image = pdf_page_to_base64(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # If specific provider requested, use it directly
    if provider:
        client, config = vlm_manager.get_client(provider)
        return extract_concentration_range_with_vlm(client, config, base64_image, example_base64)
    
    # Otherwise use fallback strategy
    def analyze_with_client(client, config):
        return extract_concentration_range_with_vlm(client, config, base64_image, example_base64)
    
    try:
        return vlm_manager.execute_with_fallback(analyze_with_client)
    except Exception as e:
        logger.error(f"All VLM providers failed: {str(e)}")
        return ImageAnalysis(
            image_type="error",
            description=f"Failed to analyze image: {str(e)}"
        )


def extract_table_markdown_with_vlm(client, config, base64_image: str) -> Optional[str]:
    """
    Core function to convert table image to markdown with specific VLM client.
    """
    system_prompt = """
    You are a specialized assistant that converts tables from images into markdown format.
    - Create proper markdown tables with aligned columns
    - Preserve all headers and data exactly as shown
    - Include any table captions or notes
    - Maintain units and formatting
    - If the image is not a table, return "NOT_A_TABLE"
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Convert this table to markdown format."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
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
        logger.error(f"Failed to convert table: {str(e)}")
        raise


def extract_table_markdown(image, provider: Optional[VLMProvider] = None) -> Optional[str]:
    """
    Convert table image to markdown format using VLM.
    
    Args:
        image: Either a PDF path string or PIL Image object
        provider: Optional VLM provider to use. If None, uses fallback strategy.
    
    Returns:
        Markdown string or None if conversion failed
    """
    # Convert image to base64
    if isinstance(image, str):
        base64_image = pdf_page_to_base64(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # If specific provider requested, use it directly
    if provider:
        client, config = vlm_manager.get_client(provider)
        return extract_table_markdown_with_vlm(client, config, base64_image)
    
    # Otherwise use fallback strategy
    def convert_with_client(client, config):
        return extract_table_markdown_with_vlm(client, config, base64_image)
    
    try:
        return vlm_manager.execute_with_fallback(convert_with_client)
    except Exception as e:
        logger.error(f"All VLM providers failed for table conversion: {str(e)}")
        return None


# Maintain backward compatibility by keeping original function names as aliases
def extract_concentration_range_legacy(image) -> ImageAnalysis:
    """Legacy function for backward compatibility"""
    return extract_concentration_range(image)


def extract_table_markdown_legacy(image) -> Optional[str]:
    """Legacy function for backward compatibility"""
    return extract_table_markdown(image)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    print("VLM-enhanced image reasoning module loaded successfully")