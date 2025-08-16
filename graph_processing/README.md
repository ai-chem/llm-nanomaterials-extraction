# Vision Pipeline for Scientific Data Extraction

A comprehensive pipeline for extracting structured data from scientific publications using computer vision and language models.

## Overview

This pipeline processes PDF documents to extract scientific data (graphs, tables, parameters) using:
- **YOLO** for object detection (graphs and tables)
- **VLM (Vision Language Models)** for content analysis and parameter extraction
- **Dataset-specific prompts** for targeted extraction

## Features

- 📊 **Multi-dataset support**: Handles 5 different scientific data types
- 🎯 **Targeted extraction**: Dataset-specific prompts for precise parameter extraction
- 🔄 **Batch processing**: Process multiple PDFs in parallel
- 🔌 **Multiple VLM backends**: Support for various VLM providers
- 📈 **Structured output**: Returns data in JSON/Pydantic models

## Supported Datasets

1. **Nanozymes**: Kinetic parameters (Km, Vmax, kcat), concentration ranges
2. **Cytotoxicity**: IC50 values, cell viability, nanoparticle properties
3. **Magnetic**: SQUID measurements, MRI relaxivity, magnetic properties
4. **Selective Toxicity**: Differential toxicity between cancer/normal cells
5. **Synergy**: Combination indices, drug interaction parameters

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster YOLO processing)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd graph_processing
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

### Environment Variables (.env)

```bash
# VLM Configuration
GEMMA_API_URL=http://your-server/v1
GEMMA_API_KEY=your-api-key

# YOLO Model Path
YOLO_PATH=/path/to/best.pt

# Optional: OpenAI fallback
OPENAI_API_KEY=your-openai-key
```

### YOLO Model

Download or train a YOLO model for detecting graphs and tables in scientific documents. Place the weights file (e.g., `best.pt`) in the project directory.

## Usage

### Basic Usage

```python
from image_extracting_vlm import pdf_analysis
from vlm_config import VLMProvider

# Process a single PDF
result = pdf_analysis(
    "path/to/paper.pdf",
    yolo_model_path="best.pt",
    vlm_provider=VLMProvider.GEMMA3_27B,
    dataset_name="nanozymes"  # or: cytotoxicity, magnetic, etc.
)

print(f"Found {len(result['analyses'])} graphs")
print(f"Found {len(result['tables'])} tables")
```

### Batch Processing

```bash
# Process all datasets
python run_vision_pipeline.py

# Process specific datasets
python run_vision_pipeline.py --datasets nanozymes cytotoxicity --splits train

# Adjust parallel workers
python run_vision_pipeline.py --workers 4
```

### Using the Shell Script

```bash
# Process all train and test datasets
./run_all.sh all

# Process only training data
./run_all.sh train

# Process only test data
./run_all.sh test
```

## Project Structure

```
graph_processing/
├── core/
│   ├── vlm_config.py           # VLM provider configuration
│   ├── dataset_prompts.py      # Dataset-specific prompts and models
│   ├── image_reasoning_dataset.py  # Dataset-specific image analysis
│   └── image_extracting_vlm.py # Main PDF processing module
├── scripts/
│   ├── run_vision_pipeline.py  # Batch processing script
│   └── run_all.sh             # Shell wrapper for batch processing
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Output Format

The pipeline returns structured data in JSON format:

```json
{
  "dataset_type": "nanozymes",
  "analyses": [
    {
      "figure_id": "fig 3",
      "concentration_data": [
        {
          "reaction_type": "TMB+H2O2",
          "c_min": 0.1,
          "c_max": 1.3,
          "c_unit": "mM"
        }
      ],
      "kinetic_parameters": {
        "km_value": 0.45,
        "km_unit": "mM",
        "vmax_value": 0.032,
        "vmax_unit": "Ms-1"
      }
    }
  ],
  "tables": ["markdown formatted tables..."],
  "extraction_stats": {
    "pages_with_images": 4,
    "graphs_detected": 3,
    "tables_detected": 1,
    "graphs_analyzed": 3,
    "tables_converted": 1
  }
}
```

## API Reference

### Main Functions

#### `pdf_analysis(pdf_path, yolo_model_path, vlm_provider, dataset_name)`

Process a PDF file and extract scientific data.

**Parameters:**
- `pdf_path` (str): Path to PDF file
- `yolo_model_path` (str): Path to YOLO weights
- `vlm_provider` (VLMProvider): VLM backend to use
- `dataset_name` (str): Type of dataset (nanozymes, cytotoxicity, etc.)

**Returns:**
- Dictionary containing analyses, tables, and statistics

### Dataset Types

- `nanozymes`: Enzyme kinetics data
- `cytotoxicity`: Cell viability and toxicity
- `magnetic`: Magnetic properties
- `selective_toxicity`: Cancer vs normal cell toxicity
- `synergy`: Drug combination effects

## Development

### Adding New Dataset Types

1. Define the data model in `dataset_prompts.py`:
```python
class YourDatasetModel(BaseModel):
    your_parameter: float
    # ... other fields
```

2. Add extraction prompt:
```python
def get_dataset_prompt(dataset_type):
    if dataset_type == DatasetType.YOUR_TYPE:
        return "Your specialized prompt..."
```

3. Map folder names to dataset type:
```python
mapping = {
    'your_dataset': DatasetType.YOUR_TYPE
}
```

### Adding New VLM Providers

1. Add configuration in `vlm_config.py`:
```python
self.configs[VLMProvider.YOUR_PROVIDER] = VLMConfig(
    provider=VLMProvider.YOUR_PROVIDER,
    api_url="your-api-url",
    api_key="your-key",
    model_name="model-name"
)
```

## Troubleshooting

### Common Issues

1. **YOLO not detecting objects**
   - Check model weights path
   - Ensure PDF has sufficient resolution
   - Verify CUDA installation for GPU support

2. **VLM connection errors**
   - Verify API credentials in .env
   - Check server availability
   - Test connection with provided scripts

3. **Memory issues with large PDFs**
   - Reduce DPI in image conversion
   - Process PDFs in smaller batches
   - Increase system swap space

## Performance Tips

- Use GPU for YOLO inference (automatic if CUDA available)
- Adjust worker count based on CPU cores
- Process similar datasets together for better caching
- Use SSD storage for temporary files

## License

[Specify your license here]

## Citation

If you use this pipeline in your research, please cite:
```bibtex
[Add citation information]
```

## Contact

For questions and support, please contact [your contact information].