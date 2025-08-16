#!/usr/bin/env python3
"""
Example usage of the Vision Pipeline
"""

from pathlib import Path
from core.image_extracting_vlm import pdf_analysis
from core.vlm_config import VLMProvider
import json


def process_single_pdf():
    """Example: Process a single PDF file"""
    
    # Configuration
    pdf_path = "path/to/your/paper.pdf"
    yolo_model = "best.pt"  # or path to your YOLO model
    dataset_type = "nanozymes"  # Choose: nanozymes, cytotoxicity, magnetic, etc.
    
    print(f"Processing {pdf_path}...")
    
    # Run analysis
    result = pdf_analysis(
        pdf_path=pdf_path,
        yolo_model_path=yolo_model,
        vlm_provider=VLMProvider.GEMMA3_27B,
        dataset_name=dataset_type
    )
    
    # Display results
    print(f"\nDataset type: {result['dataset_type']}")
    print(f"Graphs found: {len(result['analyses'])}")
    print(f"Tables found: {len(result['tables'])}")
    
    # Show extraction statistics
    if 'extraction_stats' in result:
        stats = result['extraction_stats']
        print("\nExtraction Statistics:")
        print(f"  Pages with images: {stats['pages_with_images']}")
        print(f"  Graphs detected: {stats['graphs_detected']}")
        print(f"  Graphs analyzed: {stats['graphs_analyzed']}")
    
    # Save results to JSON
    output_file = f"results_{Path(pdf_path).stem}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return result


def process_batch():
    """Example: Process multiple PDFs in a directory"""
    
    from core.vlm_config import vlm_manager
    import os
    
    # Configuration
    pdf_dir = Path("path/to/pdf/directory")
    dataset_type = "cytotoxicity"
    results = []
    
    # Process all PDFs in directory
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"\nProcessing {pdf_file.name}...")
        
        try:
            result = pdf_analysis(
                pdf_path=str(pdf_file),
                dataset_name=dataset_type,
                vlm_provider=VLMProvider.GEMMA3_27B
            )
            
            results.append({
                'filename': pdf_file.name,
                'n_graphs': len(result['analyses']),
                'n_tables': len(result['tables']),
                'data': result
            })
            
            print(f"  ✓ Processed: {len(result['analyses'])} graphs, {len(result['tables'])} tables")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'filename': pdf_file.name,
                'error': str(e)
            })
    
    # Save batch results
    with open('batch_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nBatch processing complete!")
    print(f"Processed {len(results)} files")
    print(f"Results saved to batch_results.json")
    
    return results


def extract_specific_parameters():
    """Example: Extract specific parameters from results"""
    
    # Load previous results
    with open('results_example.json', 'r') as f:
        result = json.load(f)
    
    # Extract nanozymes kinetic parameters
    if result['dataset_type'] == 'nanozymes':
        for analysis in result['analyses']:
            if 'kinetic_parameters' in analysis:
                params = analysis['kinetic_parameters']
                if 'km_value' in params:
                    print(f"Km = {params['km_value']} {params.get('km_unit', '')}")
                if 'vmax_value' in params:
                    print(f"Vmax = {params['vmax_value']} {params.get('vmax_unit', '')}")
    
    # Extract cytotoxicity IC50 values
    elif result['dataset_type'] == 'cytotoxicity':
        for analysis in result['analyses']:
            if 'ic50_value' in analysis:
                print(f"IC50 = {analysis['ic50_value']} {analysis.get('ic50_unit', '')}")
    
    # Extract magnetic properties
    elif result['dataset_type'] == 'magnetic':
        for analysis in result['analyses']:
            if 'ms_emu_g' in analysis:
                print(f"Ms = {analysis['ms_emu_g']} emu/g")
            if 'hc_kOe' in analysis:
                print(f"Hc = {analysis['hc_kOe']} kOe")


def test_vlm_connection():
    """Example: Test VLM provider connection"""
    
    from core.vlm_config import test_vlm_connection, VLMProvider
    
    print("Testing VLM connections...")
    
    # Test Gemma3-27b
    success, message = test_vlm_connection(VLMProvider.GEMMA3_27B)
    print(f"Gemma3-27b: {'✓' if success else '✗'} - {message}")
    
    # Test other providers if configured
    # success, message = test_vlm_connection(VLMProvider.GLM_4_1)
    # print(f"GLM 4.1: {'✓' if success else '✗'} - {message}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "single":
            process_single_pdf()
        elif command == "batch":
            process_batch()
        elif command == "extract":
            extract_specific_parameters()
        elif command == "test":
            test_vlm_connection()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: single, batch, extract, test")
    else:
        print("Vision Pipeline Examples")
        print("========================")
        print("\nUsage: python example_usage.py [command]")
        print("\nCommands:")
        print("  single  - Process a single PDF")
        print("  batch   - Process multiple PDFs")
        print("  extract - Extract specific parameters from results")
        print("  test    - Test VLM connections")
        print("\nEdit this file to configure paths and parameters.")