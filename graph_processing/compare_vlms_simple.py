#!/usr/bin/env python3
"""
Simple VLM comparison script
"""

import json
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import needed modules
from vlm_config import VLMProvider, vlm_manager
from image_extracting_vlm import pdf_analysis


def test_single_file_with_both_vlms(pdf_path: str, yolo_path: str = "best.pt"):
    """Test a single PDF with both VLMs"""
    
    results = {}
    
    # Test with GLM-4.1V
    logger.info(f"Testing with GLM-4.1V: {pdf_path}")
    try:
        glm_result = pdf_analysis(
            pdf_path=pdf_path,
            yolo_model_path=yolo_path,
            vlm_provider=VLMProvider.GLM_4_1V,
            dataset_name="synergy"
        )
        results['glm_4_1v'] = {
            'success': True,
            'graphs': len(glm_result.get('graphs', [])),
            'tables': len(glm_result.get('tables', [])),
            'sample': str(glm_result.get('graphs', ['No graphs'])[0])[:200] if glm_result.get('graphs') else 'No graphs'
        }
        logger.info(f"  GLM-4.1V: {results['glm_4_1v']['graphs']} graphs, {results['glm_4_1v']['tables']} tables")
    except Exception as e:
        logger.error(f"  GLM-4.1V failed: {e}")
        results['glm_4_1v'] = {'success': False, 'error': str(e)}
    
    # Test with Gemma-3-1B  
    logger.info(f"Testing with Gemma-3-1B: {pdf_path}")
    try:
        gemma_result = pdf_analysis(
            pdf_path=pdf_path,
            yolo_model_path=yolo_path,
            vlm_provider=VLMProvider.GEMMA_3_1B,
            dataset_name="synergy"
        )
        results['gemma_3_1b'] = {
            'success': True,
            'graphs': len(gemma_result.get('graphs', [])),
            'tables': len(gemma_result.get('tables', [])),
            'sample': str(gemma_result.get('graphs', ['No graphs'])[0])[:200] if gemma_result.get('graphs') else 'No graphs'
        }
        logger.info(f"  Gemma-3-1B: {results['gemma_3_1b']['graphs']} graphs, {results['gemma_3_1b']['tables']} tables")
    except Exception as e:
        logger.error(f"  Gemma-3-1B failed: {e}")
        results['gemma_3_1b'] = {'success': False, 'error': str(e)}
    
    return results


def main():
    """Main entry point"""
    
    # Find a test PDF
    test_dir = Path("extracted_pdfs/synergy/test/synergy_test_pdf")
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found")
        return
    
    # Test first 3 PDFs
    all_results = {}
    
    for pdf_path in pdf_files[:3]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        results = test_single_file_with_both_vlms(str(pdf_path))
        all_results[pdf_path.name] = results
    
    # Print comparison
    print("\n" + "="*70)
    print("VLM COMPARISON RESULTS")
    print("="*70)
    
    glm_total_graphs = 0
    glm_total_tables = 0
    gemma_total_graphs = 0
    gemma_total_tables = 0
    
    for pdf_name, results in all_results.items():
        print(f"\n{pdf_name}:")
        
        if results['glm_4_1v']['success']:
            glm_total_graphs += results['glm_4_1v']['graphs']
            glm_total_tables += results['glm_4_1v']['tables']
            print(f"  GLM-4.1V: {results['glm_4_1v']['graphs']} graphs, {results['glm_4_1v']['tables']} tables")
            print(f"    Sample: {results['glm_4_1v']['sample'][:100]}...")
        else:
            print(f"  GLM-4.1V: Failed - {results['glm_4_1v']['error']}")
        
        if results['gemma_3_1b']['success']:
            gemma_total_graphs += results['gemma_3_1b']['graphs']
            gemma_total_tables += results['gemma_3_1b']['tables']
            print(f"  Gemma-3-1B: {results['gemma_3_1b']['graphs']} graphs, {results['gemma_3_1b']['tables']} tables")
            print(f"    Sample: {results['gemma_3_1b']['sample'][:100]}...")
        else:
            print(f"  Gemma-3-1B: Failed - {results['gemma_3_1b']['error']}")
    
    print("\n" + "-"*70)
    print("TOTALS:")
    print(f"  GLM-4.1V:   {glm_total_graphs} graphs, {glm_total_tables} tables")
    print(f"  Gemma-3-1B: {gemma_total_graphs} graphs, {gemma_total_tables} tables")
    
    # Save results
    output_file = f"vlm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()