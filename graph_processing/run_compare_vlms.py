#!/usr/bin/env python3
"""
Run vision pipeline with different VLMs for comparison
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_vision_pipeline import VisionPipelineProcessor
from vlm_config import VLMProvider
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_with_vlm(vlm_provider: VLMProvider, dataset: str, split: str, max_files: int = 10):
    """Run processing with specific VLM on subset of files"""
    
    processor = VisionPipelineProcessor()
    
    # Create output directory for this VLM
    vlm_name = vlm_provider.value.replace("-", "_").replace(".", "_")
    output_dir = Path(f"vision_results_{vlm_name}")
    output_dir.mkdir(exist_ok=True)
    
    # Override results directory
    processor.results_dir = output_dir
    
    logger.info(f"Running {dataset}/{split} with {vlm_provider.value}")
    logger.info(f"Processing first {max_files} files")
    
    # Load annotations
    annotations = processor.load_csv_annotations(dataset, split)
    
    # Get PDF files (limit to max_files)
    pdf_files = processor.get_pdf_files(dataset, split)[:max_files]
    
    if not pdf_files:
        logger.warning(f"No PDF files found for {dataset}/{split}")
        return None
    
    logger.info(f"Processing {len(pdf_files)} files with {vlm_provider.value}")
    
    # Process files sequentially with specified VLM
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name}")
        
        try:
            # Import pdf_analysis function directly
            from image_extracting_vlm import pdf_analysis
            
            # Process with specific VLM
            analysis_result = pdf_analysis(
                pdf_path=str(pdf_path),
                yolo_model_path=str(processor.yolo_path),
                vlm_provider=vlm_provider,
                dataset_name=dataset
            )
            
            # Format result
            result = {
                'pdf_path': str(pdf_path),
                'pdf_name': pdf_path.name,
                'status': 'success',
                'analyses': analysis_result.get('graphs', []),
                'tables': analysis_result.get('tables', []),
                'annotations': annotations[annotations['статья'] == pdf_path.stem].to_dict('records') if annotations is not None else []
            }
            results.append(result)
            logger.info(f"  ✓ Completed: {len(result['analyses'])} graphs, {len(result['tables'])} tables")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            results.append({
                'pdf_path': str(pdf_path),
                'pdf_name': pdf_path.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results
    output_file = output_dir / f"{dataset}_{split}_{vlm_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'vlm_provider': vlm_provider.value,
            'dataset': dataset,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            'n_files': len(pdf_files),
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return output_file


def compare_vlm_results(glm_file: Path, gemma_file: Path):
    """Compare results from two VLMs"""
    
    # Load results
    with open(glm_file) as f:
        glm_data = json.load(f)
    
    with open(gemma_file) as f:
        gemma_data = json.load(f)
    
    print("\n" + "="*70)
    print("VLM COMPARISON RESULTS")
    print("="*70)
    
    print(f"\nGLM-4.1V Results ({glm_file.name}):")
    print(f"  Files processed: {glm_data['n_files']}")
    
    glm_graphs = sum(len(r.get('analyses', [])) for r in glm_data['results'])
    glm_tables = sum(len(r.get('tables', [])) for r in glm_data['results'])
    print(f"  Total graphs extracted: {glm_graphs}")
    print(f"  Total tables extracted: {glm_tables}")
    
    print(f"\nGemma-3-27B Results ({gemma_file.name}):")
    print(f"  Files processed: {gemma_data['n_files']}")
    
    gemma_graphs = sum(len(r.get('analyses', [])) for r in gemma_data['results'])
    gemma_tables = sum(len(r.get('tables', [])) for r in gemma_data['results'])
    print(f"  Total graphs extracted: {gemma_graphs}")
    print(f"  Total tables extracted: {gemma_tables}")
    
    # Compare extraction quality
    print("\n" + "-"*70)
    print("EXTRACTION COMPARISON")
    print("-"*70)
    
    # Check if same files were processed
    glm_files = {r['pdf_name'] for r in glm_data['results']}
    gemma_files = {r['pdf_name'] for r in gemma_data['results']}
    
    common_files = glm_files & gemma_files
    print(f"\nCommon files processed: {len(common_files)}")
    
    if common_files:
        print("\nPer-file comparison (first 5):")
        for pdf_name in list(common_files)[:5]:
            # Find results for this PDF
            glm_result = next(r for r in glm_data['results'] if r['pdf_name'] == pdf_name)
            gemma_result = next(r for r in gemma_data['results'] if r['pdf_name'] == pdf_name)
            
            print(f"\n  {pdf_name}:")
            print(f"    GLM-4.1V: {len(glm_result.get('analyses', []))} graphs, {len(glm_result.get('tables', []))} tables")
            print(f"    Gemma-3:  {len(gemma_result.get('analyses', []))} graphs, {len(gemma_result.get('tables', []))} tables")
            
            # Show sample extraction
            if glm_result.get('analyses'):
                sample = str(glm_result['analyses'][0])[:100]
                print(f"    GLM sample: {sample}...")
            if gemma_result.get('analyses'):
                sample = str(gemma_result['analyses'][0])[:100]
                print(f"    Gemma sample: {sample}...")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare VLM performance")
    parser.add_argument('--dataset', default='synergy', help='Dataset to test')
    parser.add_argument('--split', default='test', help='Split to test (train/test)')
    parser.add_argument('--max-files', type=int, default=5, help='Number of files to process per VLM')
    parser.add_argument('--compare-only', action='store_true', help='Only compare existing results')
    
    args = parser.parse_args()
    
    if not args.compare_only:
        # Run with GLM-4.1V
        print("\n" + "="*70)
        print("RUNNING WITH GLM-4.1V")
        print("="*70)
        glm_file = run_with_vlm(
            VLMProvider.GLM_4_1V, 
            args.dataset, 
            args.split, 
            args.max_files
        )
        
        # Run with Gemma-3-27B
        print("\n" + "="*70)
        print("RUNNING WITH GEMMA-3-27B")
        print("="*70)
        gemma_file = run_with_vlm(
            VLMProvider.GEMMA_3_27B, 
            args.dataset, 
            args.split, 
            args.max_files
        )
        
        if glm_file and gemma_file:
            # Compare results
            compare_vlm_results(glm_file, gemma_file)
    else:
        # Just compare existing results
        glm_file = Path(f"vision_results_glm_4_1v/{args.dataset}_{args.split}_glm_4_1v_results.json")
        gemma_file = Path(f"vision_results_gemma_3_1b/{args.dataset}_{args.split}_gemma_3_1b_results.json")
        
        if glm_file.exists() and gemma_file.exists():
            compare_vlm_results(glm_file, gemma_file)
        else:
            print("Result files not found. Run without --compare-only first.")


if __name__ == "__main__":
    main()