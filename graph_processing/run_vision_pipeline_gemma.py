#!/usr/bin/env python3
"""
Run vision pipeline with Gemma-3-27B for all datasets
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_vision_pipeline import VisionPipelineProcessor
from vlm_config import VLMProvider
from image_extracting_vlm import pdf_analysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_pipeline_gemma.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run vision pipeline with Gemma-3-27B')
    parser.add_argument('--datasets', nargs='+', default=['magnetic', 'nanozymes', 'seltox'],
                       help='Datasets to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                       help='Splits to process (train/test)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VisionPipelineProcessor()
    
    # Override results directory for Gemma
    processor.results_dir = Path("vision_results_gemma")
    processor.results_dir.mkdir(exist_ok=True)
    
    # Force Gemma-3-27B provider
    vlm_provider = VLMProvider.GEMMA_3_27B
    
    logger.info(f"Starting vision pipeline with Gemma-3-27B")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Results directory: {processor.results_dir}")
    
    # Process each dataset and split
    for dataset in args.datasets:
        for split in args.splits:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {dataset} - {split}")
            logger.info(f"{'='*60}")
            
            # Check if already processed
            result_file = processor.results_dir / dataset / split / f"{dataset}_{split}_results.json"
            if result_file.exists() and not args.resume:
                logger.info(f"✓ Already processed: {result_file}")
                continue
            
            try:
                # Process dataset with Gemma
                processor.process_dataset(
                    dataset=dataset,
                    split=split,
                    max_workers=args.workers
                )
                logger.info(f"✓ Completed {dataset}/{split}")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {dataset}/{split}: {e}")
                continue
    
    logger.info("\n" + "="*60)
    logger.info("GEMMA PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved in: {processor.results_dir}")


if __name__ == "__main__":
    main()