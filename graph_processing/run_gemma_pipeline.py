#!/usr/bin/env python3
"""
Batch processing script for vision pipeline with Gemma-3-27b VLM
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add graph_processing to path
sys.path.insert(0, str(Path(__file__).parent))

from vlm_config import VLMProvider, VLMManager
from image_extracting_vlm import pdf_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VisionPipelineProcessor:
    """Process documents through YOLO + Gemma VLM pipeline"""
    
    def __init__(self, vlm_provider: VLMProvider = VLMProvider.GEMMA_3_27B):
        """Initialize with Gemma-3-27b provider"""
        self.base_dir = Path(".")
        self.datasets_dir = self.base_dir / "extracted_pdfs"  # Use extracted_pdfs like GLM
        self.results_dir = self.base_dir / "vision_results_gemma"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize VLM config
        self.vlm_config = VLMManager()
        self.vlm_provider = vlm_provider
        
        # Track processing stats
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'errors': []
        }
    
    def get_dataset_files(self, dataset: str, split: str) -> List[Path]:
        """Get PDF files for a dataset/split"""
        # Build path to PDF directory - extracted_pdfs uses nested structure
        pdf_dir = self.datasets_dir / dataset / split
        
        if not pdf_dir.exists():
            logger.warning(f"Directory not found: {pdf_dir}")
            return []
        
        # Get all PDF files - they're in subdirectories
        pdf_files = list(pdf_dir.glob("*/*.pdf"))  # PDFs are in subdirectories
        if not pdf_files:
            pdf_files = list(pdf_dir.glob("*.pdf"))  # Also check direct PDFs
        
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        return pdf_files
    
    def process_single_file(self, pdf_path: Path, dataset: str) -> Dict:
        """Process a single PDF file"""
        try:
            logger.info(f"Processing: {pdf_path.name}")
            
            # Analyze document using pdf_analysis
            result = pdf_analysis(
                str(pdf_path),
                vlm_provider=self.vlm_provider,
                dataset_name=dataset
            )
            
            # Add metadata
            result['pdf_name'] = pdf_path.name
            result['dataset'] = dataset
            result['processed_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
            return {
                'pdf_name': pdf_path.name,
                'dataset': dataset,
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def process_dataset(self, dataset: str, split: str, max_workers: int = 4):
        """Process all files in a dataset/split"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset} - {split}")
        logger.info(f"{'='*60}")
        
        # Get files
        pdf_files = self.get_dataset_files(dataset, split)
        if not pdf_files:
            logger.warning(f"No files found for {dataset}/{split}")
            return
        
        # Create output directory
        output_dir = self.results_dir / dataset / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        results = []
        self.stats['total_files'] += len(pdf_files)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, pdf_file, dataset): pdf_file
                for pdf_file in pdf_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if 'error' in result:
                        self.stats['failed'] += 1
                        self.stats['errors'].append({
                            'file': pdf_file.name,
                            'error': result['error']
                        })
                    else:
                        self.stats['processed'] += 1
                        
                    # Log progress
                    total = len(pdf_files)
                    done = len(results)
                    logger.info(f"Progress: {done}/{total} ({done*100//total}%)")
                    
                except Exception as e:
                    logger.error(f"Task failed for {pdf_file.name}: {str(e)}")
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'file': pdf_file.name,
                        'error': str(e)
                    })
        
        # Save results
        output_file = output_dir / f"{dataset}_{split}_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'dataset': dataset,
                'split': split,
                'vlm_provider': self.vlm_provider.value,
                'total_files': len(pdf_files),
                'processed': len(results),
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Completed {dataset}/{split}: {len(results)}/{len(pdf_files)} files")
    
    def run_full_pipeline(self):
        """Run pipeline on all datasets"""
        # Define datasets to process
        datasets = [
            ('synergy', 'train'),
            ('synergy', 'test'),
            ('cytox', 'train'),
            ('cytox', 'test'),
            ('magnetic', 'train'),
            ('magnetic', 'test'),
            ('nanozymes', 'train'),
            ('nanozymes', 'test'),
            ('seltox', 'train'),
            ('seltox', 'test'),
        ]
        
        start_time = datetime.now()
        
        for dataset, split in datasets:
            try:
                self.process_dataset(dataset, split)
            except Exception as e:
                logger.error(f"Failed to process {dataset}/{split}: {str(e)}")
                self.stats['errors'].append({
                    'dataset': f"{dataset}/{split}",
                    'error': str(e)
                })
        
        # Save overall summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'vlm_provider': self.vlm_provider.value,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'stats': self.stats,
            'datasets_processed': [f"{d[0]}/{d[1]}" for d in datasets]
        }
        
        summary_file = self.results_dir / f"overall_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED - GEMMA-3-27B")
        print("="*60)
        print(f"Total files: {self.stats['total_files']}")
        print(f"Processed: {self.stats['processed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Results saved to: {self.results_dir}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for err in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {err}")

def main():
    """Main entry point"""
    processor = VisionPipelineProcessor(vlm_provider=VLMProvider.GEMMA_3_27B)
    processor.run_full_pipeline()

if __name__ == "__main__":
    main()