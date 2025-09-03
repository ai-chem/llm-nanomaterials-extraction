#!/usr/bin/env python3
"""
Continue Gemma pipeline from magnetic dataset with optimizations
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.insert(0, str(Path(__file__).parent))

from vlm_config import VLMProvider, VLMManager
from image_extracting_vlm import pdf_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VisionPipelineProcessor:
    """Optimized processor continuing from magnetic dataset"""
    
    def __init__(self, vlm_provider: VLMProvider = VLMProvider.GEMMA_3_27B):
        self.base_dir = Path(".")
        self.datasets_dir = self.base_dir / "extracted_pdfs"
        self.results_dir = self.base_dir / "vision_results_gemma"
        self.results_dir.mkdir(exist_ok=True)
        
        self.vlm_config = VLMManager()
        self.vlm_provider = vlm_provider
        
        # Load previous stats
        self.stats = self.load_previous_stats()
    
    def load_previous_stats(self):
        """Load stats from previous runs"""
        stats = {
            'total_files': 933,
            'processed': 261,  # Already processed files
            'failed': 0,
            'errors': [],
            'start_time': datetime.now().isoformat()
        }
        return stats
    
    def get_dataset_files(self, dataset: str, split: str) -> List[Path]:
        """Get PDF files for a dataset/split"""
        pdf_dir = self.datasets_dir / dataset / split
        
        if not pdf_dir.exists():
            logger.warning(f"Directory not found: {pdf_dir}")
            return []
        
        pdf_files = list(pdf_dir.glob("*/*.pdf"))
        if not pdf_files:
            pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        return pdf_files
    
    def process_single_file(self, pdf_path: Path, dataset: str) -> Dict:
        """Process a single PDF file with timeout"""
        try:
            logger.info(f"Processing: {pdf_path.name}")
            start_time = time.time()
            
            # Analyze document with timeout consideration
            result = pdf_analysis(
                str(pdf_path),
                vlm_provider=self.vlm_provider,
                dataset_name=dataset
            )
            
            # Add metadata
            result['pdf_name'] = pdf_path.name
            result['dataset'] = dataset
            result['processed_at'] = datetime.now().isoformat()
            result['processing_time'] = time.time() - start_time
            
            logger.info(f"✓ Completed {pdf_path.name} in {result['processing_time']:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
            return {
                'pdf_name': pdf_path.name,
                'dataset': dataset,
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def process_dataset(self, dataset: str, split: str, max_workers: int = 6):
        """Process with increased workers for better parallelization"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset} - {split} with {max_workers} workers")
        logger.info(f"{'='*60}")
        
        pdf_files = self.get_dataset_files(dataset, split)
        if not pdf_files:
            logger.warning(f"No files found for {dataset}/{split}")
            return
        
        output_dir = self.results_dir / dataset / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        batch_size = 10  # Process in smaller batches
        
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(pdf_files)-1)//batch_size + 1}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_file, pdf_file, dataset): pdf_file
                    for pdf_file in batch
                }
                
                for future in as_completed(future_to_file):
                    pdf_file = future_to_file[future]
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per file
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
                        logger.info(f"Dataset progress: {len(results)}/{len(pdf_files)} files")
                        
                    except Exception as e:
                        logger.error(f"Task failed for {pdf_file.name}: {str(e)}")
                        self.stats['failed'] += 1
                        results.append({
                            'pdf_name': pdf_file.name,
                            'dataset': dataset,
                            'error': f"Timeout or error: {str(e)}",
                            'processed_at': datetime.now().isoformat()
                        })
            
            # Save intermediate results
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
            logger.info(f"Intermediate save: {len(results)} results")
        
        logger.info(f"✓ Completed {dataset}/{split}: {len(results)}/{len(pdf_files)} files")
    
    def run_continuation(self):
        """Continue from magnetic dataset"""
        # Remaining datasets to process
        datasets = [
            ('magnetic', 'train'),
            ('magnetic', 'test'),
            ('nanozymes', 'train'),
            ('nanozymes', 'test'),
            ('seltox', 'train'),
            ('seltox', 'test'),
        ]
        
        start_time = datetime.now()
        logger.info(f"Continuing Gemma pipeline from magnetic dataset...")
        logger.info(f"Already processed: {self.stats['processed']} files")
        
        for dataset, split in datasets:
            try:
                self.process_dataset(dataset, split, max_workers=6)
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
            'continuation_start': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'stats': self.stats,
            'datasets_processed': [f"{d[0]}/{d[1]}" for d in datasets]
        }
        
        summary_file = self.results_dir / f"continuation_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("GEMMA PIPELINE CONTINUATION COMPLETED")
        print("="*60)
        print(f"Total files: {self.stats['total_files']}")
        print(f"Processed: {self.stats['processed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Results saved to: {self.results_dir}")

def main():
    processor = VisionPipelineProcessor(vlm_provider=VLMProvider.GEMMA_3_27B)
    processor.run_continuation()

if __name__ == "__main__":
    main()