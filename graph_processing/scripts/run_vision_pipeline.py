#!/usr/bin/env python3
"""
Main script for batch processing of vision pipeline on CV sets
Processes all train and test datasets with YOLO detection and VLM analysis
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import vision pipeline modules
try:
    # Try to import the VLM-enhanced version first
    from graph_processing.image_extracting_vlm import pdf_analysis
    logger.info("Using VLM-enhanced image extraction")
except ImportError:
    # Fall back to original version
    from graph_processing.image_extracting import pdf_analysis
    logger.warning("Using original image extraction (no VLM support)")

from graph_processing.vlm_config import VLMProvider, vlm_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'vision_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VisionPipelineProcessor:
    """Main processor for vision pipeline on CV marked datasets"""
    
    DATASET_MAPPING = {
        'cytox': 'cytotoxicity',
        'magnetic': 'magnetism', 
        'nanozymes': 'nanozymes',
        'seltox': 'selective_toxicity',
        'synergy': 'synergy'
    }
    
    def __init__(self, base_dir: str = None):
        """Initialize processor with base directory"""
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = Path(base_dir)
        self.extracted_pdfs_dir = self.base_dir / "extracted_pdfs"
        self.csv_dir = self.base_dir / "datasets" / "plots" / "cv_sets_marked" / "сsvs"
        self.results_dir = self.base_dir / "vision_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Set YOLO model path
        self.yolo_path = self.base_dir / "best.pt"
        os.environ['YOLO_PATH'] = str(self.yolo_path)
        
        logger.info(f"Initialized processor with base dir: {self.base_dir}")
        logger.info(f"YOLO model path: {self.yolo_path}")
    
    def load_csv_annotations(self, dataset: str, split: str) -> pd.DataFrame:
        """Load CSV annotations for a specific dataset and split"""
        # Map dataset names to CSV file patterns
        csv_patterns = {
            'cytox': 'cyto',
            'magnetic': 'mag',
            'nanozymes': 'nzyme',
            'seltox': 'seltox',
            'synergy': 'syn'
        }
        
        pattern = csv_patterns.get(dataset, dataset)
        csv_filename = f"cv sets marked - {pattern}_{split}.csv"
        csv_path = self.csv_dir / csv_filename
        
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"Loaded {len(df)} annotations from {csv_filename}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV {csv_path}: {e}")
            return pd.DataFrame()
    
    def get_pdf_files(self, dataset: str, split: str) -> List[Path]:
        """Get list of PDF files for a dataset and split"""
        pdf_dir = self.extracted_pdfs_dir / dataset / split
        
        if not pdf_dir.exists():
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return []
        
        # Get all PDF files, handling potential subdirectories
        pdf_files = []
        for pattern in ['*.pdf', '**/*.pdf']:
            pdf_files.extend(pdf_dir.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        return pdf_files
    
    def process_single_pdf(self, pdf_path: Path, annotations: pd.DataFrame = None) -> Dict:
        """Process a single PDF file through the vision pipeline"""
        result = {
            'pdf_path': str(pdf_path),
            'pdf_name': pdf_path.name,
            'status': 'pending',
            'analyses': [],
            'tables': [],
            'error': None,
            'annotations': []
        }
        
        try:
            # Get relevant annotations if available
            if annotations is not None and not annotations.empty:
                pdf_name_clean = pdf_path.stem.lower()
                relevant_annotations = annotations[
                    annotations['статья'].str.lower().str.contains(pdf_name_clean, na=False)
                ] if 'статья' in annotations.columns else pd.DataFrame()
                
                if not relevant_annotations.empty:
                    result['annotations'] = relevant_annotations.to_dict('records')
            
            # Run vision pipeline with VLM and dataset-specific prompts
            logger.info(f"Processing: {pdf_path.name}")
            
            # Get dataset name from the processing context
            dataset_name = pdf_path.parts[-3] if len(pdf_path.parts) > 3 else 'nanozymes'
            
            # Use Gemma3-27b as primary VLM provider with dataset-specific analysis
            analysis_result = pdf_analysis(
                str(pdf_path), 
                str(self.yolo_path), 
                vlm_provider=VLMProvider.GEMMA3_27B,
                dataset_name=dataset_name
            )
            
            if analysis_result:
                result['analyses'] = analysis_result.get('analyses', [])
                result['tables'] = analysis_result.get('tables', [])
                result['status'] = 'success'
                
                # Log summary
                n_graphs = len(result['analyses'])
                n_tables = len(result['tables'])
                logger.info(f"✓ {pdf_path.name}: {n_graphs} graphs, {n_tables} tables")
            else:
                result['status'] = 'no_results'
                logger.warning(f"No results for {pdf_path.name}")
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Error processing {pdf_path.name}: {e}")
            logger.debug(traceback.format_exc())
        
        return result
    
    def process_dataset(self, dataset: str, split: str, max_workers: int = 2) -> Dict:
        """Process all PDFs for a dataset and split"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset} - {split}")
        logger.info(f"{'='*60}")
        
        # Load annotations
        annotations = self.load_csv_annotations(dataset, split)
        
        # Get PDF files
        pdf_files = self.get_pdf_files(dataset, split)
        
        if not pdf_files:
            logger.warning(f"No PDF files found for {dataset}/{split}")
            return {'dataset': dataset, 'split': split, 'results': [], 'status': 'no_files'}
        
        # Process PDFs
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self.process_single_pdf, pdf_path, annotations): pdf_path
                for pdf_path in pdf_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(pdf_files), desc=f"{dataset}-{split}") as pbar:
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per PDF
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process {pdf_path}: {e}")
                        results.append({
                            'pdf_path': str(pdf_path),
                            'pdf_name': pdf_path.name,
                            'status': 'error',
                            'error': str(e)
                        })
                    pbar.update(1)
        
        # Save results
        output_dir = self.results_dir / dataset / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{dataset}_{split}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset,
                'split': split,
                'timestamp': datetime.now().isoformat(),
                'n_files': len(pdf_files),
                'n_processed': len(results),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Generate summary
        summary = self.generate_summary(results)
        summary_file = output_dir / f"{dataset}_{split}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return {
            'dataset': dataset,
            'split': split,
            'results': results,
            'status': 'completed',
            'summary': summary
        }
    
    def generate_summary(self, results: List[Dict]) -> str:
        """Generate summary statistics for processing results"""
        total = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        no_results = sum(1 for r in results if r['status'] == 'no_results')
        errors = sum(1 for r in results if r['status'] == 'error')
        
        total_graphs = sum(len(r.get('analyses', [])) for r in results)
        total_tables = sum(len(r.get('tables', [])) for r in results)
        
        # Extract concentration data statistics
        concentration_ranges = []
        for r in results:
            for analysis in r.get('analyses', []):
                if 'concentration_data' in analysis and analysis['concentration_data']:
                    for conc in analysis['concentration_data']:
                        concentration_ranges.append({
                            'pdf': r['pdf_name'],
                            'reaction': conc.get('reaction_type', 'unknown'),
                            'c_min': conc.get('c_min', 0),
                            'c_max': conc.get('c_max', 0)
                        })
        
        summary = f"""
Processing Summary
==================
Total PDFs: {total}
Successful: {successful}
No Results: {no_results}
Errors: {errors}

Extracted Data
==============
Total Graphs: {total_graphs}
Total Tables: {total_tables}
Concentration Ranges: {len(concentration_ranges)}

Concentration Data Examples:
"""
        
        for i, conc in enumerate(concentration_ranges[:5], 1):
            summary += f"\n{i}. {conc['pdf']}: {conc['reaction']} ({conc['c_min']:.2f} - {conc['c_max']:.2f} mM)"
        
        if len(concentration_ranges) > 5:
            summary += f"\n... and {len(concentration_ranges) - 5} more"
        
        return summary
    
    def run_all(self, datasets: List[str] = None, splits: List[str] = None, max_workers: int = 2):
        """Run pipeline for all or specified datasets and splits"""
        if datasets is None:
            datasets = list(self.DATASET_MAPPING.keys())
        
        if splits is None:
            splits = ['train', 'test']
        
        logger.info(f"Starting batch processing for {len(datasets)} datasets, {len(splits)} splits")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Splits: {splits}")
        
        all_results = {}
        
        for dataset in datasets:
            for split in splits:
                key = f"{dataset}_{split}"
                try:
                    result = self.process_dataset(dataset, split, max_workers)
                    all_results[key] = result
                except Exception as e:
                    logger.error(f"Failed to process {dataset}/{split}: {e}")
                    all_results[key] = {
                        'dataset': dataset,
                        'split': split,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        # Save overall summary
        self.save_overall_summary(all_results)
        
        return all_results
    
    def save_overall_summary(self, all_results: Dict):
        """Save overall processing summary"""
        summary_path = self.results_dir / f"overall_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'datasets_processed': len(all_results),
            'results': {}
        }
        
        for key, result in all_results.items():
            if 'results' in result:
                n_success = sum(1 for r in result['results'] if r['status'] == 'success')
                n_total = len(result['results'])
                summary['results'][key] = {
                    'status': result['status'],
                    'files_processed': n_total,
                    'successful': n_success,
                    'success_rate': f"{(n_success/n_total*100):.1f}%" if n_total > 0 else "0%"
                }
            else:
                summary['results'][key] = {
                    'status': result.get('status', 'unknown'),
                    'error': result.get('error', None)
                }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Overall summary saved to: {summary_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run vision pipeline on CV marked datasets")
    parser.add_argument('--datasets', nargs='+', 
                       choices=['cytox', 'magnetic', 'nanozymes', 'seltox', 'synergy'],
                       help='Specific datasets to process (default: all)')
    parser.add_argument('--splits', nargs='+', 
                       choices=['train', 'test'],
                       help='Splits to process (default: both)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel workers (default: 2)')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test VLM connections before processing')
    
    args = parser.parse_args()
    
    # Test VLM connections if requested
    if args.test_connection:
        logger.info("Testing VLM connections...")
        from graph_processing.vlm_config import test_vlm_connection, VLMProvider
        
        for provider in [VLMProvider.GEMMA3_27B, VLMProvider.GLM_4_1]:
            try:
                success, message = test_vlm_connection(provider)
                logger.info(f"{provider.value}: {'✓' if success else '✗'} - {message}")
            except Exception as e:
                logger.error(f"{provider.value}: ✗ - {e}")
        
        return
    
    # Run pipeline
    processor = VisionPipelineProcessor()
    
    try:
        results = processor.run_all(
            datasets=args.datasets,
            splits=args.splits,
            max_workers=args.workers
        )
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*60)
        
        for key, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'completed' and 'results' in result:
                n_success = sum(1 for r in result['results'] if r['status'] == 'success')
                n_total = len(result['results'])
                logger.info(f"{key}: {n_success}/{n_total} successful")
            else:
                logger.info(f"{key}: {status}")
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()