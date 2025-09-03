#!/usr/bin/env python3
"""
Evaluation metrics for vision pipeline extraction quality
Compares extracted parameters with ground truth from CSV annotations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
# We calculate metrics manually, no need for sklearn
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """Metrics for parameter extraction evaluation"""
    dataset: str
    split: str
    parameter: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1, and accuracy"""
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        
        if total == 0:
            return
            
        # Precision: TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        
        # Recall: TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        
        # F1: 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        # Accuracy: (TP + TN) / total
        self.accuracy = (self.true_positives + self.true_negatives) / total


class ExtractionEvaluator:
    """Evaluates extraction quality against ground truth"""
    
    DATASET_PARAMS = {
        'cytotoxicity': ['IC50_ug_per_ml', 'cell_type', 'cell_viability_percent'],
        'magnetic': ['Ms_emu_per_g', 'Mr_emu_per_g', 'Hc_Oe'],
        'nanozymes': ['km_value', 'vmax_value', 'kcat_value'],
        'selective_toxicity': ['IC50_ug_per_ml', 'cell_type', 'selectivity_index'],
        'synergy': ['NP_size_avg_nm', 'zeta_potential_mV', 'combination_index']
    }
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "vision_results"
        self.datasets_dir = self.base_dir / "datasets"
        self.metrics = defaultdict(list)
        
    def load_ground_truth(self, dataset: str, split: str) -> pd.DataFrame:
        """Load ground truth from CSV file"""
        csv_patterns = {
            'cytotoxicity': f'cv sets marked - cyto_{split}.csv',
            'magnetic': f'cv sets marked - mag_{split}.csv',
            'nanozymes': f'cv sets marked - nzyme_{split}.csv',
            'selective_toxicity': f'cv sets marked - seltox_{split}.csv',
            'synergy': f'cv sets marked - syn_{split}.csv'
        }
        
        # CSV files are in subdirectory
        csv_dir = self.datasets_dir / "plots" / "cv_sets_marked" / "сsvs"
        csv_file = csv_dir / csv_patterns.get(dataset, f'{dataset}_{split}.csv')
        
        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return pd.DataFrame()
            
        return pd.read_csv(csv_file)
    
    def load_extraction_results(self, dataset: str, split: str) -> Dict:
        """Load extraction results from JSON files"""
        # Results are saved in subdirectories
        results_file = self.results_dir / dataset / split / f"{dataset}_{split}_results.json"
        
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return {}
            
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def normalize_value(self, value: Any) -> Optional[float]:
        """Normalize extracted value to float for comparison"""
        if value is None or value == '' or value == 'None':
            return None
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Try to extract numeric value from string
            import re
            match = re.search(r'[-+]?\d*\.?\d+', value)
            if match:
                try:
                    return float(match.group())
                except:
                    pass
                    
        return None
    
    def compare_values(self, extracted: Any, ground_truth: Any, tolerance: float = 0.1) -> bool:
        """Compare extracted value with ground truth"""
        ext_val = self.normalize_value(extracted)
        gt_val = self.normalize_value(ground_truth)
        
        # Both None - consider as match (no data expected, no data extracted)
        if ext_val is None and gt_val is None:
            return True
            
        # One is None - mismatch
        if ext_val is None or gt_val is None:
            return False
            
        # Numeric comparison with tolerance
        if abs(ext_val - gt_val) / max(abs(gt_val), 1e-10) <= tolerance:
            return True
            
        return False
    
    def evaluate_dataset(self, dataset: str, split: str) -> List[ExtractionMetrics]:
        """Evaluate extraction quality for a dataset"""
        logger.info(f"Evaluating {dataset} - {split}")
        
        # Load data
        ground_truth = self.load_ground_truth(dataset, split)
        extraction_data = self.load_extraction_results(dataset, split)
        
        if ground_truth.empty or not extraction_data:
            logger.warning(f"No data to evaluate for {dataset} - {split}")
            return []
        
        # Extract results list from the loaded data
        extraction_results = {}
        if 'results' in extraction_data:
            # Convert list of results to dict keyed by pdf_name
            for result in extraction_data['results']:
                if isinstance(result, dict) and 'pdf_name' in result:
                    pdf_name = result['pdf_name']
                    extraction_results[pdf_name] = result
        
        if not extraction_results:
            logger.warning(f"No extraction results found in data for {dataset} - {split}")
            return []
        
        # Get parameters to evaluate
        params = self.DATASET_PARAMS.get(dataset, [])
        metrics_list = []
        
        for param in params:
            metric = ExtractionMetrics(dataset=dataset, split=split, parameter=param)
            
            # Iterate through files
            for _, row in ground_truth.iterrows():
                file_id = row.get('file_id', row.get('pdf_name', ''))
                
                if not file_id:
                    continue
                    
                # Get ground truth value
                gt_value = row.get(param)
                
                # Get extracted value
                ext_data = extraction_results.get(file_id, {})
                ext_value = None
                
                # Search for parameter in extracted graphs
                for graph in ext_data.get('graphs', []):
                    if graph and isinstance(graph, dict):
                        ext_value = graph.get(param)
                        if ext_value is not None:
                            break
                
                # Compare values
                has_gt = self.normalize_value(gt_value) is not None
                has_ext = self.normalize_value(ext_value) is not None
                
                if has_gt and has_ext:
                    # Both have values - check if they match
                    if self.compare_values(ext_value, gt_value):
                        metric.true_positives += 1
                    else:
                        metric.false_positives += 1  # Extracted wrong value
                elif has_gt and not has_ext:
                    metric.false_negatives += 1  # Failed to extract
                elif not has_gt and has_ext:
                    metric.false_positives += 1  # Extracted when shouldn't
                else:
                    metric.true_negatives += 1  # Correctly didn't extract
            
            # Calculate metrics
            metric.calculate_metrics()
            metrics_list.append(metric)
            
            logger.info(f"  {param}: P={metric.precision:.3f}, R={metric.recall:.3f}, "
                       f"F1={metric.f1:.3f}, Acc={metric.accuracy:.3f}")
        
        return metrics_list
    
    def evaluate_all(self) -> Dict:
        """Evaluate all datasets and splits"""
        all_metrics = {}
        
        datasets = ['cytotoxicity', 'magnetic', 'nanozymes', 'selective_toxicity', 'synergy']
        splits = ['train', 'test']
        
        for dataset in datasets:
            for split in splits:
                key = f"{dataset}_{split}"
                metrics = self.evaluate_dataset(dataset, split)
                if metrics:
                    all_metrics[key] = metrics
        
        return all_metrics
    
    def generate_report(self, metrics: Dict) -> str:
        """Generate evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("VISION PIPELINE EXTRACTION METRICS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall summary
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_accuracies = []
        
        for dataset_key, dataset_metrics in metrics.items():
            report.append(f"\n{dataset_key.upper()}")
            report.append("-" * 40)
            
            for metric in dataset_metrics:
                report.append(f"  {metric.parameter}:")
                report.append(f"    Precision: {metric.precision:.3f}")
                report.append(f"    Recall:    {metric.recall:.3f}")
                report.append(f"    F1-Score:  {metric.f1:.3f}")
                report.append(f"    Accuracy:  {metric.accuracy:.3f}")
                report.append(f"    TP={metric.true_positives}, FP={metric.false_positives}, "
                             f"FN={metric.false_negatives}, TN={metric.true_negatives}")
                
                all_precisions.append(metric.precision)
                all_recalls.append(metric.recall)
                all_f1s.append(metric.f1)
                all_accuracies.append(metric.accuracy)
        
        # Overall statistics
        report.append("\n" + "=" * 80)
        report.append("OVERALL STATISTICS")
        report.append("=" * 80)
        
        if all_precisions:
            report.append(f"Average Precision: {np.mean(all_precisions):.3f}")
            report.append(f"Average Recall:    {np.mean(all_recalls):.3f}")
            report.append(f"Average F1-Score:  {np.mean(all_f1s):.3f}")
            report.append(f"Average Accuracy:  {np.mean(all_accuracies):.3f}")
        
        return "\n".join(report)
    
    def save_metrics(self, metrics: Dict, output_file: str = "extraction_metrics.json"):
        """Save metrics to JSON file"""
        serializable_metrics = {}
        
        for key, metrics_list in metrics.items():
            serializable_metrics[key] = [
                {
                    'parameter': m.parameter,
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1': m.f1,
                    'accuracy': m.accuracy,
                    'true_positives': m.true_positives,
                    'false_positives': m.false_positives,
                    'false_negatives': m.false_negatives,
                    'true_negatives': m.true_negatives
                }
                for m in metrics_list
            ]
        
        with open(output_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate vision pipeline extraction metrics")
    parser.add_argument('--base-dir', default=".", help='Base directory')
    parser.add_argument('--dataset', help='Specific dataset to evaluate')
    parser.add_argument('--split', help='Specific split to evaluate (train/test)')
    parser.add_argument('--output', default="extraction_metrics.json", help='Output file for metrics')
    
    args = parser.parse_args()
    
    evaluator = ExtractionEvaluator(base_dir=args.base_dir)
    
    if args.dataset and args.split:
        # Evaluate specific dataset/split
        metrics_list = evaluator.evaluate_dataset(args.dataset, args.split)
        metrics = {f"{args.dataset}_{args.split}": metrics_list}
    else:
        # Evaluate all
        metrics = evaluator.evaluate_all()
    
    # Generate and print report
    report = evaluator.generate_report(metrics)
    print(report)
    
    # Save metrics
    evaluator.save_metrics(metrics, args.output)
    
    # Save report
    report_file = args.output.replace('.json', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()