#!/usr/bin/env python3
"""
Evaluation metrics for vision pipeline - adapted for actual data structure
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import re

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
        total_relevant = self.true_positives + self.false_negatives
        total_extracted = self.true_positives + self.false_positives
        
        # Precision: TP / (TP + FP)
        if total_extracted > 0:
            self.precision = self.true_positives / total_extracted
        
        # Recall: TP / (TP + FN)
        if total_relevant > 0:
            self.recall = self.true_positives / total_relevant
        
        # F1: 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        # Simple accuracy for extracted vs expected
        if total_relevant > 0:
            self.accuracy = self.true_positives / total_relevant


class ExtractionEvaluatorV2:
    """Evaluates extraction quality against ground truth - Russian CSV format"""
    
    # Map Russian parameter names to expected extraction keys
    PARAM_MAPPING = {
        'synergy': {
            'NP_size_avg_nm': 'NP_size_avg_nm',
            'zeta_potential_mV': 'zeta_potential_mV', 
            'CI': 'combination_index'
        },
        'cytotoxicity': {
            'IC50_ug_per_ml': 'IC50_ug_per_ml',
            'Cell_type': 'cell_type',
            'Cell_viability': 'cell_viability_percent'
        },
        'cytox': {
            'IC50_ug_per_ml': 'IC50_ug_per_ml',
            'Cell_type': 'cell_type',
            'Cell_viability': 'cell_viability_percent'
        },
        'magnetic': {
            'Ms_emu_per_g': 'Ms_emu_per_g',
            'Mr_emu_per_g': 'Mr_emu_per_g',
            'Hc_Oe': 'Hc_Oe'
        },
        'nanozymes': {
            'Km_value': 'km_value',
            'Vmax_value': 'vmax_value',
            'Kcat_value': 'kcat_value'
        },
        'selective_toxicity': {
            'IC50_ug_per_ml': 'IC50_ug_per_ml',
            'Cell_type': 'cell_type',
            'SI': 'selectivity_index'
        },
        'seltox': {
            'IC50_ug_per_ml': 'IC50_ug_per_ml',
            'Cell_type': 'cell_type',
            'SI': 'selectivity_index'
        }
    }
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "vision_results"
        self.csv_dir = self.base_dir / "datasets" / "plots" / "cv_sets_marked" / "сsvs"
        
    def load_ground_truth_russian(self, dataset: str, split: str) -> Dict:
        """Load ground truth from Russian CSV format"""
        csv_patterns = {
            'cytotoxicity': f'cv sets marked - cyto_{split}.csv',
            'cytox': f'cv sets marked - cyto_{split}.csv',
            'magnetic': f'cv sets marked - mag_{split}.csv',
            'nanozymes': f'cv sets marked - nzyme_{split}.csv',
            'selective_toxicity': f'cv sets marked - seltox_{split}.csv',
            'seltox': f'cv sets marked - seltox_{split}.csv',
            'synergy': f'cv sets marked - syn_{split}.csv'
        }
        
        csv_file = self.csv_dir / csv_patterns.get(dataset, f'{dataset}_{split}.csv')
        
        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return {}
        
        df = pd.read_csv(csv_file)
        
        # Parse Russian format: папка, статья, элемент, параметр, значение
        ground_truth = defaultdict(dict)
        
        for _, row in df.iterrows():
            article = row.get('статья', '')
            param = row.get('параметр', '')
            value = row.get('значение')
            
            if article and param:
                # Clean article name (remove .pdf if present)
                article = str(article).replace('.pdf', '') + '.pdf' if not str(article).endswith('.pdf') else str(article)
                ground_truth[article][param] = value
        
        return ground_truth
    
    def load_extraction_results(self, dataset: str, split: str) -> Dict:
        """Load extraction results from JSON files"""
        results_file = self.results_dir / dataset / split / f"{dataset}_{split}_results.json"
        
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return {}
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Convert results to dict by pdf_name
        extraction_results = {}
        if 'results' in data:
            for result in data['results']:
                if isinstance(result, dict) and 'pdf_name' in result:
                    pdf_name = result['pdf_name']
                    extraction_results[pdf_name] = result
        
        return extraction_results
    
    def extract_value_from_text(self, text: str, param: str) -> Optional[float]:
        """Try to extract parameter value from text"""
        if not text or not isinstance(text, str):
            return None
        
        # Common patterns for extracting values
        patterns = {
            'NP_size': r'(?:size|diameter|nm)[:\s]*(\d+(?:\.\d+)?)\s*nm',
            'zeta_potential': r'zeta[:\s]*([+-]?\d+(?:\.\d+)?)\s*mV',
            'IC50': r'IC50[:\s]*(\d+(?:\.\d+)?)',
            'Ms': r'Ms[:\s]*(\d+(?:\.\d+)?)',
            'Km': r'Km[:\s]*(\d+(?:\.\d+)?)',
            'CI': r'(?:CI|combination index)[:\s]*(\d+(?:\.\d+)?)'
        }
        
        # Try to find pattern matching the parameter
        for key, pattern in patterns.items():
            if key.lower() in param.lower():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except:
                        pass
        
        # Try generic number extraction
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        
        return None
    
    def normalize_value(self, value: Any) -> Optional[float]:
        """Normalize value to float for comparison"""
        if value is None or value == '' or value == 'None':
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Clean and extract number
            value = str(value).strip()
            match = re.search(r'[-+]?\d*\.?\d+', value)
            if match:
                try:
                    return float(match.group())
                except:
                    pass
        
        return None
    
    def compare_values(self, extracted: Any, ground_truth: Any, tolerance: float = 0.2) -> bool:
        """Compare extracted value with ground truth with tolerance"""
        ext_val = self.normalize_value(extracted)
        gt_val = self.normalize_value(ground_truth)
        
        if ext_val is None or gt_val is None:
            return False
        
        # Relative tolerance comparison
        if gt_val == 0:
            return abs(ext_val) < 0.01
        
        relative_diff = abs(ext_val - gt_val) / abs(gt_val)
        return relative_diff <= tolerance
    
    def evaluate_dataset(self, dataset: str, split: str) -> List[ExtractionMetrics]:
        """Evaluate extraction quality for a dataset"""
        logger.info(f"Evaluating {dataset} - {split}")
        
        # Load data
        ground_truth = self.load_ground_truth_russian(dataset, split)
        extraction_results = self.load_extraction_results(dataset, split)
        
        if not ground_truth or not extraction_results:
            logger.warning(f"No data to evaluate for {dataset} - {split}")
            return []
        
        # Get parameter mapping
        param_map = self.PARAM_MAPPING.get(dataset, {})
        metrics_list = []
        
        logger.info(f"  Ground truth files: {len(ground_truth)}")
        logger.info(f"  Extraction results: {len(extraction_results)}")
        
        for gt_param, ext_param in param_map.items():
            metric = ExtractionMetrics(dataset=dataset, split=split, parameter=gt_param)
            
            # Iterate through ground truth
            for pdf_name, gt_params in ground_truth.items():
                if gt_param not in gt_params:
                    continue
                
                gt_value = gt_params[gt_param]
                if self.normalize_value(gt_value) is None:
                    continue  # Skip if no valid ground truth value
                
                # Try to find extraction result
                ext_result = extraction_results.get(pdf_name, {})
                extracted_value = None
                
                if ext_result:
                    # Check analyses field (graphs)
                    analyses = ext_result.get('analyses', [])
                    for analysis in analyses:
                        if isinstance(analysis, str):
                            # Try to extract from text
                            extracted_value = self.extract_value_from_text(analysis, gt_param)
                            if extracted_value is not None:
                                break
                        elif isinstance(analysis, dict):
                            # Check if parameter is in dict
                            if ext_param in analysis:
                                extracted_value = analysis[ext_param]
                                break
                    
                    # Also check tables if no value found
                    if extracted_value is None:
                        tables = ext_result.get('tables', [])
                        for table in tables:
                            if isinstance(table, str):
                                extracted_value = self.extract_value_from_text(table, gt_param)
                                if extracted_value is not None:
                                    break
                
                # Compare values
                if self.compare_values(extracted_value, gt_value):
                    metric.true_positives += 1
                    logger.debug(f"    TP: {pdf_name} - {gt_param}: GT={gt_value}, Ext={extracted_value}")
                else:
                    metric.false_negatives += 1
                    logger.debug(f"    FN: {pdf_name} - {gt_param}: GT={gt_value}, Ext={extracted_value}")
            
            # Check for false positives (extracted when shouldn't)
            for pdf_name, ext_result in extraction_results.items():
                if pdf_name not in ground_truth or gt_param not in ground_truth.get(pdf_name, {}):
                    # Check if something was extracted
                    analyses = ext_result.get('analyses', [])
                    for analysis in analyses:
                        if isinstance(analysis, str) and gt_param.lower() in analysis.lower():
                            value = self.extract_value_from_text(analysis, gt_param)
                            if value is not None:
                                metric.false_positives += 1
                                break
            
            # Calculate metrics
            metric.calculate_metrics()
            metrics_list.append(metric)
            
            logger.info(f"  {gt_param}: P={metric.precision:.3f}, R={metric.recall:.3f}, "
                       f"F1={metric.f1:.3f}, Acc={metric.accuracy:.3f}")
            logger.info(f"    TP={metric.true_positives}, FP={metric.false_positives}, "
                       f"FN={metric.false_negatives}")
        
        return metrics_list
    
    def generate_report(self, all_metrics: Dict) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("VISION PIPELINE EXTRACTION METRICS REPORT V2")
        report.append("=" * 80)
        report.append("")
        
        # Collect all metrics for summary
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_accuracies = []
        
        for dataset_key, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
                
            report.append(f"\n{dataset_key.upper()}")
            report.append("-" * 40)
            
            for metric in metrics_list:
                report.append(f"  {metric.parameter}:")
                report.append(f"    Precision: {metric.precision:.3f}")
                report.append(f"    Recall:    {metric.recall:.3f}")
                report.append(f"    F1-Score:  {metric.f1:.3f}")
                report.append(f"    Accuracy:  {metric.accuracy:.3f}")
                report.append(f"    Confusion: TP={metric.true_positives}, FP={metric.false_positives}, "
                             f"FN={metric.false_negatives}")
                
                if metric.true_positives > 0 or metric.false_negatives > 0:
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
            report.append(f"")
            report.append(f"Median Precision: {np.median(all_precisions):.3f}")
            report.append(f"Median Recall:    {np.median(all_recalls):.3f}")
            report.append(f"Median F1-Score:  {np.median(all_f1s):.3f}")
        else:
            report.append("No valid metrics calculated")
        
        return "\n".join(report)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate vision pipeline extraction metrics V2")
    parser.add_argument('--dataset', help='Specific dataset to evaluate')
    parser.add_argument('--split', help='Specific split (train/test)')
    parser.add_argument('--output', default="extraction_metrics_v2.json", help='Output file')
    
    args = parser.parse_args()
    
    evaluator = ExtractionEvaluatorV2()
    
    all_metrics = {}
    
    if args.dataset and args.split:
        # Evaluate specific dataset/split
        metrics = evaluator.evaluate_dataset(args.dataset, args.split)
        all_metrics[f"{args.dataset}_{args.split}"] = metrics
    else:
        # Evaluate all completed datasets
        datasets_to_eval = ['synergy', 'cytox', 'magnetic', 'nanozymes', 'seltox']  # All processed datasets
        splits = ['train', 'test']
        
        for dataset in datasets_to_eval:
            for split in splits:
                try:
                    metrics = evaluator.evaluate_dataset(dataset, split)
                    if metrics:
                        all_metrics[f"{dataset}_{split}"] = metrics
                except Exception as e:
                    logger.error(f"Failed to evaluate {dataset} {split}: {e}")
    
    # Generate and print report
    report = evaluator.generate_report(all_metrics)
    print(report)
    
    # Save metrics
    output_data = {}
    for key, metrics_list in all_metrics.items():
        output_data[key] = [
            {
                'parameter': m.parameter,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'accuracy': m.accuracy,
                'true_positives': m.true_positives,
                'false_positives': m.false_positives,
                'false_negatives': m.false_negatives
            }
            for m in metrics_list
        ]
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Metrics saved to {args.output}")
    
    # Save report
    report_file = args.output.replace('.json', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()