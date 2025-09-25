#!/usr/bin/env python3
"""
Расчет метрик качества (Precision, Recall, F1, Accuracy) для GLM-4.1V и Gemma-3-27b
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re

class QualityMetricsCalculator:
    """Калькулятор метрик качества для сравнения моделей"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.csv_dir = self.base_dir / "datasets" / "plots" / "cv_sets_marked" / "сsvs"
        self.glm_dir = self.base_dir / "vision_results"
        self.gemma_dir = self.base_dir / "vision_results_gemma"
        
        # Маппинг параметров по датасетам
        self.param_mapping = {
            'synergy': ['NP_size_avg_nm', 'zeta_potential_mV', 'CI'],
            'cytox': ['IC50_ug_per_ml', 'Cell_type', 'Cell_viability'],
            'magnetic': ['Ms_emu_per_g', 'Mr_emu_per_g', 'Hc_Oe'],
            'nanozymes': ['Km_value', 'Vmax_value', 'Kcat_value'],
            'seltox': ['IC50_ug_per_ml', 'Cell_type', 'SI']
        }
        
        # CSV файлы маппинг
        self.csv_mapping = {
            'synergy_train': 'cv sets marked - syn_train.csv',
            'synergy_test': 'cv sets marked - syn_test.csv',
            'cytox_train': 'cv sets marked - cyto_train.csv',
            'cytox_test': 'cv sets marked - cyto_test.csv',
            'magnetic_train': 'cv sets marked - mag_train.csv',
            'magnetic_test': 'cv sets marked - mag_test.csv',
            'nanozymes_train': 'cv sets marked - nzyme_train.csv',
            'nanozymes_test': 'cv sets marked - nzyme_test.csv',
            'seltox_train': 'cv sets marked - seltox_train.csv',
            'seltox_test': 'cv sets marked - seltox_test.csv'
        }
    
    def load_ground_truth(self, dataset: str, split: str) -> Dict:
        """Загрузка ground truth из CSV"""
        csv_key = f"{dataset}_{split}"
        csv_file = self.csv_dir / self.csv_mapping.get(csv_key, f"{dataset}_{split}.csv")
        
        if not csv_file.exists():
            return {}
        
        try:
            df = pd.read_csv(csv_file)
            ground_truth = defaultdict(dict)
            
            for _, row in df.iterrows():
                article = str(row.get('статья', ''))
                param = str(row.get('параметр', ''))
                value = row.get('значение')
                
                if article and param and value is not None:
                    if not article.endswith('.pdf'):
                        article = article + '.pdf'
                    ground_truth[article][param] = value
            
            return ground_truth
        except Exception as e:
            print(f"Error loading CSV {csv_file}: {e}")
            return {}
    
    def load_model_results(self, model_dir: Path, dataset: str, split: str) -> Dict:
        """Загрузка результатов модели"""
        result_file = model_dir / dataset / split / f"{dataset}_{split}_results.json"
        
        if not result_file.exists():
            return {}
        
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        results = {}
        if 'results' in data:
            for item in data['results']:
                if 'pdf_name' in item:
                    results[item['pdf_name']] = item
        
        return results
    
    def extract_value_from_text(self, text: str, param: str) -> Optional[float]:
        """Извлечение числового значения из текста"""
        if not text or not isinstance(text, str):
            return None
        
        # Паттерны для разных параметров
        patterns = {
            'NP_size': r'(?:size|diameter|nm)[:\s]*(\d+(?:\.\d+)?)\s*nm',
            'zeta_potential': r'zeta[:\s]*([+-]?\d+(?:\.\d+)?)\s*mV',
            'IC50': r'IC50[:\s]*(\d+(?:\.\d+)?)',
            'Ms': r'Ms[:\s]*(\d+(?:\.\d+)?)\s*emu',
            'Mr': r'Mr[:\s]*(\d+(?:\.\d+)?)\s*emu',
            'Hc': r'Hc[:\s]*(\d+(?:\.\d+)?)',
            'Km': r'Km[:\s]*(\d+(?:\.\d+)?)',
            'Vmax': r'Vmax[:\s]*(\d+(?:\.\d+)?)',
            'Kcat': r'Kcat[:\s]*(\d+(?:\.\d+)?)',
            'CI': r'(?:CI|combination.?index)[:\s]*(\d+(?:\.\d+)?)',
            'SI': r'(?:SI|selectivity.?index)[:\s]*(\d+(?:\.\d+)?)'
        }
        
        for key, pattern in patterns.items():
            if key.lower() in param.lower():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except:
                        pass
        
        # Попытка найти любое число
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        
        return None
    
    def compare_values(self, extracted: Optional[float], ground_truth: float, tolerance: float = 0.2) -> bool:
        """Сравнение извлеченного значения с ground truth"""
        if extracted is None:
            return False
        
        if ground_truth == 0:
            return abs(extracted) < 0.01
        
        relative_diff = abs(extracted - ground_truth) / abs(ground_truth)
        return relative_diff <= tolerance
    
    def calculate_metrics(self, tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """Расчет метрик"""
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
        
        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            metrics['precision'] = tp / (tp + fp)
        
        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            metrics['recall'] = tp / (tp + fn)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        total = tp + tn + fp + fn
        if total > 0:
            metrics['accuracy'] = (tp + tn) / total
        
        return metrics
    
    def evaluate_model(self, model_dir: Path, model_name: str) -> Dict:
        """Оценка модели по всем датасетам"""
        all_metrics = {}
        overall_tp, overall_fp, overall_fn, overall_tn = 0, 0, 0, 0
        
        datasets = ['synergy', 'cytox', 'magnetic', 'nanozymes', 'seltox']
        splits = ['train', 'test']
        
        for dataset in datasets:
            for split in splits:
                key = f"{dataset}_{split}"
                
                # Загружаем данные
                ground_truth = self.load_ground_truth(dataset, split)
                model_results = self.load_model_results(model_dir, dataset, split)
                
                if not ground_truth or not model_results:
                    continue
                
                # Считаем метрики для каждого параметра
                params = self.param_mapping.get(dataset, [])
                dataset_metrics = {}
                
                for param in params:
                    tp, fp, fn, tn = 0, 0, 0, 0
                    
                    # Проходим по ground truth
                    for pdf_name, gt_params in ground_truth.items():
                        if param in gt_params:
                            gt_value = gt_params[param]
                            
                            # Ищем в результатах модели
                            if pdf_name in model_results:
                                result = model_results[pdf_name]
                                extracted = False
                                
                                # Проверяем analyses
                                for analysis in result.get('analyses', []):
                                    if isinstance(analysis, dict):
                                        if param in analysis:
                                            if self.compare_values(analysis[param], gt_value):
                                                tp += 1
                                                extracted = True
                                                break
                                    elif isinstance(analysis, str):
                                        value = self.extract_value_from_text(analysis, param)
                                        if self.compare_values(value, gt_value):
                                            tp += 1
                                            extracted = True
                                            break
                                
                                # Проверяем tables если не нашли
                                if not extracted:
                                    for table in result.get('tables', []):
                                        if isinstance(table, str):
                                            value = self.extract_value_from_text(table, param)
                                            if self.compare_values(value, gt_value):
                                                tp += 1
                                                extracted = True
                                                break
                                
                                if not extracted:
                                    fn += 1
                            else:
                                fn += 1
                    
                    # Проверяем false positives
                    for pdf_name, result in model_results.items():
                        if pdf_name not in ground_truth or param not in ground_truth.get(pdf_name, {}):
                            # Проверяем, извлекла ли модель что-то для этого параметра
                            for analysis in result.get('analyses', []):
                                if isinstance(analysis, dict) and param in analysis:
                                    fp += 1
                                    break
                                elif isinstance(analysis, str):
                                    if self.extract_value_from_text(analysis, param) is not None:
                                        fp += 1
                                        break
                    
                    # Считаем метрики для параметра
                    param_metrics = self.calculate_metrics(tp, fp, fn, tn)
                    param_metrics['tp'] = tp
                    param_metrics['fp'] = fp
                    param_metrics['fn'] = fn
                    param_metrics['tn'] = tn
                    
                    dataset_metrics[param] = param_metrics
                    
                    overall_tp += tp
                    overall_fp += fp
                    overall_fn += fn
                    overall_tn += tn
                
                if dataset_metrics:
                    all_metrics[key] = dataset_metrics
        
        # Общие метрики
        overall_metrics = self.calculate_metrics(overall_tp, overall_fp, overall_fn, overall_tn)
        overall_metrics['total_tp'] = overall_tp
        overall_metrics['total_fp'] = overall_fp
        overall_metrics['total_fn'] = overall_fn
        overall_metrics['total_tn'] = overall_tn
        
        return {
            'model': model_name,
            'dataset_metrics': all_metrics,
            'overall_metrics': overall_metrics
        }

def main():
    calculator = QualityMetricsCalculator()
    
    print("="*80)
    print("СРАВНЕНИЕ МЕТРИК КАЧЕСТВА: GLM-4.1V vs GEMMA-3-27B")
    print("="*80)
    print()
    
    # Оценка GLM-4.1V
    glm_results = calculator.evaluate_model(Path("vision_results"), "GLM-4.1V")
    
    # Оценка Gemma-3-27b
    gemma_results = calculator.evaluate_model(Path("vision_results_gemma"), "Gemma-3-27b")
    
    # Вывод результатов по датасетам
    print("\nМЕТРИКИ ПО ДАТАСЕТАМ")
    print("-"*80)
    
    all_datasets = set(glm_results['dataset_metrics'].keys()) | set(gemma_results['dataset_metrics'].keys())
    
    for dataset_key in sorted(all_datasets):
        print(f"\n{dataset_key.upper()}")
        print("-"*40)
        
        glm_metrics = glm_results['dataset_metrics'].get(dataset_key, {})
        gemma_metrics = gemma_results['dataset_metrics'].get(dataset_key, {})
        
        all_params = set(glm_metrics.keys()) | set(gemma_metrics.keys())
        
        for param in sorted(all_params):
            print(f"\n  {param}:")
            
            glm_m = glm_metrics.get(param, {})
            gemma_m = gemma_metrics.get(param, {})
            
            print(f"    {'Метрика':<15} {'GLM-4.1V':>12} {'Gemma-3-27b':>12}")
            print(f"    {'-'*39}")
            print(f"    {'Precision':<15} {glm_m.get('precision', 0):.3f}{' '*8} {gemma_m.get('precision', 0):.3f}")
            print(f"    {'Recall':<15} {glm_m.get('recall', 0):.3f}{' '*8} {gemma_m.get('recall', 0):.3f}")
            print(f"    {'F1-Score':<15} {glm_m.get('f1', 0):.3f}{' '*8} {gemma_m.get('f1', 0):.3f}")
            print(f"    {'Accuracy':<15} {glm_m.get('accuracy', 0):.3f}{' '*8} {gemma_m.get('accuracy', 0):.3f}")
            print(f"    {'TP/FP/FN':<15} {glm_m.get('tp', 0)}/{glm_m.get('fp', 0)}/{glm_m.get('fn', 0)}{' '*5} {gemma_m.get('tp', 0)}/{gemma_m.get('fp', 0)}/{gemma_m.get('fn', 0)}")
    
    # Общие метрики
    print("\n" + "="*80)
    print("ОБЩИЕ МЕТРИКИ КАЧЕСТВА")
    print("="*80)
    
    glm_overall = glm_results['overall_metrics']
    gemma_overall = gemma_results['overall_metrics']
    
    print(f"\n{'Метрика':<20} {'GLM-4.1V':>15} {'Gemma-3-27b':>15} {'Разница':>15}")
    print("-"*65)
    
    metrics_to_show = ['precision', 'recall', 'f1', 'accuracy']
    for metric in metrics_to_show:
        glm_val = glm_overall.get(metric, 0)
        gemma_val = gemma_overall.get(metric, 0)
        diff = gemma_val - glm_val
        diff_str = f"{diff:+.3f}" if diff != 0 else "0.000"
        print(f"{metric.capitalize():<20} {glm_val:>15.3f} {gemma_val:>15.3f} {diff_str:>15}")
    
    print(f"\n{'Confusion Matrix':<20}")
    print(f"{'True Positives':<20} {glm_overall.get('total_tp', 0):>15} {gemma_overall.get('total_tp', 0):>15}")
    print(f"{'False Positives':<20} {glm_overall.get('total_fp', 0):>15} {gemma_overall.get('total_fp', 0):>15}")
    print(f"{'False Negatives':<20} {glm_overall.get('total_fn', 0):>15} {gemma_overall.get('total_fn', 0):>15}")
    
    # Сохраняем детальные результаты
    results = {
        'glm_4_1v': glm_results,
        'gemma_3_27b': gemma_results,
        'comparison': {
            'precision_diff': gemma_overall.get('precision', 0) - glm_overall.get('precision', 0),
            'recall_diff': gemma_overall.get('recall', 0) - glm_overall.get('recall', 0),
            'f1_diff': gemma_overall.get('f1', 0) - glm_overall.get('f1', 0),
            'accuracy_diff': gemma_overall.get('accuracy', 0) - glm_overall.get('accuracy', 0)
        }
    }
    
    with open('quality_metrics_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Детальные результаты сохранены в quality_metrics_comparison.json")
    
    # Итоговый вывод
    print("\n" + "="*80)
    print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
    print("="*80)
    
    if glm_overall.get('f1', 0) > gemma_overall.get('f1', 0):
        print(f"\n🏆 GLM-4.1V показывает лучшую F1-метрику: {glm_overall.get('f1', 0):.3f} vs {gemma_overall.get('f1', 0):.3f}")
    elif gemma_overall.get('f1', 0) > glm_overall.get('f1', 0):
        print(f"\n🏆 Gemma-3-27b показывает лучшую F1-метрику: {gemma_overall.get('f1', 0):.3f} vs {glm_overall.get('f1', 0):.3f}")
    else:
        print(f"\n🤝 Модели показывают одинаковую F1-метрику: {glm_overall.get('f1', 0):.3f}")

if __name__ == "__main__":
    main()