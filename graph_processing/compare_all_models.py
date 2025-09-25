#!/usr/bin/env python3
"""
Сравнение результатов всех моделей: GLM-4.1V, Gemma-3-27b, Qwen2.5-VL-72B
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

class MultiModelComparator:
    """Сравнение результатов трех моделей"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.glm_dir = self.base_dir / "vision_results"
        self.gemma_dir = self.base_dir / "vision_results_gemma"
        self.qwen_dir = self.base_dir / "vision_results_qwen"
        
        self.datasets = ['synergy', 'cytox', 'magnetic', 'nanozymes', 'seltox']
        self.splits = ['train', 'test']
        
    def load_results(self, model_dir: Path, dataset: str, split: str) -> Dict:
        """Загрузка результатов модели"""
        result_file = model_dir / dataset / split / f"{dataset}_{split}_results.json"
        
        if result_file.exists():
            with open(result_file, 'r') as f:
                return json.load(f)
        return {}
    
    def analyze_extraction_quality(self, result: Dict) -> Dict:
        """Анализ качества извлечения для одного результата"""
        stats = {
            'values_extracted': 0,
            'null_values': 0,
            'numeric_values': 0,
            'pages_analyzed': 0,
            'extraction_rate': 0.0
        }
        
        for pdf_result in result.get('results', []):
            for analysis in pdf_result.get('analyses', []):
                stats['pages_analyzed'] += 1
                
                if isinstance(analysis, dict) and 'data' in analysis:
                    data = analysis['data']
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if value is not None:
                                stats['values_extracted'] += 1
                                if isinstance(value, (int, float)):
                                    stats['numeric_values'] += 1
                            else:
                                stats['null_values'] += 1
        
        # Рассчитываем процент успешных извлечений
        total_slots = stats['values_extracted'] + stats['null_values']
        if total_slots > 0:
            stats['extraction_rate'] = (stats['values_extracted'] / total_slots) * 100
        
        return stats
    
    def compare_models(self) -> Dict:
        """Сравнение всех моделей"""
        comparison = {
            'glm': {'model': 'GLM-4.1V-9B', 'datasets': {}},
            'gemma': {'model': 'Gemma-3-27b', 'datasets': {}},
            'qwen': {'model': 'Qwen2.5-VL-72B', 'datasets': {}}
        }
        
        for dataset in self.datasets:
            for split in self.splits:
                key = f"{dataset}/{split}"
                
                # Загружаем результаты каждой модели
                glm_result = self.load_results(self.glm_dir, dataset, split)
                gemma_result = self.load_results(self.gemma_dir, dataset, split)
                qwen_result = self.load_results(self.qwen_dir, dataset, split)
                
                # Анализируем качество
                if glm_result:
                    comparison['glm']['datasets'][key] = {
                        'total_files': glm_result.get('total_files', 0),
                        'processed': glm_result.get('processed', 0),
                        'failed': glm_result.get('failed', 0),
                        'quality': self.analyze_extraction_quality(glm_result)
                    }
                
                if gemma_result:
                    comparison['gemma']['datasets'][key] = {
                        'total_files': gemma_result.get('total_files', 0),
                        'processed': gemma_result.get('processed', 0),
                        'failed': gemma_result.get('failed', 0),
                        'quality': self.analyze_extraction_quality(gemma_result)
                    }
                
                if qwen_result:
                    comparison['qwen']['datasets'][key] = {
                        'total_files': qwen_result.get('total_files', 0),
                        'processed': qwen_result.get('processed', 0),
                        'failed': qwen_result.get('failed', 0),
                        'quality': self.analyze_extraction_quality(qwen_result)
                    }
        
        return comparison
    
    def print_comparison_report(self, comparison: Dict):
        """Вывод отчета сравнения"""
        print("="*100)
        print("СРАВНЕНИЕ МОДЕЛЕЙ: GLM-4.1V vs GEMMA-3-27B vs QWEN2.5-VL-72B")
        print("="*100)
        print()
        
        # Таблица покрытия датасетов
        print("📊 ПОКРЫТИЕ ДАТАСЕТОВ")
        print("-"*100)
        print(f"{'Dataset':<20} {'GLM-4.1V':<25} {'Gemma-3-27b':<25} {'Qwen2.5-VL-72B':<25}")
        print("-"*100)
        
        for dataset in self.datasets:
            for split in self.splits:
                key = f"{dataset}/{split}"
                
                glm_data = comparison['glm']['datasets'].get(key, {})
                gemma_data = comparison['gemma']['datasets'].get(key, {})
                qwen_data = comparison['qwen']['datasets'].get(key, {})
                
                glm_str = f"{glm_data.get('processed', 0)}/{glm_data.get('total_files', 0)}"
                gemma_str = f"{gemma_data.get('processed', 0)}/{gemma_data.get('total_files', 0)}"
                qwen_str = f"{qwen_data.get('processed', 0)}/{qwen_data.get('total_files', 0)}"
                
                if glm_data.get('total_files', 0) > 0:
                    glm_pct = glm_data.get('processed', 0) * 100 / glm_data.get('total_files', 1)
                    glm_str += f" ({glm_pct:.0f}%)"
                
                if gemma_data.get('total_files', 0) > 0:
                    gemma_pct = gemma_data.get('processed', 0) * 100 / gemma_data.get('total_files', 1)
                    gemma_str += f" ({gemma_pct:.0f}%)"
                
                if qwen_data.get('total_files', 0) > 0:
                    qwen_pct = qwen_data.get('processed', 0) * 100 / qwen_data.get('total_files', 1)
                    qwen_str += f" ({qwen_pct:.0f}%)"
                
                print(f"{key:<20} {glm_str:<25} {gemma_str:<25} {qwen_str:<25}")
        
        # Статистика извлечения
        print("\n📈 КАЧЕСТВО ИЗВЛЕЧЕНИЯ")
        print("-"*100)
        
        total_stats = {
            'glm': {'values': 0, 'nulls': 0, 'pages': 0},
            'gemma': {'values': 0, 'nulls': 0, 'pages': 0},
            'qwen': {'values': 0, 'nulls': 0, 'pages': 0}
        }
        
        for model_key, model_data in comparison.items():
            for dataset_key, dataset_data in model_data['datasets'].items():
                if 'quality' in dataset_data:
                    q = dataset_data['quality']
                    total_stats[model_key]['values'] += q.get('values_extracted', 0)
                    total_stats[model_key]['nulls'] += q.get('null_values', 0)
                    total_stats[model_key]['pages'] += q.get('pages_analyzed', 0)
        
        print(f"{'Метрика':<30} {'GLM-4.1V':<20} {'Gemma-3-27b':<20} {'Qwen2.5-VL-72B':<20}")
        print("-"*90)
        
        for model_key in ['glm', 'gemma', 'qwen']:
            stats = total_stats[model_key]
            if stats['values'] + stats['nulls'] > 0:
                rate = stats['values'] / (stats['values'] + stats['nulls']) * 100
            else:
                rate = 0
            total_stats[model_key]['rate'] = rate
        
        print(f"{'Извлечено значений':<30} {total_stats['glm']['values']:<20} {total_stats['gemma']['values']:<20} {total_stats['qwen']['values']:<20}")
        print(f"{'Null значений':<30} {total_stats['glm']['nulls']:<20} {total_stats['gemma']['nulls']:<20} {total_stats['qwen']['nulls']:<20}")
        print(f"{'Страниц проанализировано':<30} {total_stats['glm']['pages']:<20} {total_stats['gemma']['pages']:<20} {total_stats['qwen']['pages']:<20}")
        print(f"{'Успешность извлечения %':<30} {total_stats['glm']['rate']:<20.1f} {total_stats['gemma']['rate']:<20.1f} {total_stats['qwen']['rate']:<20.1f}")
        
        # Итоговая оценка
        print("\n🏆 ИТОГОВАЯ ОЦЕНКА")
        print("-"*100)
        
        # Определяем лучшую модель по покрытию
        total_processed = {
            'glm': sum(d.get('processed', 0) for d in comparison['glm']['datasets'].values()),
            'gemma': sum(d.get('processed', 0) for d in comparison['gemma']['datasets'].values()),
            'qwen': sum(d.get('processed', 0) for d in comparison['qwen']['datasets'].values())
        }
        
        best_coverage = max(total_processed.items(), key=lambda x: x[1])
        print(f"✅ Лучшее покрытие: {best_coverage[0].upper()} ({best_coverage[1]} файлов)")
        
        # Лучшая модель по качеству извлечения
        if any(total_stats[m]['rate'] > 0 for m in ['glm', 'gemma', 'qwen']):
            best_quality = max([(k, v['rate']) for k, v in total_stats.items()], key=lambda x: x[1])
            print(f"✅ Лучшее качество извлечения: {best_quality[0].upper()} ({best_quality[1]:.1f}%)")
        
        # Сохраняем детальный отчет
        with open('model_comparison_report.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n📄 Детальный отчет сохранен в model_comparison_report.json")

def main():
    comparator = MultiModelComparator()
    comparison = comparator.compare_models()
    comparator.print_comparison_report(comparison)

if __name__ == "__main__":
    main()