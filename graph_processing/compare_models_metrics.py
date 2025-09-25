#!/usr/bin/env python3
"""
Сравнение метрик GLM-4.1V и Gemma-3-27b на пересекающихся файлах
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Set, List
import numpy as np

def load_results(results_dir: Path, dataset: str, split: str) -> Dict:
    """Загрузка результатов модели"""
    result_file = results_dir / dataset / split / f"{dataset}_{split}_results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
            # Создаем словарь по именам файлов
            results = {}
            if 'results' in data:
                for item in data['results']:
                    if 'pdf_name' in item:
                        results[item['pdf_name']] = item
            return results
    return {}

def find_common_files(glm_results: Dict, gemma_results: Dict) -> Set[str]:
    """Найти общие обработанные файлы"""
    glm_files = set(glm_results.keys())
    gemma_files = set(gemma_results.keys())
    return glm_files.intersection(gemma_files)

def analyze_extraction_quality(results: Dict, dataset_type: str) -> Dict:
    """Анализ качества извлечения для датасета"""
    stats = {
        'total_files': len(results),
        'files_with_graphs': 0,
        'files_with_tables': 0,
        'total_graphs': 0,
        'total_tables': 0,
        'avg_graphs_per_file': 0,
        'avg_tables_per_file': 0,
        'extraction_success_rate': 0
    }
    
    graph_counts = []
    table_counts = []
    
    for file_name, data in results.items():
        analyses = data.get('analyses', [])
        tables = data.get('tables', [])
        
        if analyses:
            stats['files_with_graphs'] += 1
            graph_counts.append(len(analyses))
            stats['total_graphs'] += len(analyses)
            
        if tables:
            stats['files_with_tables'] += 1
            table_counts.append(len(tables))
            stats['total_tables'] += len(tables)
    
    if graph_counts:
        stats['avg_graphs_per_file'] = np.mean(graph_counts)
    if table_counts:
        stats['avg_tables_per_file'] = np.mean(table_counts)
    
    if stats['total_files'] > 0:
        stats['extraction_success_rate'] = (stats['files_with_graphs'] + stats['files_with_tables']) / stats['total_files'] * 100
    
    return stats

def main():
    glm_dir = Path("vision_results")
    gemma_dir = Path("vision_results_gemma")
    
    datasets = ['synergy', 'cytox', 'magnetic', 'nanozymes', 'seltox']
    splits = ['train', 'test']
    
    comparison_results = {}
    
    print("="*80)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ GLM-4.1V vs GEMMA-3-27B")
    print("="*80)
    print()
    
    for dataset in datasets:
        for split in splits:
            key = f"{dataset}_{split}"
            
            # Загружаем результаты обеих моделей
            glm_results = load_results(glm_dir, dataset, split)
            gemma_results = load_results(gemma_dir, dataset, split)
            
            # Находим общие файлы
            common_files = find_common_files(glm_results, gemma_results)
            
            if not common_files and not glm_results and not gemma_results:
                continue
            
            print(f"\n{dataset.upper()} - {split.upper()}")
            print("-"*40)
            print(f"Файлов GLM-4.1V: {len(glm_results)}")
            print(f"Файлов Gemma-3-27b: {len(gemma_results)}")
            print(f"Общих файлов: {len(common_files)}")
            
            if common_files:
                # Анализируем только общие файлы для справедливого сравнения
                glm_common = {f: glm_results[f] for f in common_files}
                gemma_common = {f: gemma_results[f] for f in common_files}
                
                glm_stats = analyze_extraction_quality(glm_common, dataset)
                gemma_stats = analyze_extraction_quality(gemma_common, dataset)
                
                print("\nКачество извлечения на общих файлах:")
                print(f"{'Метрика':<30} {'GLM-4.1V':>15} {'Gemma-3-27b':>15}")
                print("-"*60)
                print(f"{'Файлов с графиками':<30} {glm_stats['files_with_graphs']:>15} {gemma_stats['files_with_graphs']:>15}")
                print(f"{'Файлов с таблицами':<30} {glm_stats['files_with_tables']:>15} {gemma_stats['files_with_tables']:>15}")
                print(f"{'Всего графиков':<30} {glm_stats['total_graphs']:>15} {gemma_stats['total_graphs']:>15}")
                print(f"{'Всего таблиц':<30} {glm_stats['total_tables']:>15} {gemma_stats['total_tables']:>15}")
                print(f"{'Среднее графиков/файл':<30} {glm_stats['avg_graphs_per_file']:>15.1f} {gemma_stats['avg_graphs_per_file']:>15.1f}")
                print(f"{'Среднее таблиц/файл':<30} {glm_stats['avg_tables_per_file']:>15.1f} {gemma_stats['avg_tables_per_file']:>15.1f}")
                print(f"{'Успешность извлечения %':<30} {glm_stats['extraction_success_rate']:>15.1f} {gemma_stats['extraction_success_rate']:>15.1f}")
                
                comparison_results[key] = {
                    'glm': glm_stats,
                    'gemma': gemma_stats,
                    'common_files': len(common_files)
                }
    
    # Сводная статистика
    print("\n" + "="*80)
    print("СВОДНАЯ СТАТИСТИКА")
    print("="*80)
    
    total_glm_graphs = sum(r['glm']['total_graphs'] for r in comparison_results.values())
    total_gemma_graphs = sum(r['gemma']['total_graphs'] for r in comparison_results.values())
    total_glm_tables = sum(r['glm']['total_tables'] for r in comparison_results.values())
    total_gemma_tables = sum(r['gemma']['total_tables'] for r in comparison_results.values())
    total_common_files = sum(r['common_files'] for r in comparison_results.values())
    
    print(f"\nВсего общих файлов для сравнения: {total_common_files}")
    print(f"\nИзвлечено элементов:")
    print(f"  GLM-4.1V:    {total_glm_graphs} графиков, {total_glm_tables} таблиц")
    print(f"  Gemma-3-27b: {total_gemma_graphs} графиков, {total_gemma_tables} таблиц")
    
    if total_glm_graphs > 0:
        ratio_graphs = total_gemma_graphs / total_glm_graphs * 100
        print(f"\nСоотношение графиков Gemma/GLM: {ratio_graphs:.1f}%")
    
    if total_glm_tables > 0:
        ratio_tables = total_gemma_tables / total_glm_tables * 100
        print(f"Соотношение таблиц Gemma/GLM: {ratio_tables:.1f}%")
    
    # Сохраняем подробные результаты
    with open('model_comparison_detailed.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✅ Детальные результаты сохранены в model_comparison_detailed.json")

if __name__ == "__main__":
    main()