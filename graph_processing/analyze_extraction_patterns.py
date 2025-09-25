#!/usr/bin/env python3
"""
Анализ паттернов извлечения для улучшения промптов
"""

import json
from pathlib import Path
from typing import Dict, List, Set
import re
from collections import defaultdict, Counter

class ExtractionPatternAnalyzer:
    """Анализатор паттернов извлечения"""
    
    def __init__(self):
        self.validation_dir = Path("validation_dataset")
        self.glm_results_dir = Path("vision_results")
        
        # Паттерны для поиска значений
        self.value_patterns = {
            'size': [
                r'(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)??\s*nm',
                r'size[:\s]+(\d+(?:\.\d+)?)',
                r'diameter[:\s]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*nanometer'
            ],
            'zeta': [
                r'([+-]?\d+(?:\.\d+)?)\s*mV',
                r'zeta\s*potential[:\s]+([+-]?\d+(?:\.\d+)?)',
                r'ζ[:\s]+([+-]?\d+(?:\.\d+)?)'
            ],
            'ic50': [
                r'IC50[:\s]+(\d+(?:\.\d+)?)\s*(?:μg|ug)/ml',
                r'IC\s*50[:\s]+(\d+(?:\.\d+)?)',
                r'half.?maximal.?inhibitory[:\s]+(\d+(?:\.\d+)?)'
            ],
            'magnetic': [
                r'Ms[:\s]+(\d+(?:\.\d+)?)\s*emu/g',
                r'saturation\s*magnetization[:\s]+(\d+(?:\.\d+)?)',
                r'Mr[:\s]+(\d+(?:\.\d+)?)\s*emu/g',
                r'Hc[:\s]+(\d+(?:\.\d+)?)\s*Oe'
            ],
            'enzyme': [
                r'Km[:\s]+(\d+(?:\.\d+)?)\s*(?:mM|μM|uM)',
                r'Vmax[:\s]+(\d+(?:\.\d+)?)',
                r'Kcat[:\s]+(\d+(?:\.\d+)?)\s*s-1'
            ]
        }
    
    def load_current_extractions(self) -> Dict:
        """Загрузка текущих извлечений из валидационного набора"""
        analysis_file = self.validation_dir / 'current_extractions_analysis.json'
        
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                return json.load(f)
        return {}
    
    def analyze_extraction_text(self, text: str) -> Dict:
        """Анализ текста извлечения на наличие паттернов"""
        found_patterns = defaultdict(list)
        
        if not text or not isinstance(text, str):
            return found_patterns
        
        # Ищем различные паттерны
        for pattern_type, patterns in self.value_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    found_patterns[pattern_type].extend(matches)
        
        # Анализируем структуру текста
        text_features = {
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_units': bool(re.search(r'(nm|mV|mg|ml|μg|emu|Oe|mM|μM|s-1)', text, re.IGNORECASE)),
            'has_equations': bool(re.search(r'[=<>]', text)),
            'has_ranges': bool(re.search(r'\d+\s*[-–]\s*\d+', text)),
            'has_error_bars': bool(re.search(r'±', text)),
            'length': len(text),
            'word_count': len(text.split())
        }
        
        return {
            'patterns_found': dict(found_patterns),
            'text_features': text_features
        }
    
    def analyze_all_extractions(self) -> Dict:
        """Анализ всех извлечений"""
        current_extractions = self.load_current_extractions()
        
        analysis_results = {
            'total_files': 0,
            'total_extractions': 0,
            'extraction_types': Counter(),
            'common_patterns': defaultdict(Counter),
            'text_statistics': defaultdict(list),
            'dataset_specific': {}
        }
        
        for dataset, files in current_extractions.items():
            dataset_analysis = {
                'files': len(files),
                'total_analyses': 0,
                'total_tables': 0,
                'patterns': defaultdict(list),
                'sample_extractions': []
            }
            
            for file_info in files:
                analysis_results['total_files'] += 1
                
                # Анализируем analyses
                for analysis_text in file_info.get('sample_analysis', []):
                    if isinstance(analysis_text, str):
                        pattern_analysis = self.analyze_extraction_text(analysis_text)
                        
                        # Собираем статистику
                        for pattern_type, values in pattern_analysis['patterns_found'].items():
                            analysis_results['common_patterns'][pattern_type].update(values)
                            dataset_analysis['patterns'][pattern_type].extend(values)
                        
                        # Сохраняем пример
                        if len(dataset_analysis['sample_extractions']) < 3:
                            dataset_analysis['sample_extractions'].append({
                                'text': analysis_text[:200] + '...' if len(analysis_text) > 200 else analysis_text,
                                'patterns': pattern_analysis['patterns_found'],
                                'features': pattern_analysis['text_features']
                            })
                        
                        dataset_analysis['total_analyses'] += 1
                        analysis_results['total_extractions'] += 1
                
                # Анализируем tables
                for table_text in file_info.get('sample_table', []):
                    if isinstance(table_text, str):
                        dataset_analysis['total_tables'] += 1
                        analysis_results['extraction_types']['table'] += 1
            
            analysis_results['dataset_specific'][dataset] = dataset_analysis
        
        return analysis_results
    
    def identify_common_issues(self, analysis: Dict) -> List[str]:
        """Идентификация общих проблем"""
        issues = []
        
        # Проверяем, есть ли извлечения вообще
        if analysis['total_extractions'] == 0:
            issues.append("Нет успешных извлечений текста")
        
        # Проверяем наличие числовых данных
        has_numeric = any(analysis['common_patterns'].values())
        if not has_numeric:
            issues.append("Не найдены числовые значения в извлечениях")
        
        # Проверяем по датасетам
        for dataset, dataset_info in analysis['dataset_specific'].items():
            if dataset_info['total_analyses'] == 0:
                issues.append(f"Датасет {dataset}: нет анализов графиков")
            
            if not dataset_info['patterns']:
                issues.append(f"Датасет {dataset}: не найдены целевые паттерны")
        
        return issues
    
    def generate_prompt_improvements(self, analysis: Dict, issues: List[str]) -> Dict:
        """Генерация улучшений для промптов"""
        improvements = {
            'general': [],
            'dataset_specific': {},
            'examples': []
        }
        
        # Общие улучшения
        if "Не найдены числовые значения" in ' '.join(issues):
            improvements['general'].append({
                'issue': 'Модель не извлекает числовые значения',
                'suggestion': 'Добавить явное указание на извлечение чисел',
                'example_prompt': 'Extract ALL numerical values with their units. Focus on exact numbers from axes, labels, and data points.'
            })
        
        # Специфичные для датасетов
        for dataset in ['synergy', 'cytox', 'magnetic', 'nanozymes', 'seltox']:
            dataset_improvements = []
            
            if dataset == 'synergy':
                dataset_improvements.append({
                    'parameter': 'NP_size_avg_nm',
                    'prompt': 'Look for particle size or diameter values in nanometers (nm). Check graph axes, legends, and labels.',
                    'extraction_pattern': r'size.*?(\d+\.?\d*)\s*nm'
                })
                dataset_improvements.append({
                    'parameter': 'zeta_potential_mV',
                    'prompt': 'Find zeta potential values in millivolts (mV). Can be positive or negative.',
                    'extraction_pattern': r'zeta.*?([+-]?\d+\.?\d*)\s*mV'
                })
            
            elif dataset == 'cytox':
                dataset_improvements.append({
                    'parameter': 'IC50_ug_per_ml',
                    'prompt': 'Extract IC50 values, usually in μg/ml or similar units. Look for dose-response curves.',
                    'extraction_pattern': r'IC50.*?(\d+\.?\d*)\s*[μu]g/ml'
                })
            
            elif dataset == 'magnetic':
                dataset_improvements.append({
                    'parameter': 'Ms_emu_per_g',
                    'prompt': 'Find saturation magnetization (Ms) in emu/g from hysteresis loops.',
                    'extraction_pattern': r'Ms.*?(\d+\.?\d*)\s*emu/g'
                })
            
            improvements['dataset_specific'][dataset] = dataset_improvements
        
        # Добавляем примеры успешных извлечений
        for dataset, info in analysis['dataset_specific'].items():
            for sample in info.get('sample_extractions', [])[:1]:
                if sample['patterns']:
                    improvements['examples'].append({
                        'dataset': dataset,
                        'successful_patterns': sample['patterns'],
                        'text_snippet': sample['text'][:100]
                    })
        
        return improvements
    
    def save_analysis_report(self, analysis: Dict, issues: List[str], improvements: Dict):
        """Сохранение отчета анализа"""
        report = {
            'analysis': analysis,
            'identified_issues': issues,
            'prompt_improvements': improvements,
            'recommendations': [
                "Использовать более конкретные инструкции для извлечения чисел",
                "Добавить примеры ожидаемого формата вывода",
                "Указать конкретные единицы измерения для каждого параметра",
                "Использовать few-shot примеры успешных извлечений"
            ]
        }
        
        with open(self.validation_dir / 'extraction_pattern_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    analyzer = ExtractionPatternAnalyzer()
    
    print("="*80)
    print("АНАЛИЗ ПАТТЕРНОВ ИЗВЛЕЧЕНИЯ")
    print("="*80)
    print()
    
    # Анализируем все извлечения
    print("1. Анализ текущих извлечений...")
    analysis = analyzer.analyze_all_extractions()
    print(f"   ✅ Проанализировано {analysis['total_files']} файлов")
    print(f"   ✅ Найдено {analysis['total_extractions']} извлечений")
    
    # Идентифицируем проблемы
    print("\n2. Идентификация проблем...")
    issues = analyzer.identify_common_issues(analysis)
    if issues:
        print("   Обнаруженные проблемы:")
        for issue in issues:
            print(f"   ⚠️  {issue}")
    else:
        print("   ✅ Критических проблем не обнаружено")
    
    # Генерируем улучшения
    print("\n3. Генерация улучшений для промптов...")
    improvements = analyzer.generate_prompt_improvements(analysis, issues)
    print(f"   ✅ Сгенерировано {len(improvements['general'])} общих улучшений")
    print(f"   ✅ Сгенерированы улучшения для {len(improvements['dataset_specific'])} датасетов")
    
    # Сохраняем отчет
    print("\n4. Сохранение отчета...")
    report = analyzer.save_analysis_report(analysis, issues, improvements)
    print("   ✅ Отчет сохранен в extraction_pattern_analysis.json")
    
    # Выводим рекомендации
    print("\n" + "="*80)
    print("РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
    print("="*80)
    
    print("\n📋 Общие улучшения:")
    for improvement in improvements['general']:
        print(f"\n   Проблема: {improvement['issue']}")
        print(f"   Решение: {improvement['suggestion']}")
        if 'example_prompt' in improvement:
            print(f"   Пример: {improvement['example_prompt']}")
    
    print("\n📊 Улучшения по датасетам:")
    for dataset, dataset_improvements in improvements['dataset_specific'].items():
        if dataset_improvements:
            print(f"\n   {dataset.upper()}:")
            for imp in dataset_improvements[:2]:  # Показываем первые 2
                print(f"   - {imp['parameter']}: {imp['prompt'][:100]}...")
    
    print("\n✨ Следующие шаги:")
    print("1. Реализовать улучшенные промпты в iterative_prompt_improvement.py")
    print("2. Протестировать на валидационном наборе")
    print("3. Измерить улучшение метрик")

if __name__ == "__main__":
    main()