#!/usr/bin/env python3
"""
Создание небольшого валидационного датасета с ручной разметкой
для итеративного улучшения промптов
"""

import json
from pathlib import Path
import random
from typing import Dict, List, Tuple
import shutil
import csv

class ValidationDatasetCreator:
    """Создатель валидационного датасета"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.glm_results_dir = self.base_dir / "vision_results"
        self.extracted_pdfs_dir = self.base_dir / "extracted_pdfs"
        self.validation_dir = self.base_dir / "validation_dataset"
        
        # Создаем директорию для валидации
        self.validation_dir.mkdir(exist_ok=True)
        
        # Параметры для каждого типа датасета
        self.dataset_params = {
            'synergy': {
                'params': ['NP_size_avg_nm', 'zeta_potential_mV', 'CI'],
                'count': 4  # По 2 train + 2 test
            },
            'cytox': {
                'params': ['IC50_ug_per_ml', 'Cell_type', 'Cell_viability'],
                'count': 4
            },
            'magnetic': {
                'params': ['Ms_emu_per_g', 'Mr_emu_per_g', 'Hc_Oe'],
                'count': 4
            },
            'nanozymes': {
                'params': ['Km_value', 'Vmax_value', 'Kcat_value'],
                'count': 4
            },
            'seltox': {
                'params': ['IC50_ug_per_ml', 'Cell_type', 'SI'],
                'count': 4
            }
        }
        
    def select_representative_files(self) -> Dict[str, List[Tuple[str, str]]]:
        """Выбор репрезентативных файлов для валидации"""
        selected_files = {}
        
        for dataset in self.dataset_params.keys():
            dataset_files = []
            
            # Выбираем файлы из train и test
            for split in ['train', 'test']:
                result_file = self.glm_results_dir / dataset / split / f"{dataset}_{split}_results.json"
                
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    # Выбираем файлы с наибольшим количеством извлеченных данных
                    files_with_data = []
                    for result in data.get('results', []):
                        if result.get('analyses') or result.get('tables'):
                            score = len(result.get('analyses', [])) + len(result.get('tables', []))
                            files_with_data.append((result['pdf_name'], score, split))
                    
                    # Сортируем по количеству извлечений и выбираем топ-2
                    files_with_data.sort(key=lambda x: x[1], reverse=True)
                    for pdf_name, score, split_name in files_with_data[:2]:
                        dataset_files.append((pdf_name, split_name))
            
            selected_files[dataset] = dataset_files[:self.dataset_params[dataset]['count']]
        
        return selected_files
    
    def create_validation_structure(self, selected_files: Dict[str, List[Tuple[str, str]]]):
        """Создание структуры валидационного датасета"""
        
        validation_info = {
            'total_files': 0,
            'datasets': {}
        }
        
        for dataset, files in selected_files.items():
            dataset_dir = self.validation_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            
            dataset_info = {
                'files': [],
                'params_to_extract': self.dataset_params[dataset]['params']
            }
            
            for pdf_name, split in files:
                # Копируем PDF файл
                source_pdf = self.extracted_pdfs_dir / dataset / split / pdf_name
                if source_pdf.exists():
                    dest_pdf = dataset_dir / pdf_name
                    shutil.copy2(source_pdf, dest_pdf)
                    
                    # Добавляем информацию о файле
                    file_info = {
                        'pdf_name': pdf_name,
                        'split': split,
                        'path': str(dest_pdf.relative_to(self.base_dir)),
                        'ground_truth': {}  # Здесь будет ручная разметка
                    }
                    
                    # Инициализируем ground truth пустыми значениями
                    for param in self.dataset_params[dataset]['params']:
                        file_info['ground_truth'][param] = None
                    
                    dataset_info['files'].append(file_info)
                    validation_info['total_files'] += 1
            
            validation_info['datasets'][dataset] = dataset_info
        
        # Сохраняем структуру валидационного датасета
        with open(self.validation_dir / 'validation_dataset.json', 'w') as f:
            json.dump(validation_info, f, indent=2)
        
        return validation_info
    
    def create_annotation_template(self, validation_info: Dict):
        """Создание шаблона для ручной аннотации"""
        
        annotation_template = []
        
        for dataset, dataset_info in validation_info['datasets'].items():
            for file_info in dataset_info['files']:
                annotation_entry = {
                    'dataset': dataset,
                    'pdf_name': file_info['pdf_name'],
                    'path': file_info['path'],
                    'parameters': {}
                }
                
                # Добавляем параметры с инструкциями
                for param in dataset_info['params_to_extract']:
                    annotation_entry['parameters'][param] = {
                        'value': None,
                        'unit': self._get_unit_for_param(param),
                        'found_on_page': None,
                        'confidence': None,  # low, medium, high
                        'notes': ""
                    }
                
                annotation_template.append(annotation_entry)
        
        # Сохраняем шаблон для аннотации
        with open(self.validation_dir / 'annotation_template.json', 'w') as f:
            json.dump(annotation_template, f, indent=2)
        
        # Создаем CSV шаблон для удобства
        self._create_csv_template(annotation_template)
        
        return annotation_template
    
    def _get_unit_for_param(self, param: str) -> str:
        """Получение единицы измерения для параметра"""
        units = {
            'NP_size_avg_nm': 'nm',
            'zeta_potential_mV': 'mV',
            'CI': 'dimensionless',
            'IC50_ug_per_ml': 'μg/ml',
            'Cell_viability': '%',
            'Ms_emu_per_g': 'emu/g',
            'Mr_emu_per_g': 'emu/g',
            'Hc_Oe': 'Oe',
            'Km_value': 'mM or μM',
            'Vmax_value': 'U/mg or similar',
            'Kcat_value': 's^-1',
            'SI': 'dimensionless',
            'Cell_type': 'text'
        }
        return units.get(param, 'unknown')
    
    def _create_csv_template(self, annotation_template: List[Dict]):
        """Создание CSV шаблона для аннотации"""
        
        rows = []
        for entry in annotation_template:
            for param, param_info in entry['parameters'].items():
                row = {
                    'dataset': entry['dataset'],
                    'pdf_name': entry['pdf_name'],
                    'parameter': param,
                    'value': '',
                    'unit': param_info['unit'],
                    'page': '',
                    'confidence': '',
                    'notes': ''
                }
                rows.append(row)
        
        # Записываем CSV файл без pandas
        fieldnames = ['dataset', 'pdf_name', 'parameter', 'value', 'unit', 'page', 'confidence', 'notes']
        with open(self.validation_dir / 'annotation_template.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def analyze_current_extractions(self, selected_files: Dict[str, List[Tuple[str, str]]]):
        """Анализ текущих извлечений для выбранных файлов"""
        
        analysis_results = {}
        
        for dataset, files in selected_files.items():
            dataset_analysis = []
            
            for pdf_name, split in files:
                # Загружаем результаты GLM
                glm_result_file = self.glm_results_dir / dataset / split / f"{dataset}_{split}_results.json"
                
                if glm_result_file.exists():
                    with open(glm_result_file, 'r') as f:
                        data = json.load(f)
                    
                    # Ищем результаты для конкретного файла
                    for result in data.get('results', []):
                        if result.get('pdf_name') == pdf_name:
                            file_analysis = {
                                'pdf_name': pdf_name,
                                'analyses_count': len(result.get('analyses', [])),
                                'tables_count': len(result.get('tables', [])),
                                'sample_analysis': result.get('analyses', [])[:2] if result.get('analyses') else [],
                                'sample_table': result.get('tables', [])[:1] if result.get('tables') else []
                            }
                            dataset_analysis.append(file_analysis)
                            break
            
            analysis_results[dataset] = dataset_analysis
        
        # Сохраняем анализ
        with open(self.validation_dir / 'current_extractions_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        return analysis_results

def main():
    creator = ValidationDatasetCreator()
    
    print("="*80)
    print("СОЗДАНИЕ ВАЛИДАЦИОННОГО ДАТАСЕТА")
    print("="*80)
    print()
    
    # Выбираем репрезентативные файлы
    print("1. Выбор репрезентативных файлов...")
    selected_files = creator.select_representative_files()
    
    total_files = sum(len(files) for files in selected_files.values())
    print(f"   ✅ Выбрано {total_files} файлов для валидации")
    
    for dataset, files in selected_files.items():
        print(f"   - {dataset}: {len(files)} файлов")
    
    # Создаем структуру валидационного датасета
    print("\n2. Создание структуры валидационного датасета...")
    validation_info = creator.create_validation_structure(selected_files)
    print(f"   ✅ Создана структура в {creator.validation_dir}")
    
    # Создаем шаблон для аннотации
    print("\n3. Создание шаблона для ручной аннотации...")
    annotation_template = creator.create_annotation_template(validation_info)
    print(f"   ✅ Создан шаблон annotation_template.json")
    print(f"   ✅ Создан CSV шаблон annotation_template.csv")
    
    # Анализируем текущие извлечения
    print("\n4. Анализ текущих извлечений для выбранных файлов...")
    analysis = creator.analyze_current_extractions(selected_files)
    print(f"   ✅ Анализ сохранен в current_extractions_analysis.json")
    
    # Выводим инструкции
    print("\n" + "="*80)
    print("ИНСТРУКЦИИ ДЛЯ АННОТАЦИИ")
    print("="*80)
    print()
    print("1. Откройте PDF файлы в validation_dataset/[dataset]/")
    print("2. Заполните annotation_template.csv следующими данными:")
    print("   - value: точное числовое значение из PDF")
    print("   - page: номер страницы где найдено")
    print("   - confidence: low/medium/high")
    print("   - notes: любые заметки")
    print()
    print("3. После заполнения запустите скрипт итеративного улучшения")
    print()
    print(f"📁 Валидационный датасет создан в: {creator.validation_dir}")

if __name__ == "__main__":
    main()