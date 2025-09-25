#!/usr/bin/env python3
"""
Лучший Pipeline для Qwen2.5-VL - обработка всех страниц без YOLO
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
from PIL import Image
import fitz  # PyMuPDF

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорт модулей
try:
    from graph_processing.vlm_config import VLMProvider, vlm_manager
except ImportError:
    from vlm_config import VLMProvider, vlm_manager


class BestQwenPipeline:
    """Лучший Pipeline для обработки PDF с помощью Qwen2.5-VL"""
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.base_dir = Path(".")
        self.extracted_pdfs_dir = self.base_dir / "extracted_pdfs"
        self.results_dir = self.base_dir / "vision_results_best"
        self.results_dir.mkdir(exist_ok=True)
        
        # Инициализация VLM клиента для Qwen
        self.client, self.config = vlm_manager.get_client(VLMProvider.QWEN_2_5_VL)
        
        # УЛУЧШЕННЫЕ ПРОМПТЫ - более агрессивное извлечение
        self.prompts = {
            'synergy': """You are analyzing a scientific figure from a nanomedicine paper about drug synergy.

YOUR TASK: Extract numerical values for these parameters. Be aggressive in finding values - look everywhere!

PARAMETERS TO EXTRACT:
1. NP_size_avg_nm: Nanoparticle size in nanometers
   - Search for: size, diameter, particle size, hydrodynamic diameter, DLS, TEM, SEM
   - Look in: tables, graphs, text, figure captions, scale bars
   - Common values: 10-500 nm
   - If you see a range like "20-30 nm", extract 25

2. zeta_potential_mV: Surface charge in millivolts  
   - Search for: zeta potential, ζ-potential, surface charge, ZP
   - Look in: characterization tables, bar graphs
   - Common values: -50 to +50 mV
   - Include the sign (+ or -)

3. CI: Combination Index
   - Search for: CI, combination index, synergy index
   - Look in: tables, isobologram plots, text mentioning synergy
   - Common values: 0.1-2.0
   - CI < 1 means synergy

IMPORTANT:
- Extract ANY number you find related to these parameters
- Check axes, data points, labels, legends, captions
- Look at table headers and cells
- Read text in the image
- If unsure, extract the most likely value
- Return null ONLY if absolutely nothing found

Return JSON:
{
  "NP_size_avg_nm": <number or null>,
  "zeta_potential_mV": <number or null>,
  "CI": <number or null>
}""",

            'cytox': """You are analyzing a cytotoxicity figure from a nanomedicine paper.

YOUR TASK: Extract values for cytotoxicity parameters. Be aggressive in finding values!

PARAMETERS TO EXTRACT:
1. IC50_ug_per_ml: Half-maximal inhibitory concentration
   - Search for: IC50, IC₅₀, EC50, GI50, LD50
   - Look in: dose-response curves, tables, figure captions
   - Find the concentration at 50% cell viability
   - Common units: μg/ml, mg/ml, μM, nM
   - Common values: 0.1-1000 μg/ml

2. Cell_type: Cell line name
   - Search for: HeLa, MCF-7, A549, PC3, HepG2, Caco-2, etc.
   - Look in: figure legends, titles, axis labels
   - Extract the exact name

3. Cell_viability: Percentage of viable cells
   - Search for: % viability, % survival, cell survival
   - Look in: Y-axis of graphs, bar charts
   - Common values: 0-100%
   - Extract control or specific treatment value

IMPORTANT:
- Check dose-response curves carefully
- Look at all text in the image
- Extract from tables, graphs, and captions
- If multiple values, take the most relevant one
- Return null ONLY if absolutely nothing found

Return JSON:
{
  "IC50_ug_per_ml": <number or null>,
  "Cell_type": "<string or null>",
  "Cell_viability": <number or null>
}"""
        }
        
        self.prompt = self.prompts.get(dataset_type, self.prompts['synergy'])
    
    def pdf_to_images(self, pdf_path: Path, max_pages: Optional[int] = None) -> List[Image.Image]:
        """Конвертация PDF в изображения - ВСЕ страницы"""
        images = []
        try:
            pdf_document = fitz.open(str(pdf_path))
            num_pages = len(pdf_document)
            
            # Обрабатываем ВСЕ страницы или до лимита
            pages_to_process = min(num_pages, max_pages) if max_pages else num_pages
            
            for page_num in range(pages_to_process):
                page = pdf_document[page_num]
                # Увеличиваем разрешение для лучшего распознавания
                mat = fitz.Matrix(2.5, 2.5)  # Еще выше качество
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Конвертируем в RGB если нужно
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                images.append(img)
            
            pdf_document.close()
            logger.info(f"Extracted {len(images)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        
        return images
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Конвертация изображения в base64"""
        buffered = io.BytesIO()
        
        # Оптимизируем размер но сохраняем качество
        max_size = 2048
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Сохраняем с хорошим качеством
        image.save(buffered, format="PNG", optimize=True, quality=95)
        return base64.b64encode(buffered.getvalue()).decode()
    
    def process_single_page(self, image: Image.Image, page_num: int) -> Optional[Dict]:
        """Обработка одной страницы через Qwen API"""
        try:
            base64_image = self.image_to_base64(image)
            
            # Добавляем контекст о странице
            enhanced_prompt = f"""Page {page_num} of a scientific paper.
            
CRITICAL: Extract ALL numerical values you can find for the specified parameters.
Look at EVERYTHING: tables, graphs, text, captions, legends, axes.

{self.prompt}"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Вызов API с повторными попытками
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.1  # Низкая температура для точности
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(2)
            
            # Парсим ответ
            content = response.choices[0].message.content
            
            # Извлекаем JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                return {
                    'page': page_num,
                    'data': extracted_data,
                    'raw_response': content[:500]
                }
            else:
                # Если нет JSON, сохраняем сырой ответ
                return {
                    'page': page_num,
                    'data': None,
                    'raw_response': content[:500],
                    'parse_error': True
                }
                
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return {
                'page': page_num,
                'data': None,
                'error': str(e)
            }
    
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Обработка одного PDF файла"""
        result = {
            'pdf_name': pdf_path.name,
            'status': 'processing',
            'analyses': [],
            'error': None,
            'processing_time': 0,
            'pages_processed': 0,
            'successful_extractions': 0
        }
        
        start_time = time.time()
        
        try:
            # Конвертируем PDF в изображения - ВСЕ страницы
            images = self.pdf_to_images(pdf_path)
            
            if not images:
                result['status'] = 'error'
                result['error'] = 'No images extracted from PDF'
                return result
            
            result['pages_processed'] = len(images)
            
            # Обрабатываем КАЖДУЮ страницу
            for page_idx, image in enumerate(images):
                page_num = page_idx + 1
                logger.info(f"Processing page {page_num}/{len(images)} of {pdf_path.name}")
                
                # Извлекаем данные со страницы
                page_result = self.process_single_page(image, page_num)
                
                if page_result:
                    result['analyses'].append(page_result)
                    
                    # Считаем успешные извлечения
                    if page_result.get('data'):
                        has_values = False
                        for key, value in page_result['data'].items():
                            if value is not None:
                                has_values = True
                                result['successful_extractions'] += 1
                                break
                        
                        if has_values:
                            logger.info(f"  ✓ Found data on page {page_num}: {page_result['data']}")
                
                # Небольшая пауза между страницами
                if page_idx < len(images) - 1:
                    time.sleep(0.5)
            
            result['status'] = 'completed'
            
            # Собираем все извлеченные данные
            all_extracted = {}
            for analysis in result['analyses']:
                if analysis.get('data'):
                    for key, value in analysis['data'].items():
                        if value is not None and key not in all_extracted:
                            all_extracted[key] = value
            
            result['summary'] = all_extracted
            
            logger.info(f"Completed {pdf_path.name}: {result['successful_extractions']} successful extractions")
            if all_extracted:
                logger.info(f"  Summary: {all_extracted}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def process_dataset(self, dataset: str, split: str, max_files: Optional[int] = None):
        """Обработка датасета"""
        dataset_dir = self.extracted_pdfs_dir / dataset / split / f"{dataset}_{split}_pdf"
        
        if not dataset_dir.exists():
            logger.error(f"Directory not found: {dataset_dir}")
            return
        
        # Создаем директорию для результатов
        result_dir = self.results_dir / dataset / split
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем список PDF файлов
        pdf_files = sorted(dataset_dir.glob("*.pdf"))
        
        if max_files:
            pdf_files = pdf_files[:max_files]
            logger.info(f"Limited to {max_files} files for testing")
        
        logger.info(f"Found {len(pdf_files)} PDF files in {dataset}/{split}")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {dataset_dir}")
            return
        
        # Обрабатываем файлы
        results = []
        processed = 0
        failed = 0
        
        # Используем параллельную обработку для ускорения
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_pdf = {
                executor.submit(self.process_single_pdf, pdf): pdf 
                for pdf in pdf_files
            }
            
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    result = future.result(timeout=300)  # 5 минут на файл
                    results.append(result)
                    processed += 1
                    
                    if result['status'] == 'error':
                        failed += 1
                        logger.error(f"Failed: {pdf.name}")
                    else:
                        logger.info(f"✓ Processed: {pdf.name} ({processed}/{len(pdf_files)})")
                    
                except Exception as e:
                    logger.error(f"Exception processing {pdf.name}: {e}")
                    failed += 1
                    results.append({
                        'pdf_name': pdf.name,
                        'status': 'error',
                        'error': str(e),
                        'analyses': []
                    })
        
        # Подсчитываем статистику
        total_pages = sum(r.get('pages_processed', 0) for r in results)
        total_extractions = sum(r.get('successful_extractions', 0) for r in results)
        
        # Считаем общие метрики
        extracted_params = {'NP_size_avg_nm': 0, 'zeta_potential_mV': 0, 'CI': 0, 'IC50_ug_per_ml': 0}
        null_params = {'NP_size_avg_nm': 0, 'zeta_potential_mV': 0, 'CI': 0, 'IC50_ug_per_ml': 0}
        
        for result in results:
            for analysis in result.get('analyses', []):
                if analysis.get('data'):
                    for param in extracted_params.keys():
                        if param in analysis['data']:
                            if analysis['data'][param] is not None:
                                extracted_params[param] += 1
                            else:
                                null_params[param] += 1
        
        # Сохраняем результаты
        output_file = result_dir / f"{dataset}_{split}_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'dataset': dataset,
                'split': split,
                'total_files': len(pdf_files),
                'processed': processed,
                'failed': failed,
                'total_pages': total_pages,
                'total_extractions': total_extractions,
                'extracted_params': extracted_params,
                'null_params': null_params,
                'timestamp': datetime.now().isoformat(),
                'model': 'Qwen2.5-VL-72B-Instruct-AWQ',
                'pipeline_version': 'best_no_yolo',
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Выводим подробную статистику
        print(f"\n{'='*60}")
        print(f"📊 Results for {dataset}/{split}:")
        print(f"{'='*60}")
        print(f"Files processed: {processed}/{len(pdf_files)}")
        print(f"Failed: {failed}")
        print(f"Total pages processed: {total_pages}")
        print(f"Successful extractions: {total_extractions}")
        print(f"\nExtracted parameters:")
        for param, count in extracted_params.items():
            if count > 0:
                print(f"  {param}: {count}")
        print(f"\nNull parameters:")
        for param, count in null_params.items():
            if count > 0:
                print(f"  {param}: {count}")
        
        # Считаем процент извлечения
        total_extracted = sum(extracted_params.values())
        total_nulls = sum(null_params.values())
        if total_extracted + total_nulls > 0:
            extraction_rate = (total_extracted / (total_extracted + total_nulls)) * 100
            print(f"\n🎯 Extraction rate: {extraction_rate:.1f}%")


def main():
    """Запуск лучшего pipeline"""
    
    datasets = [
        ('synergy', 'train'),
        ('synergy', 'test'),
        ('cytox', 'train'),
        ('cytox', 'test')
    ]
    
    print("="*80)
    print("BEST QWEN2.5-VL PIPELINE (NO YOLO)")
    print("Processing ALL pages with aggressive extraction")
    print("="*80)
    print()
    
    # Тестируем подключение
    print("Testing Qwen2.5-VL connection...")
    try:
        client, config = vlm_manager.get_client(VLMProvider.QWEN_2_5_VL)
        test_response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "Say 'Ready!'"}],
            max_tokens=20
        )
        print(f"✅ API Connected: {test_response.choices[0].message.content}")
        print(f"Model: {config.model_name}")
        print()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Обрабатываем каждый датасет
    for dataset, split in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()}/{split.upper()}")
        print('='*80)
        
        pipeline = BestQwenPipeline(dataset)
        
        # Обрабатываем по 5 файлов для теста
        pipeline.process_dataset(dataset, split, max_files=5)
        
        print(f"✅ Completed {dataset}/{split}")
        time.sleep(2)  # Пауза между датасетами
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE")
    print("="*80)
    print(f"Results saved in: vision_results_best/")
    print("\nRun compare_best_metrics.py to compare with original results")


if __name__ == "__main__":
    main()