#!/usr/bin/env python3
"""
Оптимизированный Pipeline с YOLO и полной обработкой страниц
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
import numpy as np

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

# Импорт YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("YOLO not available, will process full pages only")
    YOLO_AVAILABLE = False


class OptimizedQwenPipeline:
    """Оптимизированный Pipeline для обработки PDF с помощью Qwen2.5-VL"""
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.base_dir = Path(".")
        self.extracted_pdfs_dir = self.base_dir / "extracted_pdfs"
        self.results_dir = self.base_dir / "vision_results_optimized"
        self.results_dir.mkdir(exist_ok=True)
        
        # Загружаем YOLO если доступен
        if YOLO_AVAILABLE and Path("best.pt").exists():
            logger.info("Loading YOLO model from best.pt")
            self.yolo_model = YOLO("best.pt")
        else:
            self.yolo_model = None
            logger.warning("YOLO model not loaded")
        
        # Инициализация VLM клиента для Qwen
        self.client, self.config = vlm_manager.get_client(VLMProvider.QWEN_2_5_VL)
        
        # Улучшенные промпты с акцентом на ТОЧНЫЕ значения
        self.prompts = {
            'synergy': """You are an expert at extracting quantitative data from scientific figures in nanomedicine papers.

CRITICAL TASK: Extract EXACT numerical values for these parameters:

1. NP_size_avg_nm (Nanoparticle Size):
   - Look for: "particle size", "diameter", "hydrodynamic diameter", "DLS", "TEM size", "size distribution"
   - Common locations: characterization tables, DLS graphs, TEM images with scale bars
   - Units: nm (nanometers)
   - Typical range: 10-500 nm
   - IMPORTANT: Extract the MEAN/AVERAGE value if multiple values exist

2. zeta_potential_mV (Surface Charge):
   - Look for: "ζ-potential", "zeta potential", "surface charge", "ZP"
   - Common locations: characterization tables, bar graphs
   - Units: mV (millivolts)
   - Typical range: -50 to +50 mV
   - IMPORTANT: Extract the exact value with sign (+ or -)

3. CI (Combination Index):
   - Look for: "CI", "combination index", "synergy index", "CI value"
   - Common locations: isobologram plots, CI tables, synergy analysis sections
   - Units: dimensionless (no unit)
   - Typical range: 0.1-2.0 (CI<1 indicates synergy)
   - IMPORTANT: Extract the specific CI value, not ranges

EXTRACTION RULES:
- Extract NUMBERS ONLY, not text descriptions
- If you see a range (e.g., "20-30 nm"), extract the mean (25)
- If multiple values exist, prioritize the mean/average
- Look at axis values, data point labels, table cells
- Check figure captions - they often state exact values
- Return null ONLY if the parameter is completely absent

Return EXACTLY this JSON format:
{
  "NP_size_avg_nm": <number or null>,
  "zeta_potential_mV": <number or null>,
  "CI": <number or null>
}""",

            'cytox': """You are an expert at extracting cytotoxicity data from scientific figures.

CRITICAL TASK: Extract EXACT values from dose-response curves and viability assays:

1. IC50_ug_per_ml (Half-Maximal Inhibitory Concentration):
   - Look for: "IC50", "IC₅₀", "EC50", "LD50", "half-maximal"
   - Common locations: 
     * Dose-response curves - find the concentration at 50% viability
     * Tables summarizing IC50 values
     * Figure captions often state "IC50 = X μg/ml"
   - Units: μg/ml, mg/ml, μM, nM (convert to μg/ml if possible)
   - Typical range: 0.1-1000 μg/ml
   - IMPORTANT: Extract the exact IC50 value, not the range

2. Cell_type (Cell Line Name):
   - Look for: cell line names like "HeLa", "MCF-7", "A549", "PC3", "HepG2"
   - Common locations: figure legend, axis labels, figure title
   - IMPORTANT: Extract the exact cell line name (case-sensitive)

3. Cell_viability (Viability Percentage):
   - Look for: "% viability", "% survival", "cell survival", "relative viability"
   - Common locations: Y-axis of dose-response curves, bar graphs
   - Units: percentage (%)
   - Typical range: 0-100%
   - IMPORTANT: Extract the control (100%) or specific treatment value

EXTRACTION RULES:
- For dose-response curves: Find the exact IC50 from the curve or caption
- Extract exact cell line names as written
- For viability: Extract specific percentage values
- Return null ONLY if parameter is completely absent

Return EXACTLY this JSON format:
{
  "IC50_ug_per_ml": <number or null>,
  "Cell_type": "<string or null>",
  "Cell_viability": <number or null>
}"""
        }
        
        self.prompt = self.prompts.get(dataset_type, self.prompts['synergy'])
    
    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Конвертация PDF в изображения"""
        images = []
        try:
            pdf_document = fitz.open(str(pdf_path))
            # Обрабатываем ВСЕ страницы
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                mat = fitz.Matrix(2.0, 2.0)  # 2x увеличение для лучшего качества
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            pdf_document.close()
            logger.info(f"Extracted {len(images)} images from PDF")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        return images
    
    def detect_regions_with_yolo(self, image: Image.Image) -> List[Dict]:
        """Детекция регионов с помощью YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            # Конвертируем в numpy array
            img_array = np.array(image)
            
            # Запускаем детекцию
            results = self.yolo_model(img_array, conf=0.25)  # Снизили порог для большего захвата
            
            regions = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0])
                        
                        # Определяем тип региона
                        region_type = 'graph' if cls == 0 else 'table' if cls == 1 else 'figure'
                        
                        regions.append({
                            'type': region_type,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf
                        })
            
            return regions
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def crop_region(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        """Вырезание региона из изображения"""
        try:
            x1, y1, x2, y2 = bbox
            return image.crop((x1, y1, x2, y2))
        except Exception as e:
            logger.error(f"Error cropping region: {e}")
            return image
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Конвертация изображения в base64"""
        buffered = io.BytesIO()
        # Оптимизируем размер для API
        if image.width > 2048 or image.height > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        image.save(buffered, format="PNG", optimize=True)
        return base64.b64encode(buffered.getvalue()).decode()
    
    def extract_from_image(self, image: Image.Image, context: str = "") -> Optional[Dict]:
        """Извлечение данных из изображения через Qwen API"""
        try:
            base64_image = self.image_to_base64(image)
            
            # Формируем запрос
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{context}\n\n{self.prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Вызов API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.1
            )
            
            # Парсим ответ
            content = response.choices[0].message.content
            
            # Извлекаем JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                return {
                    'data': extracted_data,
                    'raw_response': content[:500]
                }
            else:
                return {
                    'data': None,
                    'raw_response': content[:500]
                }
                
        except Exception as e:
            logger.error(f"API extraction error: {e}")
            return None
    
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Обработка одного PDF файла"""
        result = {
            'pdf_name': pdf_path.name,
            'status': 'processing',
            'analyses': [],
            'tables': [],
            'error': None,
            'processing_time': 0,
            'pages_processed': 0,
            'regions_detected': 0
        }
        
        start_time = time.time()
        
        try:
            # Конвертируем PDF в изображения
            images = self.pdf_to_images(pdf_path)
            
            if not images:
                result['status'] = 'error'
                result['error'] = 'No images extracted from PDF'
                return result
            
            result['pages_processed'] = len(images)
            logger.info(f"Processing {len(images)} pages from {pdf_path.name}")
            
            # Обрабатываем каждую страницу
            for page_idx, image in enumerate(images):
                logger.info(f"Processing page {page_idx+1}/{len(images)} of {pdf_path.name}")
                
                # ВАЖНО: Сначала обрабатываем ПОЛНУЮ страницу
                full_page_result = self.extract_from_image(
                    image, 
                    context="FULL PAGE ANALYSIS: Extract ALL relevant parameters from this scientific figure/page."
                )
                
                if full_page_result and full_page_result['data']:
                    result['analyses'].append({
                        'page': page_idx + 1,
                        'region_type': 'full_page',
                        'data': full_page_result['data'],
                        'raw_response': full_page_result['raw_response']
                    })
                
                # Затем детектируем и обрабатываем отдельные регионы с YOLO
                if self.yolo_model:
                    regions = self.detect_regions_with_yolo(image)
                    
                    if regions:
                        logger.info(f"YOLO detected {len(regions)} regions on page {page_idx+1}")
                        result['regions_detected'] += len(regions)
                        
                        for region_idx, region in enumerate(regions):
                            try:
                                # Вырезаем регион
                                region_image = self.crop_region(image, region['bbox'])
                                
                                # Контекст в зависимости от типа региона
                                if region['type'] == 'table':
                                    context = "TABLE ANALYSIS: This is a data table. Extract ALL numerical values with parameter names."
                                elif region['type'] == 'graph':
                                    context = "GRAPH ANALYSIS: This is a graph/chart. Extract values from axes, data points, and legends."
                                else:
                                    context = "FIGURE ANALYSIS: Extract all relevant numerical parameters."
                                
                                region_result = self.extract_from_image(region_image, context)
                                
                                if region_result and region_result['data']:
                                    analysis = {
                                        'page': page_idx + 1,
                                        'region': region_idx + 1,
                                        'region_type': region['type'],
                                        'confidence': region['confidence'],
                                        'data': region_result['data'],
                                        'raw_response': region_result['raw_response']
                                    }
                                    
                                    if region['type'] == 'table':
                                        result['tables'].append(analysis)
                                    else:
                                        result['analyses'].append(analysis)
                                
                            except Exception as e:
                                logger.error(f"Error processing region {region_idx+1}: {e}")
                                continue
            
            result['status'] = 'completed'
            logger.info(f"Completed {pdf_path.name}: {len(result['analyses'])} analyses, {len(result['tables'])} tables")
            
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
        
        # Обрабатываем файлы последовательно для лучшего контроля
        results = []
        processed = 0
        failed = 0
        
        for pdf in pdf_files:
            try:
                result = self.process_single_pdf(pdf)
                results.append(result)
                processed += 1
                
                if result['status'] == 'error':
                    failed += 1
                    logger.error(f"Failed: {pdf.name} - {result['error']}")
                else:
                    logger.info(f"Processed: {pdf.name} ({processed}/{len(pdf_files)}) - "
                              f"{len(result['analyses'])} analyses, {len(result['tables'])} tables")
                
            except Exception as e:
                logger.error(f"Exception processing {pdf.name}: {e}")
                failed += 1
                results.append({
                    'pdf_name': pdf.name,
                    'status': 'error',
                    'error': str(e),
                    'analyses': [],
                    'tables': []
                })
        
        # Сохраняем результаты
        output_file = result_dir / f"{dataset}_{split}_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'dataset': dataset,
                'split': split,
                'total_files': len(pdf_files),
                'processed': processed,
                'failed': failed,
                'timestamp': datetime.now().isoformat(),
                'model': 'Qwen2.5-VL-72B-Instruct-AWQ',
                'pipeline_version': 'optimized_v2',
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary: {processed}/{len(pdf_files)} processed, {failed} failed")
        
        # Выводим статистику извлечений
        total_analyses = sum(len(r['analyses']) for r in results)
        total_tables = sum(len(r['tables']) for r in results)
        total_pages = sum(r.get('pages_processed', 0) for r in results)
        total_regions = sum(r.get('regions_detected', 0) for r in results)
        
        print(f"\n📊 Extraction Statistics for {dataset}/{split}:")
        print(f"  Files processed: {processed}")
        print(f"  Total pages: {total_pages}")
        print(f"  YOLO regions detected: {total_regions}")
        print(f"  Total analyses: {total_analyses}")
        print(f"  Total tables: {total_tables}")


def main():
    """Запуск оптимизированного pipeline"""
    
    datasets = [
        ('synergy', 'train'),
        ('synergy', 'test'),
        ('cytox', 'train'),
        ('cytox', 'test')
    ]
    
    print("="*80)
    print("OPTIMIZED QWEN2.5-VL PIPELINE V2")
    print("Features: YOLO detection + Full page processing")
    print("="*80)
    print()
    
    # Тестируем подключение
    print("Testing Qwen2.5-VL connection...")
    try:
        client, config = vlm_manager.get_client(VLMProvider.QWEN_2_5_VL)
        test_response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "Say 'Ready for extraction!'"}],
            max_tokens=50
        )
        print(f"✅ {test_response.choices[0].message.content}")
        print(f"Model: {config.model_name}")
        print()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Обрабатываем каждый датасет
    for dataset, split in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset}/{split}...")
        print('='*60)
        
        pipeline = OptimizedQwenPipeline(dataset)
        
        # Для тестирования обрабатываем только 3 файла
        pipeline.process_dataset(dataset, split, max_files=3)
        
        print(f"✅ Completed {dataset}/{split}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Results saved in: vision_results_optimized/")


if __name__ == "__main__":
    main()