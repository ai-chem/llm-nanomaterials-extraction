#!/usr/bin/env python3
"""
Vision Pipeline с использованием Qwen2.5-VL-72B
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


class QwenVisionPipeline:
    """Pipeline для обработки PDF с помощью Qwen2.5-VL"""
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.base_dir = Path(".")
        self.extracted_pdfs_dir = self.base_dir / "extracted_pdfs"
        self.results_dir = self.base_dir / "vision_results_qwen"
        self.results_dir.mkdir(exist_ok=True)
        
        # Инициализация VLM клиента для Qwen
        self.client, self.config = vlm_manager.get_client(VLMProvider.QWEN_2_5_VL)
        
        # Промпты для каждого датасета с улучшениями
        self.prompts = {
            'synergy': """You are analyzing a scientific figure from a nanomedicine paper about drug synergy.

EXTRACT these EXACT numerical values:
1. NP_size_avg_nm: Average nanoparticle size in nanometers (nm). Look for values labeled as "size", "diameter", "particle size", typically 10-500 nm range.
2. zeta_potential_mV: Surface charge in millivolts (mV). Look for "zeta potential", "ζ-potential", "surface charge", typically -50 to +50 mV.
3. CI: Combination Index value. Look for "CI", "combination index", "synergy index", typically 0.1-2.0.

Focus on:
- Axis labels and values
- Data point labels
- Figure legends and captions
- Table cells with numbers

Return a JSON with the extracted values and their units.""",

            'cytox': """You are analyzing a cytotoxicity assay figure from a nanomedicine paper.

EXTRACT these EXACT numerical values:
1. IC50_ug_per_ml: Half-maximal inhibitory concentration in μg/ml. Look for "IC50", "IC₅₀", dose-response curves, typically 0.1-1000 μg/ml.
2. Cell_type: Name of the cell line (e.g., "HeLa", "MCF-7", "A549").
3. Cell_viability: Percentage of viable cells, typically 0-100%.

Focus on:
- Dose-response curves and their IC50 values
- Y-axis showing cell viability percentage
- X-axis showing concentration
- Figure legends identifying cell lines

Return a JSON with the extracted values.""",

            'magnetic': """You are analyzing magnetic characterization data from a nanomaterial paper.

EXTRACT these EXACT numerical values:
1. Ms_emu_per_g: Saturation magnetization in emu/g. Look at hysteresis loops, plateau values, typically 10-90 emu/g.
2. Mr_emu_per_g: Remnant magnetization in emu/g. Value at zero field, typically 0-20 emu/g.
3. Hc_Oe: Coercivity in Oersted (Oe). X-intercept of hysteresis loop, typically 0-500 Oe.

Focus on:
- Hysteresis loop graphs (M-H curves)
- Values at saturation (top/bottom plateaus)
- Zero-field crossing points
- Axis labels with units

Return a JSON with the extracted magnetic parameters.""",

            'nanozymes': """You are analyzing enzyme kinetics data from a nanozyme paper.

EXTRACT these EXACT numerical values:
1. Km_value: Michaelis constant in mM or μM. Look for "Km", "KM", typically 0.01-100 mM.
2. Vmax_value: Maximum velocity in U/mg or similar units. Look for "Vmax", "Vₘₐₓ".
3. Kcat_value: Turnover number in s⁻¹. Look for "kcat", "turnover".

Focus on:
- Lineweaver-Burk plots
- Michaelis-Menten curves
- Kinetic parameter tables
- Enzyme activity graphs

Return a JSON with the kinetic parameters.""",

            'seltox': """You are analyzing selective toxicity data from a targeted therapy paper.

EXTRACT these EXACT numerical values:
1. IC50_ug_per_ml: Inhibitory concentration in μg/ml for different cell types.
2. Cell_type: Cell line names (both cancer and normal cells).
3. SI: Selectivity Index. Ratio of IC50(normal)/IC50(cancer), typically 1-100.

Focus on:
- Comparative toxicity bars/plots
- IC50 values for different cell lines
- Selectivity index calculations
- Normal vs cancer cell comparisons

Return a JSON with the selectivity data."""
        }
        
        self.prompt = self.prompts.get(dataset_type, self.prompts['synergy'])
    
    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Конвертация PDF в изображения"""
        images = []
        try:
            pdf_document = fitz.open(str(pdf_path))
            for page_num in range(min(10, len(pdf_document))):  # Максимум 10 страниц
                page = pdf_document[page_num]
                mat = fitz.Matrix(2.0, 2.0)  # 2x увеличение для лучшего качества
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        return images
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Конвертация изображения в base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Обработка одного PDF файла"""
        result = {
            'pdf_name': pdf_path.name,
            'status': 'processing',
            'analyses': [],
            'tables': [],
            'error': None,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Конвертируем PDF в изображения
            images = self.pdf_to_images(pdf_path)
            
            if not images:
                result['status'] = 'error'
                result['error'] = 'No images extracted from PDF'
                return result
            
            # Обрабатываем каждую страницу
            for idx, image in enumerate(images[:5]):  # Максимум 5 страниц для анализа
                try:
                    # Подготавливаем изображение для API
                    base64_image = self.image_to_base64(image)
                    
                    # Формируем запрос к Qwen2.5-VL
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
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
                    
                    # Пытаемся извлечь JSON из ответа
                    try:
                        # Ищем JSON в ответе
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            extracted_data = json.loads(json_match.group())
                            result['analyses'].append({
                                'page': idx + 1,
                                'data': extracted_data,
                                'raw_response': content
                            })
                        else:
                            result['analyses'].append({
                                'page': idx + 1,
                                'data': None,
                                'raw_response': content
                            })
                    except json.JSONDecodeError:
                        result['analyses'].append({
                            'page': idx + 1,
                            'data': None,
                            'raw_response': content
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing page {idx+1}: {e}")
                    continue
            
            result['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def process_dataset(self, dataset: str, split: str, max_workers: int = 4):
        """Обработка всего датасета"""
        # Правильная структура директорий
        dataset_dir = self.extracted_pdfs_dir / dataset / split / f"{dataset}_{split}_pdf"
        
        if not dataset_dir.exists():
            logger.error(f"Directory not found: {dataset_dir}")
            return
        
        # Создаем директорию для результатов
        result_dir = self.results_dir / dataset / split
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем список PDF файлов
        pdf_files = sorted(dataset_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {dataset}/{split}")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {dataset_dir}")
            return
        
        # Обрабатываем файлы параллельно
        results = []
        processed = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(self.process_single_pdf, pdf): pdf 
                for pdf in pdf_files
            }
            
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                    processed += 1
                    
                    if result['status'] == 'error':
                        failed += 1
                        logger.error(f"Failed: {pdf.name} - {result['error']}")
                    else:
                        logger.info(f"Processed: {pdf.name} ({processed}/{len(pdf_files)})")
                    
                except Exception as e:
                    logger.error(f"Exception processing {pdf.name}: {e}")
                    failed += 1
                    results.append({
                        'pdf_name': pdf.name,
                        'status': 'error',
                        'error': str(e)
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
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary: {processed}/{len(pdf_files)} processed, {failed} failed")


def main():
    """Запуск pipeline для всех датасетов"""
    
    datasets = [
        ('synergy', 'train'),
        ('synergy', 'test'),
        ('cytox', 'train'),
        ('cytox', 'test'),
        ('magnetic', 'train'),
        ('magnetic', 'test'),
        ('nanozymes', 'train'),
        ('nanozymes', 'test'),
        ('seltox', 'train'),
        ('seltox', 'test')
    ]
    
    print("="*80)
    print("QWEN2.5-VL-72B VISION PIPELINE")
    print("="*80)
    print()
    
    # Тестируем подключение
    print("Testing Qwen2.5-VL connection...")
    try:
        client, config = vlm_manager.get_client(VLMProvider.QWEN_2_5_VL)
        test_response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "Say 'Connection successful!'"}],
            max_tokens=50
        )
        print(f"✅ {test_response.choices[0].message.content}")
        print(f"Model: {config.model_name}")
        print(f"API URL: {config.api_url}")
        print()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Обрабатываем каждый датасет
    for dataset, split in datasets:
        print(f"\nProcessing {dataset}/{split}...")
        pipeline = QwenVisionPipeline(dataset)
        pipeline.process_dataset(dataset, split, max_workers=4)
        print(f"✅ Completed {dataset}/{split}")
        time.sleep(2)  # Небольшая пауза между датасетами
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Results saved in: vision_results_qwen/")


if __name__ == "__main__":
    main()