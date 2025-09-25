"""
Benchmark different vision models - FIXED to match run_best_qwen_pipeline.py exactly
"""

import json
import pandas as pd
from pathlib import Path
import base64
import fitz
from PIL import Image
import io
from openai import OpenAI
import time
from typing import Dict, List, Optional
from datetime import datetime
import os

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Vision models to test (start with free versions)
VISION_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "z-ai/glm-4.5v",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
]

# Fallback to paid versions if rate limited
PAID_FALLBACKS = {
    "qwen/qwen2.5-vl-72b-instruct:free": "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-32b-instruct:free": "qwen/qwen2.5-vl-32b-instruct",
    "google/gemma-3-27b-it:free": "google/gemma-3-27b-it",
    "meta-llama/llama-4-maverick:free": "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout:free": "meta-llama/llama-4-scout",
}

# Import prompts
from vision_agent_prompts import DATASET_PROMPTS

class OpenRouterBenchmarkFixed:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.results = {}
        
    def pdf_to_images(self, pdf_path: Path, max_pages: Optional[int] = None) -> List[Image.Image]:
        """Convert PDF to images - EXACTLY like run_best_qwen_pipeline.py"""
        images = []
        try:
            pdf_document = fitz.open(str(pdf_path))
            num_pages = len(pdf_document)
            
            # Process ALL pages or up to limit
            pages_to_process = min(num_pages, max_pages) if max_pages else num_pages
            
            for page_num in range(pages_to_process):
                page = pdf_document[page_num]
                # MATCH EXACT RESOLUTION: 2.5x
                mat = fitz.Matrix(2.5, 2.5)  
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                images.append(img)
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error converting PDF: {e}")
        
        return images
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert image to base64 - EXACTLY like run_best_qwen_pipeline.py"""
        buffered = io.BytesIO()
        
        # Optimize size but keep quality
        max_size = 2048
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save with high quality
        image.save(buffered, format="PNG", optimize=True, quality=95)
        return base64.b64encode(buffered.getvalue()).decode()
    
    def process_single_page(self, image: Image.Image, page_num: int, model_name: str, prompt: str) -> Optional[Dict]:
        """Process single page through model"""
        try:
            base64_image = self.image_to_base64(image)
            
            # Add page context like run_best_qwen_pipeline.py
            enhanced_prompt = f"""Page {page_num} of a scientific paper.
            
CRITICAL: Extract ALL numerical values you can find for the specified parameters.
Look at EVERYTHING: tables, graphs, text, captions, legends, axes.

{prompt}"""
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "llm-nanomaterials-extraction",
                    "X-Title": "Scientific Data Extraction",
                },
                model=model_name,
                messages=[
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
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = completion.choices[0].message.content
            
            # Parse JSON response
            if '{' in content and '}' in content:
                json_str = content[content.index('{'):content.rindex('}')+1]
                result = json.loads(json_str)
                return result
                
        except Exception as e:
            if ":free" in model_name and "rate" in str(e).lower():
                # Try paid fallback
                paid_model = PAID_FALLBACKS.get(model_name)
                if paid_model:
                    print(f"  Switching to paid: {paid_model}")
                    return self.process_single_page(image, page_num, paid_model, prompt)
            return None
        
        return None
    
    def extract_with_model(self, model_name: str, pdf_path: Path, dataset_type: str) -> Dict:
        """Extract parameters - process ALL pages and aggregate like run_best_qwen_pipeline.py"""
        
        prompt = DATASET_PROMPTS.get(dataset_type, DATASET_PROMPTS['cytox'])
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        
        if not images:
            return {}
        
        # Process ALL pages and collect results
        all_results = []
        
        for page_num, image in enumerate(images, 1):
            result = self.process_single_page(image, page_num, model_name, prompt)
            if result and any(v is not None for v in result.values()):
                all_results.append(result)
        
        # Aggregate results - take first non-null value for each field
        final_result = {}
        
        if all_results:
            # Get all possible keys
            all_keys = set()
            for r in all_results:
                all_keys.update(r.keys())
            
            # For each key, find first non-null value
            for key in all_keys:
                for result in all_results:
                    value = result.get(key)
                    if value is not None and key not in final_result:
                        final_result[key] = value
        
        return final_result
    
    def load_gold_standard(self, dataset_type: str) -> pd.DataFrame:
        """Load gold standard CSV for a dataset type"""
        
        # Use TEST datasets as requested
        csv_mapping = {
            'cytox': 'cv sets marked - cyto_test.csv',
            'synergy': 'cv sets marked - syn_test.csv', 
            'magnetic': 'cv sets marked - mag_test.csv',
            'nanozymes': 'cv sets marked - nzyme_test.csv',
            'seltox': 'cv sets marked - seltox_test.csv'
        }
        
        csv_file = f"datasets/plots/cv_sets_marked/сsvs/{csv_mapping[dataset_type]}"
        return pd.read_csv(csv_file, encoding='utf-8')
    
    def calculate_metrics(self, extracted: Dict, gold_df: pd.DataFrame, filename: str) -> Dict:
        """Calculate precision, recall, F1 for extracted vs gold"""
        
        # Get gold values for this file
        base_name = Path(filename).stem
        gold_rows = gold_df[gold_df.iloc[:, 1].str.contains(base_name, na=False, case=False)]
        
        if gold_rows.empty:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        
        # Count matches
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Get expected parameters from gold
        gold_params = {}
        for _, row in gold_rows.iterrows():
            param_name = str(row.iloc[3]).lower()
            param_value = row.iloc[4]
            if not pd.isna(param_value):
                gold_params[param_name] = param_value
        
        # Check extracted values
        for key, value in extracted.items():
            if value is not None:
                # Check if this parameter exists in gold
                found_match = False
                for gold_key in gold_params:
                    if gold_key in key.lower() or key.lower() in gold_key:
                        found_match = True
                        true_positives += 1
                        break
                
                if not found_match:
                    false_positives += 1
        
        # Check for missing parameters
        for gold_key in gold_params:
            found_in_extracted = False
            for ext_key, ext_value in extracted.items():
                if ext_value is not None and (gold_key in ext_key.lower() or ext_key.lower() in gold_key):
                    found_in_extracted = True
                    break
            
            if not found_in_extracted:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives
        }
    
    def benchmark_dataset(self, dataset_type: str, num_files: int = None):
        """Benchmark all models on a specific dataset"""
        
        print(f"\n{'='*60}")
        print(f"BENCHMARKING DATASET: {dataset_type}")
        print(f"{'='*60}")
        
        # Load gold standard
        gold_df = self.load_gold_standard(dataset_type)
        
        # Get unique PDF files
        pdf_files = gold_df.iloc[:, 1].unique()
        
        if num_files:
            pdf_files = pdf_files[:num_files]
        
        print(f"Processing {len(pdf_files)} files from {dataset_type}")
        
        # Results for this dataset
        dataset_results = {}
        
        for model in VISION_MODELS:
            print(f"\nTesting model: {model}")
            model_metrics = {
                'total_precision': 0,
                'total_recall': 0,
                'total_f1': 0,
                'extraction_rate': 0,
                'files_processed': 0,
                'files_extracted': 0
            }
            
            for i, pdf_file in enumerate(pdf_files, 1):
                # Find actual PDF path
                filename = Path(pdf_file).name
                if not filename.endswith('.pdf'):
                    filename = filename + '.pdf'
                
                pdf_path = None
                # Look in test folders first
                for pattern in [
                    f"extracted_pdfs/{dataset_type}/test/{dataset_type}_test_pdf/{filename}",
                    f"extracted_pdfs/{dataset_type}/train/{dataset_type}_train_pdf/{filename}"
                ]:
                    if Path(pattern).exists():
                        pdf_path = Path(pattern)
                        break
                
                if not pdf_path:
                    continue
                
                model_metrics['files_processed'] += 1
                
                # Extract with model - processes ALL pages
                try:
                    print(f"  [{i}/{len(pdf_files)}] {filename[:30]}...", end="", flush=True)
                    extracted = self.extract_with_model(model, pdf_path, dataset_type)
                    
                    if extracted:
                        model_metrics['files_extracted'] += 1
                        
                        # Calculate metrics
                        metrics = self.calculate_metrics(extracted, gold_df, filename)
                        model_metrics['total_precision'] += metrics['precision']
                        model_metrics['total_recall'] += metrics['recall']
                        model_metrics['total_f1'] += metrics['f1']
                        
                        print(f" ✓ (P:{metrics['precision']:.2f} R:{metrics['recall']:.2f})")
                    else:
                        print(" ✗")
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f" Error: {str(e)[:50]}")
            
            # Calculate averages
            if model_metrics['files_processed'] > 0:
                n = model_metrics['files_extracted'] if model_metrics['files_extracted'] > 0 else 1
                model_metrics['avg_precision'] = model_metrics['total_precision'] / n
                model_metrics['avg_recall'] = model_metrics['total_recall'] / n
                model_metrics['avg_f1'] = model_metrics['total_f1'] / n
                model_metrics['extraction_rate'] = model_metrics['files_extracted'] / model_metrics['files_processed']
            
            dataset_results[model] = model_metrics
            
            print(f"\n  Model Summary:")
            print(f"    Extraction Rate: {model_metrics.get('extraction_rate', 0)*100:.1f}%")
            print(f"    Avg Precision: {model_metrics.get('avg_precision', 0):.3f}")
            print(f"    Avg Recall: {model_metrics.get('avg_recall', 0):.3f}")
            print(f"    Avg F1: {model_metrics.get('avg_f1', 0):.3f}")
        
        self.results[dataset_type] = dataset_results
        return dataset_results
    
    def run_full_benchmark(self, datasets=None):
        """Run benchmark on all datasets"""
        
        if datasets is None:
            datasets = ['cytox', 'synergy', 'magnetic', 'nanozymes', 'seltox']
        
        for dataset in datasets:
            try:
                self.benchmark_dataset(dataset)
            except Exception as e:
                print(f"Error benchmarking {dataset}: {e}")
        
        # Save results
        with open('openrouter_benchmark_results_fixed.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comparison report"""
        
        report = []
        report.append("# OpenRouter Vision Models Benchmark Report (FIXED)")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Configuration")
        report.append("- Processing: ALL pages from each PDF")
        report.append("- Resolution: 2.5x (matching run_best_qwen_pipeline.py)")
        report.append("- Image optimization: resize to 2048px max, LANCZOS resampling")
        report.append("- Prompts: Enhanced with page context and CRITICAL instructions")
        
        report.append("\n## Summary by Model\n")
        
        # Aggregate metrics across all datasets
        model_summary = {}
        
        for dataset, dataset_results in self.results.items():
            for model, metrics in dataset_results.items():
                if model not in model_summary:
                    model_summary[model] = {
                        'total_extraction_rate': 0,
                        'total_precision': 0,
                        'total_recall': 0,
                        'total_f1': 0,
                        'datasets_tested': 0
                    }
                
                model_summary[model]['total_extraction_rate'] += metrics.get('extraction_rate', 0)
                model_summary[model]['total_precision'] += metrics.get('avg_precision', 0)
                model_summary[model]['total_recall'] += metrics.get('avg_recall', 0)
                model_summary[model]['total_f1'] += metrics.get('avg_f1', 0)
                model_summary[model]['datasets_tested'] += 1
        
        # Sort models by average F1 score
        sorted_models = sorted(model_summary.items(), 
                              key=lambda x: x[1]['total_f1'] / max(x[1]['datasets_tested'], 1), 
                              reverse=True)
        
        report.append("| Model | Avg Extraction Rate | Avg Precision | Avg Recall | Avg F1 | Cost |")
        report.append("|-------|---------------------|---------------|------------|--------|------|")
        
        for model, summary in sorted_models:
            n = max(summary['datasets_tested'], 1)
            cost = "Free" if ":free" in model else "Paid"
            report.append(f"| {model.split('/')[1].split(':')[0]} | "
                         f"{summary['total_extraction_rate']/n*100:.1f}% | "
                         f"{summary['total_precision']/n:.3f} | "
                         f"{summary['total_recall']/n:.3f} | "
                         f"{summary['total_f1']/n:.3f} | "
                         f"{cost} |")
        
        report.append("\n## Detailed Results by Dataset\n")
        
        for dataset, dataset_results in self.results.items():
            report.append(f"\n### {dataset.upper()}\n")
            report.append("| Model | Extraction Rate | Precision | Recall | F1 |")
            report.append("|-------|-----------------|-----------|--------|-----|")
            
            sorted_dataset = sorted(dataset_results.items(), 
                                   key=lambda x: x[1].get('avg_f1', 0), 
                                   reverse=True)
            
            for model, metrics in sorted_dataset:
                report.append(f"| {model.split('/')[1].split(':')[0]} | "
                             f"{metrics.get('extraction_rate', 0)*100:.1f}% | "
                             f"{metrics.get('avg_precision', 0):.3f} | "
                             f"{metrics.get('avg_recall', 0):.3f} | "
                             f"{metrics.get('avg_f1', 0):.3f} |")
        
        report.append("\n## Recommendations\n")
        
        if sorted_models:
            best_model = sorted_models[0][0]
            best_free = next((m for m, _ in sorted_models if ":free" in m), None)
            
            report.append(f"1. **Best Overall Model**: {best_model}")
            if best_free:
                report.append(f"2. **Best Free Model**: {best_free}")
            report.append("3. **For Production**: Use free models first, fallback to paid when rate limited")
        
        # Save report
        report_text = "\n".join(report)
        with open('openrouter_benchmark_report_fixed.md
        ', 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)


if __name__ == "__main__":
    benchmark = OpenRouterBenchmarkFixed()
    # Run for remaining datasets with updated prompts
    print("\n" + "="*80)
    print("BENCHMARKING ALL MODELS ON CYTOX, SYNERGY, SELTOX WITH UPDATED PROMPTS")
    print("="*80)
    print("Features: Multiple experiments support, JSON arrays")
    print("="*80)

    # Test cytox, synergy, seltox with updated prompts
    for dataset in ["cytox", "synergy", "seltox"]:
        print(f"\n🔄 Starting benchmark for {dataset.upper()}...")
        benchmark.benchmark_dataset(dataset)

    # Generate final report
    benchmark.generate_report()

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = f"remaining_datasets_benchmark_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump(benchmark.results, f, indent=2)
    print(f"\n📊 Combined results saved to: {combined_file}")