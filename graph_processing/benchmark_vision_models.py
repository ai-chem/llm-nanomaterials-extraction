"""
Benchmark different vision models from OpenRouter on gold standard datasets
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
from typing import Dict, List, Tuple
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

# Import prompts from our best Qwen prompts
from vision_agent_prompts import DATASET_PROMPTS

class OpenRouterBenchmark:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.results = {}
        
    def extract_with_model(self, model_name: str, pdf_path: Path, dataset_type: str, max_pages: int = None) -> Dict:
        """Extract parameters using a specific model - processes ALL pages by default"""
        
        prompt = DATASET_PROMPTS.get(dataset_type, DATASET_PROMPTS['cytox'])
        
        # Convert PDF pages to images
        pdf = fitz.open(str(pdf_path))
        
        # Process ALL pages unless limited
        pages_to_process = len(pdf) if max_pages is None else min(max_pages, len(pdf))
        
        for page_num in range(pages_to_process):
            page = pdf[page_num]
            mat = fitz.Matrix(2.0, 2.0)  # 2x resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to base64
            img = Image.open(io.BytesIO(img_data))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            try:
                # Call OpenRouter API
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
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
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
                    
                    # Check if we found values
                    if any(v is not None for v in result.values()):
                        pdf.close()
                        return result
                        
            except Exception as e:
                print(f"Error with {model_name} on page {page_num}: {e}")
                
                # Try fallback to paid version if rate limited
                if ":free" in model_name and "rate" in str(e).lower():
                    paid_model = PAID_FALLBACKS.get(model_name)
                    if paid_model:
                        print(f"Switching to paid version: {paid_model}")
                        return self.extract_with_model(paid_model, pdf_path, dataset_type, max_pages)
        
        pdf.close()
        return {}
    
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
    
    def benchmark_dataset(self, dataset_type: str, num_files: int = 5):
        """Benchmark all models on a specific dataset"""
        
        print(f"\n{'='*60}")
        print(f"BENCHMARKING DATASET: {dataset_type}")
        print(f"{'='*60}")
        
        # Load gold standard
        gold_df = self.load_gold_standard(dataset_type)
        
        # Get unique PDF files
        pdf_files = gold_df.iloc[:, 1].unique()[:num_files]
        
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
            
            for pdf_file in pdf_files:
                # Find actual PDF path
                filename = Path(pdf_file).name
                if not filename.endswith('.pdf'):
                    filename = filename + '.pdf'
                
                pdf_path = None
                # Look in test folders first since we're using test datasets
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
                
                # Extract with model
                try:
                    extracted = self.extract_with_model(model, pdf_path, dataset_type)
                    
                    if extracted:
                        model_metrics['files_extracted'] += 1
                        
                        # Calculate metrics
                        metrics = self.calculate_metrics(extracted, gold_df, filename)
                        model_metrics['total_precision'] += metrics['precision']
                        model_metrics['total_recall'] += metrics['recall']
                        model_metrics['total_f1'] += metrics['f1']
                        
                        print(f"  ✓ {filename}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
                    else:
                        print(f"  ✗ {filename}: No extraction")
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"  ✗ {filename}: Error - {e}")
            
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
    
    def run_full_benchmark(self):
        """Run benchmark on all datasets"""
        
        datasets = ['cytox', 'synergy', 'magnetic', 'nanozymes', 'seltox']
        
        for dataset in datasets:
            try:
                self.benchmark_dataset(dataset, num_files=3)  # Test 3 files per dataset
            except Exception as e:
                print(f"Error benchmarking {dataset}: {e}")
        
        # Save results
        with open('openrouter_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comparison report"""
        
        report = []
        report.append("# OpenRouter Vision Models Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        with open('openrouter_benchmark_report.md', 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)


if __name__ == "__main__":
    benchmark = OpenRouterBenchmark()
    benchmark.run_full_benchmark()
