#!/usr/bin/env python3
"""
Debug extraction results to understand structure
"""

import json
import pandas as pd
from pathlib import Path

def debug_extraction():
    """Debug extraction results and CSV structure"""
    
    # Check synergy train CSV
    csv_file = Path("datasets/plots/cv_sets_marked/сsvs/cv sets marked - syn_train.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print("=" * 70)
        print("SYNERGY TRAIN CSV STRUCTURE")
        print("=" * 70)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("\nParameter columns present:")
        for col in ['NP_size_avg_nm', 'zeta_potential_mV', 'combination_index', 'file_id', 'pdf_name']:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"  {col}: {non_null}/{len(df)} non-null values")
                if non_null > 0:
                    print(f"    Sample values: {df[col].dropna().head(3).tolist()}")
    
    # Check synergy train results
    results_file = Path("vision_results/synergy/train/synergy_train_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        print("\n" + "=" * 70)
        print("SYNERGY TRAIN RESULTS STRUCTURE")
        print("=" * 70)
        print(f"Keys: {list(data.keys())}")
        print(f"Number of results: {len(data.get('results', []))}")
        
        if 'results' in data and data['results']:
            # Check first result
            first_result = data['results'][0]
            print(f"\nFirst result keys: {list(first_result.keys())}")
            print(f"PDF name: {first_result.get('pdf_name')}")
            print(f"Status: {first_result.get('status')}")
            
            # Check graphs
            graphs = first_result.get('graphs', [])
            print(f"Number of graphs: {len(graphs)}")
            if graphs:
                print("\nFirst graph structure:")
                first_graph = graphs[0]
                if isinstance(first_graph, dict):
                    print(f"  Keys: {list(first_graph.keys())}")
                    for key in ['NP_size_avg_nm', 'zeta_potential_mV', 'combination_index']:
                        if key in first_graph:
                            print(f"  {key}: {first_graph[key]}")
                else:
                    print(f"  Type: {type(first_graph)}")
                    print(f"  Content: {str(first_graph)[:200]}")
            
            # Check tables
            tables = first_result.get('tables', [])
            print(f"\nNumber of tables: {len(tables)}")

if __name__ == "__main__":
    debug_extraction()