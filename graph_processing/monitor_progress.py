#!/usr/bin/env python3
"""
Monitor vision pipeline progress and show intermediate metrics
"""

import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from collections import defaultdict

def check_progress():
    """Check current processing progress"""
    base_dir = Path(".")
    
    # Check log file
    log_files = list(base_dir.glob("vision_pipeline_*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        # Find processing stages
        stages = []
        for line in lines:
            if "Processing" in line and "- train" in line or "- test" in line:
                stages.append(line.strip())
        
        print("=" * 70)
        print("VISION PIPELINE PROGRESS")
        print("=" * 70)
        print(f"Log file: {latest_log}")
        print(f"Total lines: {len(lines)}")
        print("\nProcessing stages completed:")
        for stage in stages[-10:]:  # Show last 10 stages
            print(f"  {stage}")
        
        # Count processed files
        processed = len([l for l in lines if "✓" in l and ".pdf:" in l])
        print(f"\nTotal PDFs processed: {processed}")
        
        # Count errors
        errors = len([l for l in lines if "ERROR" in l])
        print(f"Total errors: {errors}")
        
        # Get last activity
        if lines:
            last_line = lines[-1].strip()
            if len(last_line) > 100:
                last_line = last_line[:100] + "..."
            print(f"\nLast activity: {last_line}")
    
    # Check saved results
    results_dir = base_dir / "vision_results"
    if results_dir.exists():
        result_files = list(results_dir.glob("*_results.json"))
        
        print("\n" + "=" * 70)
        print("SAVED RESULTS")
        print("=" * 70)
        
        if result_files:
            for rf in sorted(result_files):
                size = rf.stat().st_size
                mtime = datetime.fromtimestamp(rf.stat().st_mtime)
                print(f"  {rf.name}: {size:,} bytes, modified {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Try to load and show summary
                try:
                    with open(rf, 'r') as f:
                        data = json.load(f)
                    
                    if 'results' in data:
                        n_files = len(data['results'])
                        n_graphs = sum(len(r.get('graphs', [])) for r in data['results'].values())
                        n_tables = sum(len(r.get('tables', [])) for r in data['results'].values())
                        print(f"    → {n_files} files, {n_graphs} graphs, {n_tables} tables")
                except:
                    pass
        else:
            print("No result files found yet")
    
    # Check CSV files for ground truth
    datasets_dir = base_dir / "datasets"
    if datasets_dir.exists():
        csv_files = list(datasets_dir.glob("cv sets marked*.csv"))
        
        print("\n" + "=" * 70)
        print("GROUND TRUTH DATA")
        print("=" * 70)
        
        total_annotations = 0
        for cf in csv_files:
            df = pd.read_csv(cf)
            n_rows = len(df)
            total_annotations += n_rows
            print(f"  {cf.name}: {n_rows} annotations")
        
        print(f"\nTotal annotations: {total_annotations}")
    
    print("\n" + "=" * 70)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def main():
    """Main monitoring loop"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor vision pipeline progress")
    parser.add_argument('--once', action='store_true', help='Run once instead of continuous monitoring')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    if args.once:
        check_progress()
    else:
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        try:
            while True:
                check_progress()
                time.sleep(args.interval)
                print("\n" * 2)  # Clear space for next update
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()