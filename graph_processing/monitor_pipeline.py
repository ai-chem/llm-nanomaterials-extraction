#!/usr/bin/env python3
"""
Monitor vision pipeline progress in real-time
"""

import time
import re
from pathlib import Path
from datetime import datetime, timedelta
import sys

def get_progress_stats(log_file="vision_pipeline_continuation.log"):
    """Extract progress statistics from log file"""
    
    if not Path(log_file).exists():
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    stats = {
        'start_time': None,
        'current_dataset': None,
        'current_split': None,
        'processed_files': 0,
        'errors': 0,
        'current_progress': None,
        'extraction_count': 0
    }
    
    # Find processing stages and progress
    for line in lines:
        # Check for dataset processing start
        if "Processing" in line and ("- train" in line or "- test" in line):
            match = re.search(r'Processing (\w+) - (\w+)', line)
            if match:
                stats['current_dataset'] = match.group(1)
                stats['current_split'] = match.group(2)
        
        # Check for start time
        if stats['start_time'] is None and re.match(r'\d{4}-\d{2}-\d{2}', line):
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                stats['start_time'] = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        
        # Count processed files
        if "✓" in line and ".pdf:" in line:
            stats['processed_files'] += 1
        
        # Count errors
        if "ERROR" in line:
            stats['errors'] += 1
        
        # Count extractions
        if "✓ Extracted" in line:
            stats['extraction_count'] += 1
        
        # Get progress bar
        if "%" in line and "|" in line:
            # Extract progress like "magnetic-train:  3%|▎         | 18/602"
            match = re.search(r'(\w+-\w+):\s+(\d+)%\|.*\|\s+(\d+)/(\d+)', line)
            if match:
                stats['current_progress'] = {
                    'dataset': match.group(1),
                    'percent': int(match.group(2)),
                    'current': int(match.group(3)),
                    'total': int(match.group(4))
                }
    
    return stats

def display_dashboard(stats):
    """Display progress dashboard"""
    
    # Clear screen for update
    print("\033[2J\033[H")  # Clear screen and move cursor to top
    
    print("=" * 70)
    print("VISION PIPELINE MONITOR - GLM-4.1V")
    print("=" * 70)
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if stats['start_time']:
        elapsed = datetime.now() - stats['start_time']
        print(f"Elapsed Time: {str(elapsed).split('.')[0]}")
    
    print("\n" + "-" * 70)
    print("CURRENT STATUS")
    print("-" * 70)
    
    if stats['current_dataset']:
        print(f"Dataset: {stats['current_dataset']}")
        print(f"Split:   {stats['current_split']}")
    
    if stats['current_progress']:
        prog = stats['current_progress']
        print(f"\nProgress: {prog['dataset']}")
        print(f"  {prog['current']}/{prog['total']} files ({prog['percent']}%)")
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * prog['percent'] / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  [{bar}] {prog['percent']}%")
        
        # Time estimate
        if prog['current'] > 0 and stats['start_time']:
            elapsed = datetime.now() - stats['start_time']
            per_file = elapsed.total_seconds() / prog['current']
            remaining = (prog['total'] - prog['current']) * per_file
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"  ETA: {eta.strftime('%H:%M:%S')} ({timedelta(seconds=int(remaining))})")
    
    print("\n" + "-" * 70)
    print("STATISTICS")
    print("-" * 70)
    print(f"Files Processed:     {stats['processed_files']}")
    print(f"Extractions Made:    {stats['extraction_count']}")
    print(f"Errors Encountered:  {stats['errors']}")
    
    if stats['processed_files'] > 0:
        success_rate = (stats['processed_files'] - stats['errors']) / stats['processed_files'] * 100
        print(f"Success Rate:        {success_rate:.1f}%")
        
        if stats['start_time']:
            elapsed = datetime.now() - stats['start_time']
            rate = stats['processed_files'] / (elapsed.total_seconds() / 60)
            print(f"Processing Rate:     {rate:.1f} files/min")
    
    print("\n" + "-" * 70)
    print("REMAINING DATASETS")
    print("-" * 70)
    
    # Check which datasets are complete
    completed = []
    for dataset in ['magnetic', 'nanozymes', 'seltox']:
        for split in ['train', 'test']:
            result_file = Path(f"vision_results/{dataset}/{split}/{dataset}_{split}_results.json")
            if result_file.exists():
                completed.append(f"{dataset}-{split}")
    
    all_datasets = ['magnetic-train', 'magnetic-test', 'nanozymes-train', 
                   'nanozymes-test', 'seltox-train', 'seltox-test']
    
    remaining = [d for d in all_datasets if d not in completed]
    
    print(f"Completed: {', '.join(completed) if completed else 'None'}")
    print(f"Remaining: {', '.join(remaining) if remaining else 'None'}")
    
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop monitoring")

def main():
    """Main monitoring loop"""
    
    print("Starting pipeline monitor...")
    print("Reading from: vision_pipeline_continuation.log")
    
    try:
        while True:
            stats = get_progress_stats()
            
            if stats:
                display_dashboard(stats)
            else:
                print("Waiting for log file...")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()