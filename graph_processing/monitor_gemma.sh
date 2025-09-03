#!/bin/bash

# Monitor Gemma pipeline progress

echo "========================================="
echo "GEMMA-3-27B PIPELINE MONITOR"
echo "========================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "python.*run_gemma_pipeline.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "⚠️  Pipeline is NOT running"
else
    echo "✅ Pipeline is running (PID: $PID)"
    echo ""
    
    # Show CPU and memory usage
    echo "Process stats:"
    ps aux | grep "python.*run_gemma_pipeline.py" | grep -v grep | awk '{printf "CPU: %s%%, Memory: %s%%\n", $3, $4}'
    echo ""
fi

# Count processed files
echo "Progress by dataset:"
echo "-----------------------------------------"

for dataset in synergy cytox magnetic nanozymes seltox; do
    for split in train test; do
        result_file="vision_results_gemma/$dataset/$split/${dataset}_${split}_results.json"
        if [ -f "$result_file" ]; then
            count=$(grep -o '"pdf_name"' "$result_file" 2>/dev/null | wc -l | tr -d ' ')
            echo "$dataset/$split: $count files processed ✓"
        else
            echo "$dataset/$split: Not started yet"
        fi
    done
done

echo ""
echo "Latest activity:"
echo "-----------------------------------------"

# Show last processed files
if [ -d "vision_results_gemma" ]; then
    latest=$(find vision_results_gemma -name "*.json" -type f -exec ls -lt {} + 2>/dev/null | head -5)
    if [ ! -z "$latest" ]; then
        echo "$latest" | while read line; do
            echo "$line" | awk '{print $6, $7, $8, $9}'
        done
    fi
fi

echo ""
echo "Real-time log (last 10 lines):"
echo "-----------------------------------------"
# Show last log entries from running process
if [ ! -z "$PID" ]; then
    # Try to capture output from the process
    lsof -p $PID 2>/dev/null | grep -E "\.txt|\.log" | awk '{print $NF}' | while read logfile; do
        if [ -f "$logfile" ]; then
            tail -10 "$logfile" 2>/dev/null
            break
        fi
    done
fi

# Estimate completion time
echo ""
echo "Estimation:"
echo "-----------------------------------------"
# Actual file counts per dataset
declare -A file_counts
file_counts[synergy_train]=69
file_counts[synergy_test]=18
file_counts[cytox_train]=278
file_counts[cytox_test]=70
file_counts[magnetic_train]=602
file_counts[magnetic_test]=152
file_counts[nanozymes_train]=632
file_counts[nanozymes_test]=158
file_counts[seltox_train]=262
file_counts[seltox_test]=66

total_files=2307  # Actual total from extracted_pdfs
processed=$(find vision_results_gemma -name "*.json" -type f -exec grep -o '"pdf_name"' {} \; 2>/dev/null | wc -l | tr -d ' ')
if [ "$processed" -gt 0 ]; then
    echo "Total files: $total_files"
    echo "Processed: $processed"
    echo "Remaining: $((total_files - processed))"
    
    # Calculate percentage
    percent=$((processed * 100 / total_files))
    echo "Progress: ${percent}%"
    
    # Progress bar
    echo -n "["
    for i in $(seq 1 50); do
        if [ $i -le $((percent / 2)) ]; then
            echo -n "="
        else
            echo -n " "
        fi
    done
    echo "] ${percent}%"
fi

echo ""
echo "========================================="
echo "Press Ctrl+C to exit monitoring"
echo "To run continuously: watch -n 10 ./monitor_gemma.sh"