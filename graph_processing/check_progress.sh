#!/bin/bash

echo "=========================================="
echo "📊 QWEN2.5-VL PROGRESS CHECK"
echo "=========================================="
echo ""

TOTAL_EXPECTED=2307
TOTAL_PROCESSED=0

# Подсчет обработанных файлов
for dataset in synergy cytox magnetic nanozymes seltox; do
    for split in train test; do
        FILE="vision_results_qwen/$dataset/$split/${dataset}_${split}_results.json"
        if [ -f "$FILE" ]; then
            COUNT=$(grep -o '"pdf_name"' "$FILE" 2>/dev/null | wc -l | tr -d ' ')
            if [ -n "$COUNT" ]; then
                echo "✅ $dataset/$split: $COUNT files"
                TOTAL_PROCESSED=$((TOTAL_PROCESSED + COUNT))
            fi
        else
            echo "⏳ $dataset/$split: pending"
        fi
    done
done

echo ""
echo "------------------------------------------"
PERCENT=$((TOTAL_PROCESSED * 100 / TOTAL_EXPECTED))
echo "📈 TOTAL: $TOTAL_PROCESSED/$TOTAL_EXPECTED ($PERCENT%)"

# Проверка активности процесса
if pgrep -f "run_qwen_pipeline.py" > /dev/null; then
    echo "✅ Status: RUNNING"
    
    # Последняя активность
    LAST_LOG=$(ls -t vision_results_qwen/*/*/*.json 2>/dev/null | head -1)
    if [ -n "$LAST_LOG" ]; then
        LAST_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LAST_LOG" 2>/dev/null || stat -c "%y" "$LAST_LOG" 2>/dev/null | cut -d' ' -f1-2)
        echo "📝 Last update: $LAST_TIME"
    fi
else
    if [ $TOTAL_PROCESSED -eq $TOTAL_EXPECTED ]; then
        echo "🎉 Status: COMPLETED!"
    else
        echo "⚠️ Status: STOPPED at $PERCENT%"
    fi
fi

echo ""