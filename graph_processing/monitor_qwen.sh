#!/bin/bash

# Скрипт мониторинга прогресса Qwen2.5-VL pipeline

RESULTS_DIR="vision_results_qwen"
TOTAL_FILES=2307

while true; do
    clear
    echo "================================================================================"
    echo "                      QWEN2.5-VL-72B PROCESSING MONITOR"
    echo "================================================================================"
    echo ""
    echo "📊 Progress by Dataset:"
    echo ""
    
    TOTAL_PROCESSED=0
    
    # Проверяем каждый датасет
    for dataset in synergy cytox magnetic nanozymes seltox; do
        for split in train test; do
            RESULT_FILE="$RESULTS_DIR/$dataset/$split/${dataset}_${split}_results.json"
            if [ -f "$RESULT_FILE" ]; then
                # Извлекаем количество обработанных файлов
                PROCESSED=$(grep -o '"pdf_name"' "$RESULT_FILE" 2>/dev/null | wc -l | tr -d ' ')
                TOTAL=$(grep '"total_files"' "$RESULT_FILE" 2>/dev/null | grep -o '[0-9]*' | head -1)
                
                if [ -n "$PROCESSED" ] && [ -n "$TOTAL" ]; then
                    PERCENT=$((PROCESSED * 100 / TOTAL))
                    printf "%-20s: %3d/%3d files (%3d%%)" "$dataset/$split" "$PROCESSED" "$TOTAL" "$PERCENT"
                    
                    # Добавляем прогресс-бар
                    echo -n " ["
                    BAR_LENGTH=20
                    FILLED=$((PERCENT * BAR_LENGTH / 100))
                    for ((i=0; i<$FILLED; i++)); do echo -n "█"; done
                    for ((i=$FILLED; i<$BAR_LENGTH; i++)); do echo -n "░"; done
                    echo "]"
                    
                    TOTAL_PROCESSED=$((TOTAL_PROCESSED + PROCESSED))
                else
                    printf "%-20s: Processing..." "$dataset/$split"
                    echo ""
                fi
            else
                printf "%-20s: Pending" "$dataset/$split"
                echo ""
            fi
        done
    done
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    
    # Общая статистика
    if [ $TOTAL_PROCESSED -gt 0 ]; then
        OVERALL_PERCENT=$((TOTAL_PROCESSED * 100 / TOTAL_FILES))
        echo "📈 Overall Progress: $TOTAL_PROCESSED/$TOTAL_FILES files ($OVERALL_PERCENT%)"
        
        # Оценка времени
        UPTIME=$(ps aux | grep "run_qwen_pipeline.py" | grep -v grep | head -1 | awk '{print $10}')
        if [ -n "$UPTIME" ]; then
            echo "⏱️  Running time: $UPTIME"
        fi
        
        # Скорость обработки
        if [ -n "$UPTIME" ]; then
            # Преобразуем время в секунды (приблизительно)
            MINUTES=$(echo $UPTIME | cut -d: -f1)
            SECONDS=$(echo $UPTIME | cut -d: -f2)
            TOTAL_SECONDS=$((MINUTES * 60 + SECONDS))
            if [ $TOTAL_SECONDS -gt 0 ]; then
                RATE=$((TOTAL_PROCESSED * 3600 / TOTAL_SECONDS))
                echo "⚡ Processing rate: ~$RATE files/hour"
                
                # Оценка оставшегося времени
                REMAINING=$((TOTAL_FILES - TOTAL_PROCESSED))
                if [ $RATE -gt 0 ]; then
                    ETA_HOURS=$((REMAINING / RATE))
                    ETA_MINUTES=$(((REMAINING % RATE) * 60 / RATE))
                    echo "⏳ Estimated time remaining: ${ETA_HOURS}h ${ETA_MINUTES}m"
                fi
            fi
        fi
    else
        echo "📈 Overall Progress: Starting..."
    fi
    
    echo ""
    
    # Проверка, запущен ли процесс
    if pgrep -f "run_qwen_pipeline.py" > /dev/null; then
        echo "✅ Status: Pipeline is running"
        
        # Показываем использование ресурсов
        PID=$(pgrep -f "run_qwen_pipeline.py" | head -1)
        if [ -n "$PID" ]; then
            CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
            MEM=$(ps aux | grep $PID | grep -v grep | awk '{print $4}')
            echo "💻 Resources: CPU: ${CPU}%, Memory: ${MEM}%"
        fi
    else
        echo "⏸️  Status: Pipeline is not running"
        
        # Проверяем, завершился ли процесс
        if [ $TOTAL_PROCESSED -eq $TOTAL_FILES ]; then
            echo "🎉 Processing complete!"
        elif [ $TOTAL_PROCESSED -gt 0 ]; then
            echo "⚠️  Pipeline stopped at $TOTAL_PROCESSED/$TOTAL_FILES files"
        fi
    fi
    
    echo ""
    echo "================================================================================"
    echo "Press Ctrl+C to exit monitoring"
    
    sleep 10
done