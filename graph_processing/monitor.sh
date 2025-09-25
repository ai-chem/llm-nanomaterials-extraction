#!/bin/bash

# Скрипт для мониторинга прогресса обработки

clear
echo "========================================"
echo "  МОНИТОРИНГ VISION PIPELINE (GLM-4.1V)"
echo "========================================"
echo ""

# Подсчет статистики
TOTAL_PROCESSED=$(grep -c "✓.*\.pdf:" vision_pipeline_continuation.log 2>/dev/null || echo 0)
ERRORS=$(grep -c "ERROR" vision_pipeline_continuation.log 2>/dev/null || echo 0)

# Текущее время
echo "Время: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Общая статистика
echo "📊 ОБЩАЯ СТАТИСТИКА:"
echo "-------------------"
echo "Обработано файлов: $TOTAL_PROCESSED"
echo "Ошибок: $ERRORS"
echo ""

# Текущий прогресс
echo "📈 ТЕКУЩИЙ ПРОГРЕСС:"
echo "-------------------"
grep "%" vision_pipeline_continuation.log | tail -1
echo ""

# Последние обработанные файлы
echo "📄 ПОСЛЕДНИЕ 5 ФАЙЛОВ:"
echo "----------------------"
grep "✓.*\.pdf:" vision_pipeline_continuation.log | tail -5 | sed 's/.*✓ /✓ /'
echo ""

# Проверка какой датасет обрабатывается
CURRENT_DATASET=$(tail -100 vision_pipeline_continuation.log | grep -E "Processing (magnetic|nanozymes|seltox) - (train|test)" | tail -1)
if [ ! -z "$CURRENT_DATASET" ]; then
    echo "🔄 ТЕКУЩИЙ ДАТАСЕТ:"
    echo "-------------------"
    echo "$CURRENT_DATASET"
    echo ""
fi

# Оценка времени
if [ $TOTAL_PROCESSED -gt 0 ]; then
    echo "⏱️  ОЦЕНКА ВРЕМЕНИ:"
    echo "-------------------"
    
    # Примерные размеры датасетов
    MAGNETIC_TRAIN=602
    MAGNETIC_TEST=152
    NANOZYMES_TRAIN=672
    NANOZYMES_TEST=118
    SELTOX_TRAIN=266
    SELTOX_TEST=62
    TOTAL_FILES=$((MAGNETIC_TRAIN + MAGNETIC_TEST + NANOZYMES_TRAIN + NANOZYMES_TEST + SELTOX_TRAIN + SELTOX_TEST))
    
    REMAINING=$((TOTAL_FILES - TOTAL_PROCESSED))
    echo "Осталось файлов: ~$REMAINING"
    
    # Средняя скорость ~30 сек на файл
    REMAINING_TIME=$((REMAINING * 30 / 60))
    echo "Примерное время до завершения: ~$REMAINING_TIME минут (~$((REMAINING_TIME / 60)) часов)"
fi

echo ""
echo "========================================"
echo "Нажмите Ctrl+C для выхода"
echo "Обновление каждые 10 секунд..."