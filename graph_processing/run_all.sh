#!/bin/bash

# Script to run vision pipeline on all datasets
# Usage: ./run_all.sh [train|test|all]

SPLIT=${1:-all}

echo "=========================================="
echo "Vision Pipeline Batch Processing"
echo "=========================================="
echo "Split: $SPLIT"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Test connections first
echo "Testing VLM connections..."
python test_pipeline.py

read -p "Continue with batch processing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Run based on split
if [ "$SPLIT" = "train" ] || [ "$SPLIT" = "all" ]; then
    echo ""
    echo "Processing TRAIN datasets..."
    python run_vision_pipeline.py --splits train --workers 2
fi

if [ "$SPLIT" = "test" ] || [ "$SPLIT" = "all" ]; then
    echo ""
    echo "Processing TEST datasets..."
    python run_vision_pipeline.py --splits test --workers 2
fi

echo ""
echo "=========================================="
echo "Processing Complete!"
echo "Results saved in: vision_results/"
echo "==========================================" 