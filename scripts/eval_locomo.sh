#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

# Set output directory
outputdir=your_results

# Create output directory
mkdir -p $outputdir

# Run LoCoMo evaluation
python3 eval/locomo_test.py \
    --data data/locomo/locomo10.json \
    --outdir $outputdir \
    --start-idx 0 \
    --memory-api-key "empty" \
    --memory-base-url "http://localhost:8001/v1" \
    --memory-model "qwen2.5-7B-Instruct" \
    --memory-api-type "vllm" \
    --research-api-key "empty" \
    --research-base-url "http://localhost:8001/v1" \
    --research-model "qwen2.5-7B-Instruct" \
    --research-api-type "vllm" \
    --working-api-key "empty" \
    --working-base-url "http://localhost:8001/v1" \
    --working-model "qwen2.5-7B-Instruct" \
    --working-api-type "vllm"
