#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

export HF_DATASETS_CACHE=".cache/huggingface/datasets"
export TRANSFORMERS_CACHE=".cache/huggingface/transformers"
export HF_HOME=".cache/huggingface"

mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE


outputdir=your_output_path

# Run one seed at a time. Replace with either commented value for another run.
SEED=43
# SEED=42
# SEED=44

# Create output directory
mkdir -p $outputdir

# Run NarrativeQA evaluation
python3 eval/narrativeqa_test.py \
    --data-dir data/narrativeqa \
    --split test \
    --outdir $outputdir \
    --start-idx 0 \
    --end-idx 300 \
    --max-tokens 2048 \
    --seed $SEED \
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
    --working-api-type "vllm" \
    --embedding-model-path your_path_/BAAI/bge-m3
