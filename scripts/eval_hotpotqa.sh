#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

# Set output directory
base_outputdir=your_output_path

# Create output directory
mkdir -p $base_outputdir

# Run HotpotQA evaluation
for dataset in "eval_400" "eval_1600" "eval_3200"
do
    echo "Processing dataset: $dataset"
    outputdir=$base_outputdir/${dataset}

    python3 eval/hotpotqa_test.py \
        --data data/hotpotqa/${dataset}.json \
        --outdir $outputdir \
        --start-idx 0 \
        --max-tokens 2048 \
        --memory-api-key "empty" \
        --memory-base-url "http://localhost:8002/v1" \
        --memory-model "qwen2.5-7B-Instruct" \
        --memory-api-type "vllm" \
        --research-api-key "empty" \
        --research-base-url "http://localhost:8002/v1" \
        --research-model "qwen2.5-7B-Instruct" \
        --research-api-type "vllm" \
        --working-api-key "empty" \
        --working-base-url "http://localhost:8002/v1" \
        --working-model "qwen2.5-7B-Instruct" \
        --working-api-type "vllm"
done
