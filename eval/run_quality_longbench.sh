#!/bin/bash

set -o pipefail

EXPECTED_SPEEDUP=$1
GPU_ID=$2

if [ -z "$EXPECTED_SPEEDUP" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: bash run.sh <expected_speedup> <gpu_id>"
    echo "Example: bash run.sh 0.5 0"
    exit 1
fi

PYTHON_SCRIPT="super_param_longbench.py"
IS_E_DATASET=false

MODEL_PATH="/data/model/llama_3_8B_Instruct"
SPEEDUP_FILE="configs/llama-3.1-8b_summed.pkl"
EXP_NAME="llama_sp${EXPECTED_SPEEDUP}_sink${SINK_LEN}_ctx${CONTEXT_LEN}_test"
LOG_FILE="log/log_${EXP_NAME}.log"

MODEL_PATH="/data/model/Qwen2___5-7B-Instruct"
SPEEDUP_FILE="configs/qwen-2.5-7b_summed.pkl"
EXP_NAME="qwen_sp${EXPECTED_SPEEDUP}_sink${SINK_LEN}_ctx${CONTEXT_LEN}_test"
LOG_FILE="log/log_${EXP_NAME}.log"

mkdir -p log

if [ "$IS_E_DATASET" = true ]; then
    ALL_DATASETS=(
        "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "multi_news" 
        "trec" "triviaqa" "samsum" "passage_count" "passage_retrieval_en" 
        "lcc" "lsht" "repobench-p" "gov_report" "narrativeqa"
    )
    E_FLAG="--e"
else
    ALL_DATASETS=(
        "passage_retrieval_zh"
        "multifieldqa_zh"
        "trec"
        "multi_news"
        "multifieldqa_en"
        "passage_retrieval_en"
        "2wikimqa"
        "hotpotqa"
        "musique"
        "samsum"
        "qasper"
        "dureader"
        "triviaqa"
        "passage_count"
        "lcc"
        "qmsum"
        "vcsum"
        "lsht"
        "repobench-p"
        "gov_report"
        "narrativeqa"
    )
    E_FLAG=""
fi

echo "------------------------------------------------"
echo "Using GPU: $GPU_ID"
echo "Speedup: $EXPECTED_SPEEDUP"
echo "Sink Len: $SINK_LEN"
echo "Context Len: $CONTEXT_LEN"
echo "Exp Name: $EXP_NAME"
echo "Total Datasets: ${#ALL_DATASETS[@]}"
echo "Log File: $LOG_FILE"
echo "------------------------------------------------"

for DATASET in "${ALL_DATASETS[@]}"
do
    echo ""
    echo "================================================"
    echo "Starting Dataset: $DATASET"
    echo "================================================"

    RESULT_FILE="./results_hyper/${EXP_NAME}/${DATASET}.jsonl"

    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT \
        --model_path "$MODEL_PATH" \
        --data_dir /data/longbench/data/ \
        --exp_name "$EXP_NAME" \
        $E_FLAG \
        --target_datasets "$DATASET" \
        --expected_speedup "$EXPECTED_SPEEDUP" \
        --speedup_file "$SPEEDUP_FILE" \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Dataset $DATASET failed (exit code=$EXIT_CODE)"
        continue
    fi

    echo "✅ Dataset $DATASET completed"
done

python eval.py --exp_path "./results_hyper/${EXP_NAME}/"

echo "------------------------------------------------"
echo "All datasets processed! Results saved in results/$EXP_NAME/"
echo "------------------------------------------------"