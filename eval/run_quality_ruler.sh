
#!/bin/bash

GPU_ID=$1
SPEEDUP_RATE=$2
L=$2 

if [ -z "$L" ]; then
    echo "Usage: bash run.sh <gpu_id> <speedup_rate> <length>"
    echo "Example: bash run.sh 0 0.55 4096"
    exit 1
fi

MODEL_PATH="/data/model/llama_3_8B_Instruct"
DATA_ROOT="/data/ruler/llama3.1-8b-chat/synthetic"

SPEEDUP_FILE="configs/llama-3.1-8b_summed.pkl"

PYRAMID_DIR="pred_pyramid_sp${SPEEDUP_RATE}_sink${SINK_LEN}_ctx${CONTEXT_LEN}"

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

echo "======================================================"
echo "Starting PyramidCache Evaluation"
echo "Length:  $L"
echo "Speedup: $SPEEDUP_RATE"
echo "Sink:    $SINK_LEN"
echo "Context: $CONTEXT_LEN"
echo "======================================================"

for T in "${TASKS[@]}"; do
    INPUT_FILE="${DATA_ROOT}/${L}/data/${T}/validation.jsonl"
    
    OUTPUT_FILE="${DATA_ROOT}/${L}/${PYRAMID_DIR}/${T}.jsonl"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: $INPUT_FILE not found, skipping..."
        continue
    fi
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $T at length $L, already predicted."
        continue
    fi

    echo "Testing $T at length $L with PyramidCache..."

    mkdir -p "$(dirname "$OUTPUT_FILE")"

    CUDA_VISIBLE_DEVICES=$GPU_ID python super_param_ruler.py \
        --model_path "$MODEL_PATH" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --expected_speedup "$SPEEDUP_RATE" \
        --speedup_file "$SPEEDUP_FILE" 
done

echo "Done for length $L."