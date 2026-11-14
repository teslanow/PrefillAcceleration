

# CUDA_VISIBLE_DEVICES=4 python latency_longbench_integrated.py \
#         --model_path "/data/model/llama_3_8B_Instruct" \
#         --min-input-len 4096 \
#         --max-input-len 32768 \
#         --exp_name "ours_llama_sepeed0.5" \
#         --expected_speedup 0.5 \
#         --speedup_file "configs/llama-3.1-8b_summed.pkl" \
#         --max-new-tokens 32
#         --max-new-tokens 32 \

CUDA_VISIBLE_DEVICES=4 python latency_longbench_integrated.py \
        --model_path "/data/model/Qwen2___5-7B-Instruct/" \
        --min-input-len 4096 \
        --max-input-len 32768 \
        --exp_name "ours_qwen_sepeed0.5" \
        --expected_speedup 0.5 \
        --speedup_file "configs/qwen-2.5-7b_summed.pkl" \
        --max-new-tokens 32