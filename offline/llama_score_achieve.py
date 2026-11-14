import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaAttention, 
    apply_rotary_pos_emb, 
    repeat_kv, 
    ALL_ATTENTION_FUNCTIONS
)
import os
import json
import time
import argparse
import random
import gc
import math
from tqdm import tqdm

layer_results = {}
WINDOW_SIZE = 64

def optimized_llama_forward_patch(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: torch.Tensor = None,
    past_key_value = None,
    cache_position = None,
    **kwargs,
):
    output_attentions = kwargs.get("output_attentions", False)
    
    bsz, q_len, _ = hidden_states.shape
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if output_attentions:
        with torch.no_grad():
            current_ws = min(WINDOW_SIZE, q_len)
            key_states_for_weights = repeat_kv(key_states, self.num_key_value_groups)
            q_last = query_states[:, :, -current_ws:, :]
            
            weights = torch.matmul(q_last, key_states_for_weights.transpose(2, 3)) * self.scaling

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, -current_ws:, :key_states.shape[-2]]
                weights = weights + causal_mask

            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
            
            mean_heads = weights[0].mean(dim=1) 
            
            layer_results[f"layer_{self.layer_idx}"] = {
                "avg": mean_heads.mean(dim=0).cpu(),
                "max": mean_heads.max(dim=0).values.cpu()
            }
            del weights, q_last, key_states_for_weights, mean_heads

    sdpa_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
    
    attn_output, _ = sdpa_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None

LlamaAttention.forward = optimized_llama_forward_patch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="/data/model/llama_3_2_1B_Instruct")
    parser.add_argument('--data_dir', type=str, default="/data/longbench/data")
    parser.add_argument('--output_dir', type=str, default="/data/13_project/profiling_results_gpu")
    parser.add_argument('--max_tokens', type=int, default=65536)
    parser.add_argument('--ws', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    global WINDOW_SIZE
    WINDOW_SIZE = args.ws
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Loading model (SDPA mode, GPU)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model.eval()

    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", 
                "narrativeqa", "multifieldqa_zh", "musique", "dureader", "qmsum"]
    
    config_path = "./configs/dataset2prompt.json"
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found {config_path}")
        return
    with open(config_path, "r") as f:
        dataset2prompt = json.load(f)

    for dataset_name in datasets:
        print(f"\n--- Processing dataset: {dataset_name} ---")
        data_path = os.path.join(args.data_dir, f"{dataset_name}.jsonl")
        if not os.path.exists(data_path): continue
            
        dataset_out_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)
            
            success_count = 0
            count_limit = int(len(lines) * 0.2)
            for line in lines:
                if success_count >= count_limit: break
                
                json_obj = json.loads(line)
                prompt = dataset2prompt.get(dataset_name, "{context}").format(**json_obj)
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                token_count = inputs.input_ids.shape[1]
                
                if token_count <= args.max_tokens:
                    print(f"Sample {success_count+1}/{count_limit}, Length: {token_count}. Inferring...")
                    layer_results.clear()
                    start_time = time.perf_counter()
                    
                    try:
                        with torch.no_grad():
                            model(**inputs, output_attentions=True)
                        
                        if not layer_results:
                            print("⚠️ Warning: No attention layer data captured, please check if the patch is effective.")
                            continue

                        summed_dict = {k: v["avg"] for k, v in layer_results.items()}
                        maxed_dict = {k: v["max"] for k, v in layer_results.items()}
                        
                        duration = time.perf_counter() - start_time
                        torch.save(summed_dict, os.path.join(dataset_out_dir, f"sample_{success_count}_summed.pt"))
                        torch.save(maxed_dict, os.path.join(dataset_out_dir, f"sample_{success_count}_maxed.pt"))
                        
                        with open(os.path.join(args.output_dir, "profile_log.txt"), "a") as log:
                            log.write(f"Dataset: {dataset_name}, Sample: {success_count}, Len: {token_count}, Time: {duration:.4f}s\n")
                        
                        print(f"✅ Success. Time elapsed: {duration:.2f}s")
                        success_count += 1
                        
                    except Exception as e:
                        print(f"❌ Failed: {e}")
                    
                    finally:
                        del inputs
                        torch.cuda.empty_cache()
                        gc.collect()

if __name__ == "__main__":
    main()