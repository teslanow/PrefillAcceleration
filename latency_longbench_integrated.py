import argparse
import json
import os
import gc
import random
import math
import numpy as np
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import pickle
import bisect
from util import get_adaptive_max_memory

streamingllm_sink_len = None
min_context_length = None

import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CustomForwardScheduler:
    def __init__(self, C_method_0, C_method_1, C_method_2):
        self.c0_0, self.c0_1, self.c0_2, self.c0_3 = C_method_0
        self.c1_0, self.c1_1, self.c1_2, self.c1_3 = C_method_1
        self.c2_0, self.c2_1, self.c2_2 = C_method_2

    def __call__(self, seq_len, must_keep_len, top_k_ratio):
        size_after_pruning = int(seq_len * top_k_ratio)
        if must_keep_len >= size_after_pruning or top_k_ratio >= 1.0:
            return 2
        t_0 = self.c0_0 + self.c0_1 * seq_len + self.c0_2 * top_k_ratio * seq_len + self.c0_3 * (top_k_ratio ** 2) * (seq_len ** 2)
        t_1 = self.c1_0 + self.c1_1 * seq_len + self.c1_2 * top_k_ratio * seq_len + self.c1_3 * (seq_len ** 2)
        t_2 = self.c2_0 + self.c2_1 * seq_len + self.c2_2 * (seq_len ** 2)
        strategy_idx = 2
        if t_0 < t_2:
            strategy_idx = 0
        else:
            strategy_idx = 2
        return strategy_idx

class RatiosManager:
    def __init__(self, format_data):
        self.format_data = format_data
    

def set_proper_ratios(ratiosManager: RatiosManager, model, real_length, expected_speedup):
    if expected_speedup not in ratiosManager.format_data:
        raise RuntimeError(f"your format file has not data for speedup {expected_speedup}")

    ratios = ratiosManager.format_data[expected_speedup]
    setattr(model.config, "layer_keeping_ratios", ratios)

    return model
    
def parse_args():
    parser = argparse.ArgumentParser(description="Custom Llama PyramidCache Latency Benchmark (Full Time)")
    parser.add_argument('--model_path', type=str, required=True, help="such as /data/model/llama_3_8B_Instruct")
    parser.add_argument('--exp_name', type=str, default="pyramid_cache_latency_test")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--expected_speedup", type=float, default=0.5, help="Expected speedup to original, such as 0.5")
    parser.add_argument("--speedup_file", type=str, default=None, help="use which file as speedup data")
    
    parser.add_argument('--min-input-len', type=int, default=4096)
    parser.add_argument('--max-input-len', type=int, default=32768)
    parser.add_argument('--num-points', type=int, default=10, help="Number of points to sample between min and max seq len")
    parser.add_argument('--max-new-tokens', type=int, default=1, help="Number of tokens to generate for full latency measurement")
    parser.add_argument('--num-iters-warmup', type=int, default=3, help="Warmup iterations per sequence length")
    parser.add_argument('--num-iters', type=int, default=10, help="Measurement iterations per sequence length")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_benchmark(
    model, 
    tokenizer,
    cache,
    ratiosManager,
    expected_speedup,
    args,
    output_dir
):
    results = {}

    setattr(model.config, "streamingllm_sink_len", streamingllm_sink_len)
    setattr(model.config, "min_context_length", min_context_length)

    for seq_len in range(args.max_input_len, args.min_input_len - 1, -4096):
        print(f"\n--- Testing Sequence Length: {seq_len} ---")
        
        dummy_input_ids = torch.randint(1000, 10000, (1, seq_len), device=model.device)
        
        set_proper_ratios(ratiosManager, model, seq_len, expected_speedup)

        def measure_latency():
            cache.clear()
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=dummy_input_ids,
                    max_new_tokens=args.max_new_tokens, 
                    do_sample=False,
                    temperature=0,
                    top_p=1,
                    pad_token_id=tokenizer.pad_token_id,
                    past_key_values=cache,
                    use_cache=True,
                    num_logits_to_keep=1
                )
            end_event.record()
            torch.cuda.synchronize()
            
            return start_event.elapsed_time(end_event) / 1000.0
        try:
            print("Warming up...")
            for _ in range(args.num_iters_warmup):
                measure_latency()

            print("Measuring latency...")
            latencies = []
            for _ in tqdm(range(args.num_iters), desc=f"SeqLen {seq_len}"):
                lat = measure_latency()
                latencies.append(lat)
                
                gc.collect()
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:

            torch.cuda.empty_cache() 
            break  
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                torch.cuda.empty_cache()
                break  
            else:
                raise e  
            
        avg_latency = np.mean(latencies)
        p90_latency = np.percentile(latencies, 90)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Avg Full Latency: {avg_latency:.4f} s | P90: {p90_latency:.4f} s | P99: {p99_latency:.4f} s")
        
        results[int(seq_len)] = {
            "avg_latency": avg_latency,
            "p90": p90_latency,
            "p99": p99_latency,
            "all_latencies": latencies
        }

    output_path = os.path.join(output_dir, "latency_benchmark_results.csv")
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.writer(f)
        

        writer.writerow(["Sequence_Length", "Avg_Latency_s", "P90_Latency_s"])
        
        for seq in sorted(results.keys()):
            metrics = results[seq]
            writer.writerow([
                seq, 
                f"{metrics['avg_latency']:.4f}", 
                f"{metrics['p90']:.4f}"
            ])
    print(f"\nAll benchmark results saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16

    if "llama" in args.model_path.lower():
        import models.modeling_llama_ours as Model_module
        from models.modeling_llama_ours import LlamaForCausalLM as ModelForCausalLM
        from models.modeling_llama_ours import PyramidCache
        with open('configs/llama.pkl', 'rb') as f:
            x = pickle.load(f)
            Model_module.forward_scheduler = CustomForwardScheduler(x['strategy_0'], x['strategy_1'], x['strategy_2'])
        
        if args.speedup_file is None:
            with open('configs/llama-3.1-8b_summed.pkl', 'rb') as f:
                format_data = pickle.load(f)
        else:
            with open(args.speedup_file, 'rb') as f:
                format_data = pickle.load(f)
        ratiosManager = RatiosManager(format_data)
        streamingllm_sink_len = 16
        min_context_length = 16

    elif "qwen" in args.model_path.lower():
        import models.modeling_qwen2_ours as Model_module
        from models.modeling_qwen2_ours import Qwen2ForCausalLM as ModelForCausalLM
        from models.modeling_llama_ours import PyramidCache
        with open('configs/qwen.pkl', 'rb') as f:
            x = pickle.load(f)
            Model_module.forward_scheduler = CustomForwardScheduler(x['strategy_0'], x['strategy_1'], x['strategy_2'])
        
        if args.speedup_file is None:
            with open('configs/qwen-2.5-7b_summed.pkl', 'rb') as f:
                format_data = pickle.load(f)
        else:
            with open(args.speedup_file, 'rb') as f:
                format_data = pickle.load(f)
        ratiosManager = RatiosManager(format_data)
        streamingllm_sink_len = 64
        min_context_length = 8

    else:
        raise RuntimeError("Not supported model")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    custom_max_memory = get_adaptive_max_memory(model_size_gb=15)
    if custom_max_memory is None:
        model = ModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map='auto',
            trust_remote_code=True
        )
    else:
        model = ModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map='auto',
            trust_remote_code=True,
            max_memory=custom_max_memory
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cache = PyramidCache(1, model.config.num_key_value_heads, head_dim, num_layers, model.dtype, runtime_expand=True)


    output_dir = f"./results/latency_{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        ratiosManager=ratiosManager,
        expected_speedup=args.expected_speedup,
        args=args,
        output_dir=output_dir
    )