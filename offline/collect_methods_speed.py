import torch
import os
import csv
import time
import numpy as np
from tqdm import tqdm

from models.modeling_llama_ours import LlamaDecoderLayer, LlamaRotaryEmbedding, LlamaConfig, PyramidCache
import models.modeling_llama_ours as llama_module

from models.modeling_qwen2_ours import Qwen2DecoderLayer, Qwen2RotaryEmbedding, Qwen2Config
import models.modeling_qwen2_ours as qwen_module

CURRENT_STRATEGY = 0

def custom_forward_scheduler(seq_len, must_keep_len, top_k_ratio):
    return CURRENT_STRATEGY

llama_module.forward_scheduler = custom_forward_scheduler
qwen_module.forward_scheduler = custom_forward_scheduler

def get_model_setup(model_type, device, dtype):
    if model_type == "llama":
        config = LlamaConfig(
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            attention_bias=False,
            mlp_bias=False
        )
        layer_cls = LlamaDecoderLayer
        rope_cls = LlamaRotaryEmbedding
    elif model_type == "qwen":
        config = Qwen2Config(
            hidden_size=3584,
            intermediate_size=18944,
            num_attention_heads=28,
            num_key_value_heads=4, 
            max_position_embeddings=131072,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            attention_bias=True, 
            mlp_bias=False
        )
        layer_cls = Qwen2DecoderLayer
        rope_cls = Qwen2RotaryEmbedding
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    config.min_context_length = 64
    config.streamingllm_sink_len = 64
    config.layer_keeping_ratios = [1.0] * config.num_hidden_layers
    
    layer = layer_cls(config, layer_idx=0).to(device, dtype=dtype).eval()
    rotary_emb = rope_cls(config=config).to(device)
    
    return config, layer, rotary_emb

def run_benchmark():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    model_types = ["qwen"] 
    seq_lens = list(range(1024, 65537, 2048)) 
    top_k_ratios = [round(r, 2) for r in np.arange(0.1, 0.91, 0.05)]
    strategies = [2] 
    
    warmup_steps = 3
    test_steps = 10
    batch_size = 1

    global CURRENT_STRATEGY
    
    for model_type in model_types:
        save_dir = f"results/strategy_speed/{model_type}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"\n{'='*20} Preparing {model_type.upper()} benchmark {'='*20}")
        
        config, layer, rotary_emb = get_model_setup(model_type, device, dtype)
        head_dim = config.hidden_size // config.num_attention_heads
        
        print(f"Configuration info: Hidden={config.hidden_size}, KV_Heads={config.num_key_value_heads}, Head_Dim={head_dim}")

        for strategy in strategies:
            CURRENT_STRATEGY = strategy
            file_path = os.path.join(save_dir, f"strategy_{strategy}.csv")
            
            with open(file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["seq_len", "top_k_ratio", "avg_ms", "std_ms"])

            print(f"\n[{model_type.upper()} | Strategy {strategy}] Benchmark in progress...")

            for seq_len in tqdm(seq_lens, desc=f"{model_type} Strat {strategy}"):
                hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                selected_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                sink_indices = torch.arange(0, 64, device=device, dtype=torch.long)
                
                dummy_qkv = torch.zeros(batch_size, seq_len, config.num_key_value_heads, head_dim, device=device, dtype=dtype)
                cos, sin = rotary_emb(dummy_qkv, position_ids)
                position_embeddings = (cos, sin)
                
                for ratio in top_k_ratios:
                    cache = PyramidCache(
                        max_b=batch_size,
                        h=config.num_key_value_heads,
                        d=head_dim,
                        layer_num=1,
                        dtype=dtype,
                        runtime_expand=True
                    )

                    def run_iter():
                        cache.clear()
                        with torch.no_grad():
                            return layer(
                                hidden_states=hidden_states,
                                position_ids=position_ids,
                                past_key_value=cache,
                                use_cache=True,
                                is_prefill=True,
                                min_context_length=64,
                                streamingllm_sink_len=64,
                                sink_indices=sink_indices,
                                top_k_ratio=ratio,
                                position_embeddings=position_embeddings,
                                selected_position_ids=selected_position_ids
                            )

                    for _ in range(warmup_steps):
                        run_iter()
                    torch.cuda.synchronize()

                    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(test_steps)]
                    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(test_steps)]

                    for i in range(test_steps):
                        start_events[i].record()
                        run_iter()
                        end_events[i].record()
                    
                    torch.cuda.synchronize()
                    
                    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
                    avg_ms = np.mean(times)
                    std_ms = np.std(times)

                    with open(file_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([seq_len, ratio, f"{avg_ms:.4f}", f"{std_ms:.4f}"])
                    
                    del cache
                    torch.cuda.empty_cache()

                del hidden_states, position_ids, selected_position_ids, cos, sin
                torch.cuda.empty_cache()

        del layer, rotary_emb
        torch.cuda.empty_cache()

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_benchmark()