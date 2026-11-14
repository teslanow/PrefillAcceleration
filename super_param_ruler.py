import argparse
import json
import os
import gc
import random
import math
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import pickle
import bisect
from util import get_adaptive_max_memory

streamingllm_sink_len = None
min_context_length = None

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
        if t_0 < t_1:
            if t_0 < t_2:
                strategy_idx = 0
            else:
                strategy_idx = 2
        else:
            if t_1 < t_2:
                strategy_idx = 0
            else:
                strategy_idx = 2
        return strategy_idx

class RatiosManager:
    def __init__(self, format_data):
        self.format_data = format_data
    
    def _get_index(self, real_ratio):
        return math.ceil((real_ratio - self.a) / self.b)

def set_proper_ratios(ratiosManager: RatiosManager, model, real_length, expected_speedup):
    if expected_speedup not in ratiosManager.format_data:
        raise RuntimeError(f"The format file does not contain data for speedup factor: {expected_speedup}")
    ratios = ratiosManager.format_data[expected_speedup]
    setattr(model.config, "layer_keeping_ratios", ratios)
    return model
    
def parse_args():
    parser = argparse.ArgumentParser(description="Custom PyramidCache Testing for RULER")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model directory")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the RULER validation.jsonl file")
    parser.add_argument('--output_file', type=str, required=True, help="Path for the RULER prediction output file")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--expected_speedup", type=float, default=0.5, help="Target speedup ratio")
    parser.add_argument("--speedup_file", type=str, default="configs/llama-3.1-8b_summed.pkl")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--streamingllm_sink_len", type=int, default=None)
    parser.add_argument("--min_context_length", type=int, default=None)
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_predictions_ruler(model, tokenizer, cache, input_file, output_file, max_gen, ratiosManager, expected_speedup):
    setattr(model.config, "streamingllm_sink_len", streamingllm_sink_len)
    setattr(model.config, "min_context_length", min_context_length)

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"Evaluating {len(data)} samples from {input_file}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "a", encoding="utf-8") as f_out:
        for json_obj in tqdm(data):
            input_text = json_obj["input"]
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                cache.clear() 
                set_proper_ratios(ratiosManager, model, inputs["input_ids"].shape[1], expected_speedup)
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_gen, 
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    past_key_values=cache,
                )
            
            pred = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )

            del inputs
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
            
            json_obj["pred"] = pred
            f_out.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            f_out.flush()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16

    model_type = "llama" if "llama" in args.model_path.lower() else "qwen"
    target_datasets = os.path.basename(args.input_file).split('.')[0]

    sink_val = args.streamingllm_sink_len
    ctx_val = args.min_context_length

    if sink_val is None or ctx_val is None:
        try:
            with open("configs/hyper_parameter.json", "r", encoding="utf-8") as f:
                hp_config = json.load(f)
                if isinstance(hp_config, list):
                    for item in hp_config:
                        if (item.get("model_type") == model_type and 
                            (item.get("target_datasets") == target_datasets or item.get("target_datasets") == "all") and 
                            abs(float(item.get("speedup", 0)) - args.expected_speedup) < 1e-5):
                            sink_val = item.get("streamingllm_sink_len") if sink_val is None else sink_val
                            ctx_val = item.get("min_context_length") if ctx_val is None else ctx_val
                            break
                elif isinstance(hp_config, dict):
                    sp_key = str(args.expected_speedup)
                    data = hp_config.get(model_type, {}).get(target_datasets, {}).get(sp_key, {})
                    if data:
                        sink_val = data.get("streamingllm_sink_len") if sink_val is None else sink_val
                        ctx_val = data.get("min_context_length") if ctx_val is None else ctx_val
        except Exception:
            pass
        
        if sink_val is None: sink_val = 8
        if ctx_val is None: ctx_val = 64

    streamingllm_sink_len = sink_val
    min_context_length = ctx_val

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
    else:
        raise RuntimeError("Model type not supported")

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
    model.eval()

    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cache = PyramidCache(1, model.config.num_key_value_heads, head_dim, num_layers, model.dtype, runtime_expand=True)

    get_predictions_ruler(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        input_file=args.input_file,
        output_file=args.output_file,
        max_gen=args.max_new_tokens,
        ratiosManager=ratiosManager,
        expected_speedup=args.expected_speedup
    )