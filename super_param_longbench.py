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
        if t_0 < t_2:
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
        raise RuntimeError(f"The format file does not contain data for speedup: {expected_speedup}")

    ratios = ratiosManager.format_data[expected_speedup]
    setattr(model.config, "layer_keeping_ratios", ratios)
    return model
    
def parse_args():
    parser = argparse.ArgumentParser(description="Custom Llama PyramidCache Testing")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="/data/longbench/data/")
    parser.add_argument('--dataset2prompt', type=str, default="./configs/dataset2prompt.json")
    parser.add_argument('--dataset2maxlen', type=str, default="./configs/dataset2maxlen.json")
    parser.add_argument('--exp_name', type=str, default="pyramid_cache_test")
    parser.add_argument('--e', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_datasets', type=str, default=None)
    parser.add_argument("--expected_speedup", type=float, default=0.5)
    parser.add_argument("--speedup_file", type=str, default="configs/llama-3.1-8b_summed.pkl")
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

def get_predictions(model, tokenizer, cache, data, prompt_format, dataset_name, output_path, max_gen, ratiosManager, expected_speedup):
    print(f"Evaluating {dataset_name} with {len(data)} samples...")
    setattr(model.config, "streamingllm_sink_len", streamingllm_sink_len)
    setattr(model.config, "min_context_length", min_context_length)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        if dataset_name in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        else:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

        with torch.no_grad():
            cache.clear()
            set_proper_ratios(ratiosManager, model, inputs["input_ids"].shape[1], expected_speedup)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_gen, 
                do_sample=False,
                temperature=0,
                top_p=1,
                pad_token_id=tokenizer.pad_token_id,
                past_key_values=cache,
            )
        
        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def get_predictions_qwen(model, tokenizer, cache, data, prompt_format, dataset_name, output_path, max_gen, ratiosManager, expected_speedup):
    print(f"Evaluating {dataset_name} with {len(data)} samples...")
    setattr(model.config, "streamingllm_sink_len", streamingllm_sink_len)
    setattr(model.config, "min_context_length", min_context_length)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        if dataset_name in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            input_text = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            cache.clear()
            set_proper_ratios(ratiosManager, model, inputs["input_ids"].shape[1], expected_speedup)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_gen, 
                do_sample=False,
                temperature=0,
                top_p=1,
                pad_token_id=tokenizer.pad_token_id,
                past_key_values=cache,
            )
        
        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    dtype = torch.bfloat16
    model_type = "llama" if "llama" in args.model_path.lower() else "qwen"
    
    sink_len = args.streamingllm_sink_len
    ctx_len = args.min_context_length
    
    if sink_len is None or ctx_len is None:
        json_sink = None
        json_ctx = None
        try:
            with open("configs/hyper_parameter.json", "r", encoding="utf-8") as f:
                hp_data = json.load(f)
                if isinstance(hp_data, list):
                    for item in hp_data:
                        if (item.get("model_type") == model_type and 
                            item.get("target_datasets") == args.target_datasets and 
                            abs(float(item.get("speedup", 0)) - args.expected_speedup) < 1e-5):
                            json_sink = item.get("streamingllm_sink_len")
                            json_ctx = item.get("min_context_length")
                            break
                elif isinstance(hp_data, dict):
                    if (model_type in hp_data and 
                        args.target_datasets in hp_data[model_type] and 
                        str(args.expected_speedup) in hp_data[model_type][args.target_datasets]):
                        match_data = hp_data[model_type][args.target_datasets][str(args.expected_speedup)]
                        json_sink = match_data.get("streamingllm_sink_len")
                        json_ctx = match_data.get("min_context_length")
        except Exception:
            pass
            
        if sink_len is None:
            if json_sink is not None:
                sink_len = json_sink 
            else:
                if "llama" in args.model_path.lower():
                    sink_len = 16
                elif "qwen" in args.model_path.lower():
                    sink_len = 64
                else:
                    sink_len = 16
        if ctx_len is None:
            if json_ctx is not None:
                ctx_len = json_ctx
            else:
                if "llama" in args.model_path.lower():
                    sink_len = 16
                elif "qwen" in args.model_path.lower():
                    sink_len = 8
                else:
                    sink_len = 16

    streamingllm_sink_len = sink_len
    min_context_length = ctx_len

    dataset_name = args.target_datasets
    output_dir = f"./results_hyper/{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    suffix = "_e" if args.e else ""
    file_path = os.path.join(args.data_dir, f"{dataset_name}{suffix}.jsonl")
    if not os.path.exists(file_path):
        print(f"Skip {dataset_name}: File not found")
        exit(0)

    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = sum(1 for _ in f)
        if existing_lines >= len(data):
            print(f"File {output_path} already has all {len(data)} records. Exiting.")
            exit(0)
        else:
            print(f"Incomplete output found ({existing_lines}/{len(data)} records). Resuming.")
            data = data[existing_lines:]

    if "llama" in args.model_path.lower():
        import models.modeling_llama_ours as Model_module
        from models.modeling_llama_ours import LlamaForCausalLM as ModelForCausalLM
        from models.modeling_llama_ours import PyramidCache
        with open('configs/llama.pkl', 'rb') as f:
            x = pickle.load(f)
            Model_module.forward_scheduler = CustomForwardScheduler(x['strategy_0'], x['strategy_1'], x['strategy_2'])
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
        with open(args.speedup_file, 'rb') as f:
            format_data = pickle.load(f)
        ratiosManager = RatiosManager(format_data)
    else:
        raise RuntimeError("Model not supported")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    custom_max_memory = get_adaptive_max_memory(model_size_gb=15)
    
    if custom_max_memory is None:
        model = ModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
    else:
        model = ModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map='auto', trust_remote_code=True, max_memory=custom_max_memory)
    model.eval()

    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cache = PyramidCache(1, model.config.num_key_value_heads, head_dim, num_layers, model.dtype, runtime_expand=True)

    dataset2prompt = json.load(open(args.dataset2prompt, "r"))
    dataset2maxlen = json.load(open(args.dataset2maxlen, "r"))
    prompt_format = dataset2prompt[dataset_name]
    max_gen = dataset2maxlen[dataset_name]

    if "llama" in args.model_path.lower(): 
        get_predictions(
            model=model, tokenizer=tokenizer, cache=cache, data=data, 
            dataset_name=dataset_name, prompt_format=prompt_format, 
            max_gen=max_gen, output_path=output_path, 
            ratiosManager=ratiosManager, expected_speedup=args.expected_speedup
        )
    elif "qwen" in args.model_path.lower():
        get_predictions_qwen(
            model=model, tokenizer=tokenizer, cache=cache, data=data, 
            dataset_name=dataset_name, prompt_format=prompt_format, 
            max_gen=max_gen, output_path=output_path, 
            ratiosManager=ratiosManager, expected_speedup=args.expected_speedup
        )