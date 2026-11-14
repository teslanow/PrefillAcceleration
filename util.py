import torch
import math

def load_config(model, config):
    prefill_config = config['prefill_stage']
    for k, v in prefill_config.items():
        setattr(model.config, k, v)
    return model

def get_adaptive_max_memory(model_size_gb):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return None
    elif num_gpus == 2:
        return {0 : '8GiB', 1: '11GiB'}
    elif num_gpus == 3:
        return {0 : '4GiB', 1: '6GiB', 2: '9GiB'}
    elif num_gpus == 4:
        return {0 : '4GiB', 1: '4GiB', 2: '6GiB', 3: '6GB'}
    else:
        raise RuntimeError("Too much gpus")

def evalute_speed_up(format_data, c2_div_c1):
    def get_actual_speedup(A_retention, c2_div_c1, token_num):
        A_ret = torch.as_tensor(A_retention, dtype=torch.float32)
        if A_ret.dim() == 1:
            A_ret = A_ret.unsqueeze(0)
        
        A_cum = torch.cumprod(A_ret, dim=1)
        
        layer_num = A_cum.shape[1]
        gamma = c2_div_c1 / token_num
        term1 = torch.sum(A_cum ** 2, dim=1)
        term2 = torch.sum(A_cum, dim=1)
        
        g = (term1 + gamma * term2) / ((1 + gamma) * layer_num)
        return g.item()
    for expected_ratio in format_data.keys():
        # for length in format_data[expected_ratio].keys():
        print(f"Expected: {expected_ratio}, Estimated: {get_actual_speedup(format_data[expected_ratio], c2_div_c1, c2_div_c1)}")

if __name__ == "__main__":
    import pickle
    from offline.training_with_cab_set2 import helper_caculate_c2_div_c1
    with open('configs/qwen-2.5-7b_summed.pkl', 'rb') as f:
        format_data = pickle.load(f)
        evalute_speed_up(format_data, helper_caculate_c2_div_c1('qwen-2.5-7b'))