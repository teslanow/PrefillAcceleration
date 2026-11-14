import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import numpy as np
import cma
import argparse
import json
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

def reference_loss_loop(A, dataset_folder):
    device = A.device
    n_layers = len(A)
    total_loss = 0.0
    sample_count = 0
    
    file_list = [f for f in os.listdir(dataset_folder) if f.endswith('.pt')]
    sample_count = len(file_list)

    for name in file_list:
        path = os.path.join(dataset_folder, name)
        data = torch.load(path, map_location=device)
        
        m = data["layer_0"].shape[0]
        d = torch.zeros((n_layers, m), device=device)
        for i in range(n_layers):
            d[i] = data[f"layer_{i}"]
            
        is_pruned = torch.zeros(m, device=device, dtype=torch.bool)
        sample_loss = 0.0
        k_prev = 0
        
        for i in range(n_layers):
            k_cum = int(torch.round(torch.tensor(m * A[i].item())).item())
            delta_k = max(0, k_cum - k_prev)
            
            if delta_k > 0:
                current_layer_scores = d[i].clone()
                current_layer_scores[is_pruned] = -1e9
                _, top_indices = torch.topk(current_layer_scores, delta_k)
                is_pruned[top_indices] = True
            
            layer_loss = (d[i][is_pruned] ** 2).sum()
            sample_loss += layer_loss
            k_prev = k_cum
            
        total_loss += sample_loss
        
    return total_loss / sample_count

class OptimizedDataset:
    def __init__(self, folder, device, batch_num):
        super().__init__()
        self.device = device
        self.batch_num = batch_num
        self.batches = []
        self.batch_masks = []
        self.batch_lengths = []
        self.batch_suffix_sq = [] 
        self.sample_num = 0
        self.minimal_length = None
        self.maximal_length = None
        all_samples = []
        for name in os.listdir(folder):
            if name.endswith(".pt"):
                path = os.path.join(folder, name)
                data = torch.load(path, map_location='cpu')
                layer_num = len(data)
                for layer_no in range(layer_num):
                    if torch.isnan(data[f"layer_{layer_no}"]).any():
                        raise RuntimeError(f"{name} contains NaN")
                dim = data["layer_0"].shape[0]
                sample = torch.zeros(layer_num, dim)
                for layer_no in range(layer_num):
                    sample[layer_no, :] = data[f"layer_{layer_no}"]
                all_samples.append(sample)

        all_samples.sort(key=lambda x: x.shape[1], reverse=True)
        self.sample_num = len(all_samples)
        samples_per_batch = (self.sample_num + batch_num - 1) // batch_num

        self.avg_length = 0
        for i in range(0, self.sample_num, samples_per_batch):
            batch_list = all_samples[i : i + samples_per_batch]
            if not batch_list: continue
            
            curr_bs = len(batch_list)
            max_m = batch_list[0].shape[1]
            n_layers = batch_list[0].shape[0]
            
            padded_batch = torch.zeros((curr_bs, n_layers, max_m), device=device)
            mask = torch.zeros((curr_bs, max_m), device=device, dtype=torch.bool)
            lengths = []
            
            for j, s in enumerate(batch_list):
                m_curr = s.shape[1]
                padded_batch[j, :, :m_curr] = s.to(device)
                mask[j, :m_curr] = True
                lengths.append(m_curr)
                self.minimal_length = min(self.minimal_length, m_curr) if self.minimal_length is not None else m_curr
                self.maximal_length = max(self.maximal_length, m_curr) if self.maximal_length is not None else m_curr
                self.avg_length = self.avg_length * (j / (j + 1)) + m_curr / (j + 1)
            
            sq = padded_batch ** 2
            suffix_sq = torch.flip(torch.cumsum(torch.flip(sq, dims=[1]), dim=1), dims=[1])
            
            self.batches.append(padded_batch)
            self.batch_masks.append(mask)
            self.batch_lengths.append(torch.tensor(lengths, device=device))
            self.batch_suffix_sq.append(suffix_sq)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx], self.batch_masks[idx], self.batch_lengths[idx], self.batch_suffix_sq[idx]

@torch.compile
def compute_loss_optimized(A, batch_data, batch_mask, batch_lengths, suffix_sq):
    n = len(A)
    batch_size, _, max_m = batch_data.shape
    device = A.device
    
    K_cum = torch.round(batch_lengths.unsqueeze(1) * A.unsqueeze(0)).long()
    K_prev = torch.cat([torch.zeros((batch_size, 1), device=device, dtype=torch.long), K_cum[:, :-1]], dim=1)
    delta_K = (K_cum - K_prev).clamp(min=0)

    is_pruned = torch.zeros((batch_size, max_m), device=device, dtype=torch.bool)
    total_loss = torch.tensor(0.0, device=device)
    
    ranks = torch.arange(max_m, device=device).unsqueeze(0)

    for i in range(n):
        max_k = int(delta_K[:, i].max().item())
        if max_k <= 0: continue
            
        current_scores = batch_data[:, i, :].detach().clone()
        current_scores.masked_fill_(is_pruned | ~batch_mask, -1e9)
        
        _, topk_indices = torch.topk(current_scores, k=max_k, dim=1)
        
        valid_topk_mask = ranks[:, :max_k] < delta_K[:, i].unsqueeze(1)
        
        new_mask = torch.zeros((batch_size, max_m), device=device, dtype=torch.bool)
        new_mask.scatter_(1, topk_indices, valid_topk_mask)
        
        total_loss += (suffix_sq[:, i, :].masked_select(new_mask)).sum()
        is_pruned |= new_mask
        
    return total_loss

def compute_avg_loss(dataset, A):
    final_loss = 0.0
    for data, mask, lens, suffix in dataset:
        final_loss += compute_loss_optimized(A, data, mask, lens, suffix)
    return final_loss / dataset.sample_num

@torch.compile
def compute_loss_population_parallel(A_pop, batch_data, batch_mask, batch_lengths, suffix_sq):
    P, N = A_pop.shape
    B, _, M = batch_data.shape
    device = A_pop.device

    K_cum = torch.round(batch_lengths[None, :, None] * A_pop[:, None, :]).long()
    K_prev = torch.cat([torch.zeros((P, B, 1), device=device, dtype=torch.long), K_cum[:, :, :-1]], dim=2)
    delta_K = (K_cum - K_prev).clamp(min=0) 

    is_pruned = torch.zeros((P, B, M), device=device, dtype=torch.bool)
    total_loss_pop = torch.zeros(P, device=device)

    ranks = torch.arange(M, device=device).view(1, 1, M)

    for i in range(N):
        max_k = int(delta_K[:, :, i].max().item())
        if max_k <= 0: continue

        current_scores = batch_data[:, i, :].unsqueeze(0).expand(P, -1, -1).clone()
        current_scores.masked_fill_(is_pruned | ~batch_mask.unsqueeze(0), -1e9)

        _, topk_indices = torch.topk(current_scores, k=max_k, dim=2) 

        valid_topk_mask = ranks[:, :, :max_k] < delta_K[:, :, i].unsqueeze(2)

        new_mask = torch.zeros((P, B, M), device=device, dtype=torch.bool)
        new_mask.scatter_(2, topk_indices, valid_topk_mask)

        layer_loss = (suffix_sq[:, i, :].unsqueeze(0) * new_mask).sum(dim=(1, 2))
        total_loss_pop += layer_loss

        is_pruned |= new_mask

    return total_loss_pop

def compute_avg_loss_parallel(dataset, A):
    final_loss = 0.0
    for data, mask, lens, suffix in dataset:
        final_loss += compute_loss_population_parallel(A, data, mask, lens, suffix)
    return final_loss / dataset.sample_num


def helper_caculate_c2_div_c1(model_type):
    if model_type == 'llama-3.1-1b':
        return (6 * 2048 * 8192 + 4 * 2048 * 2048 + 4 * 2048 * 64) / (2 * 2048)
    elif model_type == 'llama-3.1-8b':
        return (6 * 4096 * 14336 + 4 * 4096 * 4096 + 4 * 4096 * 128) / (2 * 4096)
    elif model_type == 'qwen-2.5-7b':
        return (6 * 3584 * 18944 + 4 * 3584 * 3584 + 4 * 3584 * 512) / (2 * 3584)
    raise NotImplementedError(f"Model {model_type} not supported")

def compute_g(A_valid, c2_div_c1, token_num):
    R_remaining = 1.0 - A_valid
    layer_num = R_remaining.shape[1]
    gamma = (c2_div_c1 / token_num)
    
    term1 = torch.sum(R_remaining ** 2, dim=1)
    term2 = torch.sum(R_remaining, dim=1)
    
    actual_g = (term1 + gamma * term2) / (layer_num * (1 + gamma))
    return actual_g


def get_ratio_for_our_system(best_x_final):
    x_ratio = torch.clamp(best_x_final, 0.0, 1.0)
    retention_ratios = 1.0 - x_ratio
    return retention_ratios.squeeze()


def optimize_pruning_cmaes2(args, dataset, target_c, n_layers, g_params, device):
    print(f"--- Starting Dynamic Penalty CMA-ES Optimization | Target speedup c={target_c} | Base prompt length = {g_params['token_num']} ---")
    
    with torch.no_grad():
        A_init = torch.linspace(0.1, 0.5, n_layers, device=device).unsqueeze(0)
        base_loss = compute_avg_loss_parallel(dataset, A_init).item()
        
    initial_lambda = base_loss / (0.1 ** 2)
    current_lambda = initial_lambda 
    
    print(f"Initial Base Loss: {base_loss:.4f}, Initial Lambda: {initial_lambda:.2f}")

    def objective_batch(x_pop, c_threshold):
        X_tensor = torch.tensor(np.array(x_pop), dtype=torch.float32, device=device)
        X_ratio = torch.clamp(X_tensor, 0.0, 1.0)
        A_valid = 1.0 - torch.cumprod(1.0 - X_ratio, dim=1)
        
        with torch.no_grad():
            f_vals = compute_avg_loss_parallel(dataset, A_valid)
            g_vals = compute_g(A_valid, **g_params)
            violations = torch.clamp(torch.abs(g_vals - target_c) - c_threshold, min=0.0)
            penalties = current_lambda * (violations ** 2)
            fitness = f_vals + penalties
            
        return fitness.cpu().numpy()

    A_target = np.linspace(0.1, 0.5, n_layers)
    x0 = np.zeros(n_layers)
    x0[0] = A_target[0]
    for i in range(1, n_layers):
        x0[i] = 1.0 - (1.0 - A_target[i]) / (1.0 - A_target[i-1])
        
    sigma0 = 0.2
    opts = cma.CMAOptions()
    opts.set('bounds', [0.0, 1.0])
    opts.set('tolfun', 1e-5)
    opts.set('maxiter', int(args.epochs))
    c_threshold = float(args.ft_limit)
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    iteration = 0
    while not es.stop():
        current_lambda = initial_lambda * (1 + iteration * 0.1)
        solutions = es.ask()
        fitness_values = objective_batch(solutions, c_threshold)
        es.tell(solutions, fitness_values)
        
        if iteration % 10 == 0:
            best_x = torch.tensor(es.result.xbest, dtype=torch.float32, device=device).unsqueeze(0)
            best_X_ratio = torch.clamp(best_x, 0.0, 1.0)
            best_A = 1.0 - torch.cumprod(1.0 - best_X_ratio, dim=1)
            cur_g = compute_g(best_A, **g_params).item()
            print(f"Iteration {iteration}: Lambda={current_lambda:.1f}, Best g={cur_g:.4f}, Loss={es.result.fbest:.4f}")
            
        iteration += 1
        es.disp()
            
    best_x_final = torch.tensor(es.result.xbest, dtype=torch.float32, device=device).unsqueeze(0)
    best_X_ratio_final = torch.clamp(best_x_final, 0.0, 1.0)
    best_A_final = 1.0 - torch.cumprod(1.0 - best_X_ratio_final, dim=1)
    
    with torch.no_grad():
        final_pure_loss = compute_avg_loss_parallel(dataset, best_A_final).item()
    
    print("\n=== Optimization Completed ===")
    print(f"Final Lambda Weight: {current_lambda:.2f}")
    print(f"Final Speedup Ratio: {compute_g(best_A_final, **g_params).item():.4f}")
    print(f"Final Pure Loss: {final_pure_loss:.6f}")
    print(f"Final Pruning Result: {best_A_final.cpu().numpy()}")
    ratios = get_ratio_for_our_system(best_x_final).cpu().numpy().tolist()
    print(f"Retention ratios for inference framework: {ratios}")
    print(f"Total data count: {len(ratios)}")
    
    return ratios, final_pure_loss


def test_speed(dataset, layer_num, device):
    A = torch.linspace(0.1, 0.8, layer_num).to(device)
    for _ in range(5):
        _ = compute_avg_loss(dataset, A)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        _ = compute_avg_loss(dataset, A)
    end_event.record()
    
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event)
    print(f"GPU computation time: {ms / 100 :.2f} ms")

def generate_universal_grid(m_min, m_max, c2_div_c1, N_left=5, N_right=5, alpha=1.5):
    i = np.arange(N_left + 1)
    m_left = c2_div_c1 - (c2_div_c1 - m_min) * (i / N_left) ** alpha
    
    j = np.arange(1, N_right + 1)
    m_right = c2_div_c1 + (m_max - c2_div_c1) * (j / N_right) ** alpha
    
    grid = np.concatenate((m_left[::-1], m_right))
    grid = np.append(grid, [m_min, m_max, c2_div_c1])
    
    return np.unique(grid).astype(int)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default="summed", help="use maxed dataset or summed dataset")
    parser.add_argument('--model_type', type=str, default="llama-3.1-8b")
    parser.add_argument('--epochs', type=int, default=1000, help="number of optimization iterations")
    parser.add_argument('--ft_limit', type=float, default=0.02, help="tolerance between target and actual speedup ratio")
    return parser.parse_args()



if __name__ == "__main__":
    device = 'cuda'
    args = parse_args()
    data_dir_dict = {
        ("llama-3.1-8b", "maxed") : "/data/13_project/llama_8b_result/maxed",
        ("llama-3.1-8b", "summed") : "/data/13_project/llama_8b_result/summed",
        ("qwen-2.5-7b", "summed") : "/data/13_project/qwen2_5_7B_result/summed",
        ("qwen-2.5-7b", "maxed") : "/data/13_project/qwen2_5_7B_result/maxed",
    }
    data_folder = data_dir_dict.get((args.model_type, args.data_type))
    if data_folder is None:
        raise NotImplementedError(f"Invalid model_type {args.model_type} or data_type {args.data_type}")
    
    dataset = OptimizedDataset(data_folder, device, 4)
    layer_num = dataset.batches[0].shape[1]
    c2_div_c1 = helper_caculate_c2_div_c1(args.model_type)
    run_all = True
    num_trials = 3
    
    if run_all:
        all_result = {}
        ratios = [0.5]
        for ratio in ratios:
            ratio = float(ratio)
            best_ratios = None
            best_loss = float('inf')
            for trial in range(num_trials):
                current_ratios, current_loss = optimize_pruning_cmaes2(
                    args, dataset, ratio, layer_num, 
                    {'c2_div_c1': c2_div_c1, 'token_num': dataset.avg_length}, device
                )
                if current_loss < best_loss:
                    print(f"[*] Better result found! Loss decreased from {best_loss:.6f} to {current_loss:.6f}")
                    best_loss = current_loss
                    best_ratios = current_ratios
            
            if ratio not in all_result:
                all_result[ratio] = {}
            all_result[ratio] = best_ratios
            
        os.makedirs("configs", exist_ok=True)
        output_file = f"configs/{args.model_type}_{args.data_type}_ttt.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(all_result, f)
        
    else:
        optimize_pruning_cmaes2(
            args, dataset, 0.3, layer_num, 
            {'c2_div_c1': c2_div_c1, 'token_num': dataset.avg_length}, device
        )