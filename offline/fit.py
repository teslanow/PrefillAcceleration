import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pickle

def parse_csv(file_path):
    seq_len_list = []
    top_k_ratio_list = []
    avg_ms_list = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row: continue
            seq_len_list.append(int(row[0]))
            top_k_ratio_list.append(float(row[1]))
            avg_ms_list.append(float(row[2]))
    return np.array(seq_len_list), np.array(top_k_ratio_list), np.array(avg_ms_list)

def get_pred_strategy_0(L_data, r_data, C):
    c0, c1, c2, c3 = C
    return c0 + c1 * L_data + c2 * L_data * r_data + c3 * (L_data**2) * (r_data**2)

def get_pred_strategy_1(L_data, r_data, C):
    c0, c1, c2, c3 = C
    return c0 + c1 * L_data + c2 * L_data * r_data + c3 * (L_data**2)

def check_strategy_consistency(file_path_0, file_path_1, C0, C1):
    def load_data(path):
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row: data.append([float(x) for x in row])
        return np.array(data)

    data0 = load_data(file_path_0)
    data1 = load_data(file_path_1)

    L = data0[:, 0]
    r = data0[:, 1]
    y0_true = data0[:, 2]
    y1_true = data1[:, 2]

    y0_pred = C0[0] + C0[1]*L + C0[2]*L*r + C0[3]*(L**2)*(r**2)
    y1_pred = C1[0] + C1[1]*L + C1[2]*L*r + C1[3]*(L**2)

    actual_0_faster = y0_true < y1_true
    pred_0_faster = y0_pred < y1_pred

    consistent_mask = (actual_0_faster == pred_0_faster)
    inconsistent_indices = np.where(~consistent_mask)[0]
    
    total = len(consistent_mask)
    mismatches = len(inconsistent_indices)
    accuracy = (total - mismatches) / total * 100

    print("-" * 40)
    print(f"Strategy Comparison Report (Strategy 0 vs Strategy 1)")
    print(f"Total test samples: {total}")
    print(f"Consistent predictions: {total - mismatches}")
    print(f"Inconsistent predictions (Decision errors): {mismatches}")
    print(f"Directional Accuracy: {accuracy:.2f}%")
    print("-" * 40)

    if mismatches > 0:
        print("Sample inconsistent points (L, r, S0_true, S1_true):")
        for i in inconsistent_indices[:5]:
            print(f"L={L[i]:<6} r={r[i]:<6} S0_true={y0_true[i]:.2f}ms S1_true={y1_true[i]:.2f}ms")

    return mismatches, accuracy

def get_pred_strategy_2(L_data, r_data, C):
    c0, c1, c2 = C 
    return c0 + c1 * L_data + c2 * (L_data**2)

def fit_strategy_0(seq_len_data, top_k_ratio_data, avg_ms_data):
    N = len(seq_len_data)
    X = np.column_stack([
        np.ones(N),
        seq_len_data,
        seq_len_data * top_k_ratio_data,
        (seq_len_data ** 2) * (top_k_ratio_data ** 2)
    ])
    C, _, _, _ = np.linalg.lstsq(X, avg_ms_data, rcond=None)
    print(f"Strategy 0 Parameters: c0={C[0]}, c1={C[1]}, c2={C[2]}, c3={C[3]}")
    return C

def fit_strategy_1(seq_len_data, top_k_ratio_data, avg_ms_data):
    N = len(seq_len_data)
    X = np.column_stack([
        np.ones(N),
        seq_len_data,
        seq_len_data * top_k_ratio_data,
        seq_len_data ** 2
    ])
    C, _, _, _ = np.linalg.lstsq(X, avg_ms_data, rcond=None)
    print(f"Strategy 1 Parameters: c0={C[0]}, c1={C[1]}, c2={C[2]}, c3={C[3]}")
    return C

def fit_strategy_2(seq_len_data, top_k_ratio_data, avg_ms_data):
    N = len(seq_len_data)
    X = np.column_stack([
        np.ones(N),
        seq_len_data,
        seq_len_data ** 2
    ])
    C, _, _, _ = np.linalg.lstsq(X, avg_ms_data, rcond=None)
    print(f"Strategy 2 Parameters: c0={C[0]}, c1={C[1]}, c2={C[2]}")
    return C

def plot_fit_results(L_data, r_data, y_true, C, strategy):
    if strategy == "strategy_0":
        y_pred = get_pred_strategy_0(L_data, r_data, C)
    elif strategy == "strategy_1":
        y_pred = get_pred_strategy_1(L_data, r_data, C)
    elif strategy == "strategy_2":
        y_pred = get_pred_strategy_2(L_data, r_data, C)
    
    relative_errors = np.abs((y_true - y_pred) / y_true)
    mre = np.mean(relative_errors) * 100
    print(f">>> {strategy} Mean Relative Error (MRE): {mre:.2f}%")

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(y_true, y_pred, color='steelblue', alpha=0.7, edgecolors='k')
    line_range = [y_true.min(), y_true.max()]
    ax1.plot(line_range, line_range, 'r--', linewidth=2)
    ax1.set_xlabel('Measured Time (ms)')
    ax1.set_ylabel('Predicted Time (ms)')
    ax1.set_title(f'{strategy}: True vs Predicted')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(L_data, r_data, y_true, color='blue', s=20)
    L_mesh = np.linspace(L_data.min(), L_data.max(), 40)
    r_mesh = np.linspace(r_data.min(), r_data.max(), 40)
    L_grid, r_grid = np.meshgrid(L_mesh, r_mesh)
    
    if strategy == "strategy_0":
        f_grid = get_pred_strategy_0(L_grid, r_grid, C)
    elif strategy == "strategy_1":
        f_grid = get_pred_strategy_1(L_grid, r_grid, C)
    elif strategy == "strategy_2":
        f_grid = get_pred_strategy_2(L_grid, r_grid, C)
    
    ax2.plot_surface(L_grid, r_grid, f_grid, cmap='viridis', alpha=0.4)
    ax2.set_xlabel('L')
    ax2.set_ylabel('r')
    ax2.set_zlabel('Time (ms)')
    
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figures/{strategy}.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    strategies = ["strategy_0", "strategy_1", "strategy_2"]
    all_results = {}
    model = "qwen"
    base_dir = f"./results/strategy_speed/{model}"
    
    for s in strategies:
        file_path = os.path.join(base_dir, f"{s}.csv")
        try:
            L, r, y = parse_csv(file_path)
            if s == "strategy_0":
                C = fit_strategy_0(L, r, y)
                all_results["strategy_0"] = (C[0], C[1], C[2], C[3])
            elif s == "strategy_1":
                C = fit_strategy_1(L, r, y)
                all_results["strategy_1"] = (C[0], C[1], C[2], C[3])
            elif s == "strategy_2":
                C = fit_strategy_2(L, r, y)
                all_results["strategy_2"] = (C[0], C[1], C[2])
            plot_fit_results(L, r, y, C, s)
        except FileNotFoundError:
            print(f"Skipping {s}: File not found at {file_path}")
            
    os.makedirs("configs", exist_ok=True)
    with open(f"configs/{model}.pkl", 'wb') as f:
        pickle.dump(all_results, f)