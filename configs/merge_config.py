file_list = ["configs/llama-3.1-8b_summed.pkl", "configs/llama-3.1-8b_summed_ttt.pkl"]
# file_list = ["configs/qwen-2.5-7b_summed.pkl", "configs/qwen-2.5-7b_summed_ttt.pkl"]
all_results = {

}
import pickle
for file in file_list:
    with open(file, 'rb') as f:
        format_data = pickle.load(f)
        for key, value in format_data.items():
            all_results[key] = value
with open("configs/merge.pkl", "wb") as f:
    pickle.dump(all_results, f)