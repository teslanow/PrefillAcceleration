
#!/bin/bash

data_types=("summed")
model_types=("llama-3.1-8b")
# model_types=("qwen-2.5-7b")

for data_type in "${data_types[@]}"
do
    for model_type in "${model_types[@]}"
    do
        echo "Running: data_type=${data_type}, model_type=${model_type}"
        CUDA_VISIBLE_DEVICES=6 python offline/training_with_cab_set2.py --data_type ${data_type} --model_type ${model_type} --epochs 2000
    done
done

