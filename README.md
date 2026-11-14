This is the official implementation of NeurIPS 2026 paper "Accelerating Long-Context LLM Prefill via Layer-wise Progressive Token Pruning in Local Deployment"
# Offline
## Get the pruning schedule
- `cd offline`
- get score: `python XX_score_achieve.py --mode xx --data_dir xx --output_dir xx`
- move to the dir: `./move.sh`, set the directory path in the script
- run `../eval/run_training_with_cab_set2.sh`

## Get the fit parameter
- `python collect_methods_speed.py` to get speed for each length and pruning ratio
- `python fit.py` to generate the fit coefficients

# Online
## Quality experiments
- run `./eval/run_quality_longbench.sh 0.5 0` to achieve longbench result: 0.5 means expected speedup, 0 means use cuda:0 
- run `./eval/run_quality_ruler.sh 0 0.5 4096` to achieve ruler result: 0 means using cuda:0, 0.5 means expected speedup, 4096 means the ruler length tested

## Efficiency experiments
- run `./eval/run_latency_baseline_origin.sh` to achieve latency results 