import os
from itertools import product

num_prop_tokens_per_layer = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
graph_sparsity = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
num_neighbours = [1,2,3,4,5,6,7,8,9,10]
alphas = [0.1,0.2,0.3,0.4,0.5]
token_scale = [True, False]

graph_type = ['Mixed'] # ['None', 'Spatial', 'Semantic', 'Mixed']
token_select_strategy = ["MixedAttnMax"] # ['CLSAttnMean', 'CLSAttnMax', 'IMGAttnMean', 'IMGAttnMax', 'DiagAttnMean', 'DiagAttnMax', 'MixedAttnMean', 'MixedAttnMax', 'CosSimMean', 'CosSimMax', 'SumAttnMax','Random', 'None']
propagation_method = ["GraphProp"] # ['None', 'Mean', 'GraphProp']

base_model = "graph_propagation_deit_base_patch16_224"
weight_path = "./weights/deit_base_patch16_224-b5f2ef4d.pth"
data_path = "/mnt/data/imagenet/"

candidate_hyperparams = product(num_prop_tokens_per_layer, graph_sparsity, num_neighbours, alphas, token_scale, graph_type, token_select_strategy, propagation_method)
for set_hyperparam in candidate_hyperparams:
    num_prop = set_hyperparam[0]
    sparsity = set_hyperparam[1]
    num_neighbour = set_hyperparam[2]
    alpha = set_hyperparam[3]
    scale = set_hyperparam[4]
    graph = set_hyperparam[5]
    selection = set_hyperparam[6]
    propagation = set_hyperparam[7]
    cmd = fcmd = f"python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model {base_model} --data-path {data_path} --batch-size 256 --resume {weight_path}  --eval --selection {selection} --propagation {propagation} --sparsity {sparsity} --alpha {alpha} --token_scale {scale} --graph_type {graph} --num_prop {num_prop} --num_neighbours {num_neighbour}"

    os.system(cmd)