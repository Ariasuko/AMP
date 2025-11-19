# Script to reproduce results with specific settings

import time
import os
import numpy as np
import torch

from sa import amp_no_placement_strategy
from amp_utils import simulate, to_float_torch
from amp_config import AMP_Config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Whether to run real trials")
parser.add_argument("--budget", type=int, default=-1, help="how many real trials to launch")
parser.add_argument("--experiment", type=int, default=None, help="which experiment to reproduce")

args = parser.parse_args()

if not 1 <= args.experiment <= 3:
    print(f"Experiment number must be between 1 and 3:{args.experiment}")
    exit(1)

time_s = time.time()

# number of GPU per node, number of nodes
M = AMP_Config["AMP"]["GPU_PER_NODE"]
N = AMP_Config["AMP"]["NUMBER_OF_NODES"]

# inter-node bandwidth, intra-node bandwidth
cluster_info = {}
for i in range(N):
    cluster_info[i] = [bw * 1e9 / 32 for bw in AMP_Config["SERVER"]["BANDWIDTH"][i]]

if args.experiment == 3:
    from cost_het_cluster import AMP

    exp_name = f"het_model"
    depth = [12, 0, 0, 0, 0, 12]
    model_config = {"hidden_size": torch.tensor([1024]).float(),
                    "sequence_length": torch.tensor([1024]).float(),
                    "num_layers": torch.tensor([sum(depth)]).float(),
                    "vocab_size": torch.tensor([52256]).float(),
                    "type": "transgan",
                    "depth": depth,
                    "bottom": 9}

else:
    model_config = {"hidden_size": torch.tensor([1024]).float(),
                    "sequence_length": torch.tensor([1024]).float(),
                    "num_layers": torch.tensor([24]).float(),
                    "vocab_size": torch.tensor([52256]).float(),
                    "type": "gpt2"}
    if args.experiment == 1:
        from cost_homo import AMP

        exp_name = f"homogeneous"
    else:
        from cost_het_cluster import AMP

        exp_name = f"het_cluster"

main_logs_path = os.path.expanduser(AMP_Config["AMP"]["MAIN_LOGS_PATH"])
if not os.path.exists(main_logs_path):
    os.mkdir(main_logs_path)
time_stamp = int(time_s)
record_file = f"{os.path.join(main_logs_path, exp_name)}_{time_stamp}.txt"

global_bs = AMP_Config["AMP"]["GLOBAL_BS"]
model = AMP(model_config, exp_name)
assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irregular"

want_simulate = []
feasible = {}

with open(record_file, "a") as fp:
    fp.write(f"{model_config}\n")
    fp.write(f"gbs:{global_bs}\n")
known = None
iter_count = 0

# Estimating best configurations
while True:
    ret = amp_no_placement_strategy(M=M, N=N, gbs=global_bs, known=known)
    if ret is None:
        break
    else:
        h, w, mbs, known = ret
        oth = {"mp_deg": torch.ones(1, ) * h, "dp_deg": torch.ones(1, ) * w,
               "pp_deg": torch.ones(1, ) * (M * N / (h * w))}
        fake_config = np.ones((M, N)) * (-1)
        model_args = (fake_config, global_bs, mbs, cluster_info, model_config, oth)

        rank_map, partition, cost = model(model_args)

        want_simulate.append(((mbs, oth, rank_map, partition), cost))
    iter_count += 1
    if iter_count % 10 == 0:
        print(f"AMP finishes {iter_count} iterations")
time_e = time.time()
print(f"AMP finishes without placement in {iter_count} iterations in {time_e - time_s}")

sorted_settings = sorted(want_simulate, key=lambda kv: kv[1])
with open(record_file, "a") as fp:
    for index, ((mbs, oth, rank_map, partition), cost) in enumerate(sorted_settings):
        fp.write(
            f"[rank {index}]: MBS={mbs} | mp={int(oth['mp_deg'].item())} dp={int(oth['dp_deg'].item())} pp={int(oth['pp_deg'].item())} | rank_map={rank_map} partition={partition} | Cost={cost.item()}\n")

# Run real trials to get ground truth runtime
if args.full:
    if args.budget == -1:
        budget = len(sorted_settings)
    else:
        budget = args.budget
    simulate_start = time.time()
    for i in range(budget):
        mbs, oth, _, partition = sorted_settings[i][0]
        rmap = None
        gt_cost = simulate(partition, global_bs, mbs, model_config, [oth], exp_name)
        gt_cost = gt_cost[0]
        with open(record_file, "a") as fp:
            fp.write(
                f"Simulating result: {rmap}, {partition}, {mbs}, {oth}, with p_cost: {sorted_settings[i][1]}, r_cost: {gt_cost} \n")
            fp.write(f"running real trials till iter {i} takes {time.time() - time_s} \n")
