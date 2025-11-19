## This repo is forked in order to reproduce in servers other than AWS with/without using AMI ID.

original repo is [here](https://github.com/DachengLi1/AMP).

## Main modifications

### all-in-one.py to run

Using src/all-in-one.py, you can run experiment 1-3 with option `--experiment [1,2,3]`, along with original options `--full` and `-budget X`.
This script will handle different parameters each experiment requires, and keep modified code(e.g. how to represent in log.txt) the same across all experiments.

### config.toml & de-hardcoding parameters

One problem I have encountered during reproduction is that username, private key pate, path of log folder and so on, are not always the same as the script expected if you choose to use servers other than AWS.

In this case, along with all-in-one.py mentioned above, I make `src/config.toml` and `src/amp_config.py` for the convenience, and moves hardcoded parameters as many as I can to config.toml.(except `vocab.json` path and so on in `DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_pipe.sh`)

However, there are a few points here you may take note of:

* First of all, you may need to rename `src/config.toml.template` to `config.toml` before using `all-in-one.py`.
* The original scripts(homogeneous.py, het-cluster.py and het-model.py) won't make use of config.toml since I have not modified them.
* All parameters are kept the original except `GLOBAL_BS`(32 for exp 1&2, 64 for 3), in case you'd like to use AWS and AMI ID.
* It is not guaranteed that changes to `GPU_PER_NODE` will take effect, as it seems this parameter is hardcoded into dependency Megatron-Deepspeed.

### re-factor class AMP

There is a guess that the original idea is to use AI or somewhat as class AMP that solves partition problem inherits `torch.nn.Module` without truly using Neural Network training or inference stuff.

Anyway, there is no need to do so if not use it, so I remove the inheritance, which results in that predict step of experiment 1 takes around 300s now(500s before).

## Modifications still under working

There are still some things I would like to do ,if I'm not too drained by then...

### figure out wheel version

Another problem is that nowadays, many methods that package provided are different, or incompatible with newer version.
However, package info original repo mentioned is not comprehensive enough if choose to set up.

Following the instruction provided below will still result in unsatisfied requirements and some workarounds(e.g. modify site-packages' code) are needed, which I think it's not elegant.

In short, determine if it's possible to run the script using the newer version with as few workarounds as possible. Eventually, try to provide a full `requirements.txt` or file.

### switch backend engine

"Backend engine"(or "frame") here refers to the Megatron-Deepspeed([original repo for reference](https://github.com/deepspeedai/DeepSpeedExamples/tree/20ea07a2a069696abec212e25476a9bf76aced70/Megatron-LM-v1.1.5-3D_parallelism)), which enables 3D-parallel training.

At the time of writing this paper, it was the perfect choice, but there are better options now. Colossal-AI and Megatron-LM can now do 3D parallel training. Furthermore, the Megatron-LM itself can now specify how each stage is assigned to each GPU without making many modifications to the package code.

So, it is considerable to make an adapter to make use of whatever backend engine you prefer.


## Below is original README.md of repo. Note that since having modified the both the code and engine that the code uses, there is no guarantee that result will remain exact the same that mentioned in paper.

---

# AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness (NeurIPS 2022) 
[**Paper**](https://arxiv.org/pdf/2210.07297.pdf) | 
[**Usage**](#usage) |
[**Citation**](#citation) |
[**Presentation**](https://recorder-v3.slideslive.com/?share=74667&s=aa9ce793-0697-43bc-9d8f-f7b139471f95) 

This repository contains the official code for our NeurIPS 2022 paper **AMP**. AMP is an **automatic** approach to find fast model-parallel strategies to train large Deep Learning models. We design AMP to tackle real-world scnerios where users are training **hetergeneous** models with uneven layers and **hetergeneous** cluster with mixed generations of GPUs. Concretely, it contributes
- A valid **representation** of model-parallelism strategies.
- A **cost model** that accurately predicts the running time of a strategy without launching expensive real trials.
- An **automatic optimization** procedure that uses the cost model and a dynamic programming algorithm to efficiently find fast strategies.

<img src="figures/workflow.png" width="600">

## Performance 
AMP finds strategies that have similar performance to the state-of-the-art strategy finder[[1]](#1) when no heterogeneity in the model and in the cluster. AMP fins strategies that are **1.54x** better than the SOTA when heterogeneity exists in the cluster, and **1.77x** better when heterogeneity exists in the model. In particular, the cost model in AMP can accurately predict low costs for top strategies. 

<img src="figures/speedup.PNG" width="600"> <img src="figures/cost_vs_real.png" width="600" >

## Usage
We provide two settings: (1) use AMP to predict top strategies, (2) Additionally launch real trials with DeepSpeed to validate the ground truth runtime. Setting 1 requires a single CPU, while Setting 2 requires 16 GPUs in AWS EC2 (we provide the instance details in the paper). We have installed the environment and prepare necessary intermediate results for Setting 2 in an AMI for ease of setup.

#### Set up environment for setting 1
````
cd ~
git clone https://github.com/MccRee17/AMP
conda create -n amp python=3.7.3
conda activate amp
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests
pip install tqdm spur torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
````

#### Set up environment for setting 2
Use our AWS AMI:

| AMI Name       | AMI ID                | Region    | Instances in the paper               |
|----------------|-----------------------|-----------|--------------------------------------|
| 	AMP-Oct-31   | ami-011d5dd7f6fe79d32 | us-west-2 | G4dn.12xlarge, P3.8xlarge            |

Launch instances specified in the paper, e.g. 4 G4dn.12xlarge instances for the homogeneous experiment. Lauch them within the **same** placement group using the **cluster** strategy in EC2 so that they have the maximum bandwidth. Assume that AWS assigns 4 machines with IP address IP1, IP2, IP3, and IP4. Then do several steps on the master Machine IP1:
- ssh into IP[1-4] and exit to store the public ssh key of all machines in IP1, so the ssh verification does not prompt during trials.
- Add IP[1-4] to ``~/hostfile`` and state the number of GPUs in each machine ([DeepSpeed Tutorial](https://www.deepspeed.ai/getting-started/)). For instance, all 4x4 clusters in the paper are specified by: 
````
IP1 slots=4
IP2 slots=4
IP3 slots=4
IP4 slots=4
````
- Activate our environment by ``source anaconda3/bin/activate; conda activate amp``.

Suggestions: (1) Warm up **each** AWS machine before running, otherwise trials may get terminated by timeout. A simple warmup is ``python; import torch; a = torch.randn(100,100).cuda()`` (2) If some trials hang, one can manually login to each machine and kill GPU processes. The optimization algorithms runs on CPU and will not be affected. A useful command to check processes on GPU: ````sudo fuser -v /dev/nvidia*````. (3) If processes constantly get stuck, try removing all caches by ``rm -rf ~/amp_simulate; rm -rf ~/tmp``. If there are other blockers in launching distributed experiments, please leave an issue here or send the author an [email](dacheng2@andrew.cmu.edu).

### Experiment 1: Homogeneous
With Setting 1:
````
cd ~/AMP/src
python homogeneous.py 
````
This will finish in around 500 seconds and store the result in ~/amp_main_logs/homogeneous_[time_stamp].txt.

With Setting 2:
```` 
cd ~/AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism
python homogeneous.py --full --budget 10
````
This will run the prediction and launch top 10 predicted strategies. It will finish in around 1500 seconds and store the result in ~/amp_main_logs/homogeneous_[time_stamp].txt. To run x numbers of real trials, use argument ````--budget x````. The raw log from our modified DeepSpeed contains a lot of details such as the pipeline schedule, we recommend redirecting it into another log.txt for further interpretation.

Cached results with 53 real trials are in AMP/src/results/homogeneous_results.txt and logs in AMP/src/results/homogeneous_log.txt.

### Experiment 2: Hetergeneous cluster
With Setting 1:
````
cd ~/AMP/src
python het_cluster.py 
````
This will finish in around 500 seconds and store the result in ~/amp_main_logs/het_cluster_[time_stamp].txt.

With Setting 2:
```` 
cd ~/AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism
python het_cluster.py --full --budget 10
````
Predicting and 10 real trials takes around 1600 seconds. Cached results with 53 real trials are in AMP/src/results/het_cluster_results.txt and logs in AMP/src/results/het_cluster_log.txt.

### Experiment 3: Hetergeneous model
With Setting 1:
````
cd ~/AMP/src
python het_model.py 
````
This will finish in around 200 seconds and store the result in ~/amp_main_logs/het_model_[time_stamp].txt.

With Setting 2:
```` 
cd ~/AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism
python het_model.py --full --budget 10
````
Predicting and 10 real trials takes around 1300 seconds. Cached results with 65 real trials are in AMP/src/results/het_model_results.txt and logs in AMP/src/results/het_model_log.txt.

## Code Logic
Basic logic of AMP is implemented in several files:
- The main function (homogeneous.py, het_cluster.py, het_model.py) iteratively applies the cost model and optimize. 
- cost_xxx.py implements the cost model.
- pipe.py implements the dynamic programming algorithm.
- sa.py provides possible candidates for the main function. 
- amp_utils.py implements other functions such as launching real trials with given configurations.

## Citation
If you find this repository useful, please cite our paper using
````
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
````
## References
<a id="1">[1]</a> 
Narayanan, Deepak, et al. "Efficient large-scale language model training on gpu clusters using megatron-lm." Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2021.
