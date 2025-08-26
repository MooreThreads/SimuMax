# SimuMax: a static analytical model for LLM distributed training

* [Introduction](#introduction)
* [Getting Started](#Installation)
* [Unitest](#Unitest)
* [TODO](#todo)
* [Acknowledgements](#acknowledgements)
* [Community](#Community)

## Introduction
SimuMax is a distributed training simulator designed for large-scale language model (LLM) workloads. It leverages a static analytical model to simulate and analyze both performance and memory usage, providing detailed insights into training efficiency without running the actual training process. Based on these analyses, SimuMax helps users explore potential ways to maximize computational efficiency.

We have taken into account the current real-world implementations of distributed training, such as Megatron-LM and DeepSpeed, and implemented two key analytical models: **cost model** and **memory model**. By combining these with a roofline model, we simulate and estimate the training process, offering support for various distributed strategies such as tensor parallelism (TP), sequence parallelism (SP), pipeline parallelism (PP), fused kernels, ZeRO 1, recomputation, and more.

It's appropriate to address various use-cases:
1. For user who wants to find an optimal strategy to maximize the training efficiency.
2. For the framework/large model algorithm engineer, it provides optimization directions and debugging analysis.
3. For the chip manufacturer, it provides a tool that can predict performance is required as a reference to assist in the design of various specifications.



### Support features
- [x] Data Parallel
- [x] Tensor Parallel
- [x] 1F1B Pipeline Parallel
- [x] Sequence Parallel
- [x] Expert Parallel
- [x] Zero1
- [x] MoE (only balanced workload)
- [x] Full Recompute
- [x] Selective Recompute
- [x] MLA
- [x] Layer Specification for first/last-stage layer
- [x] Customizable dense layers for MoE
- [x] Megatron Compatibility: Simplified model migration pipeline
- [x] Finer-grained selective recompute
- [x] Efficiency measurement across shapes/layouts


### Benchmarks
Performance of some models on a single node. Llama3-70B was trimmed to 12 layers and DeepSeek-236B was trimmed to 4 layers.
Details can be found in  [FULL_RESULTS](docs/FULL_RESULTS.md) 



#### A100-Pcie
![alt text](assets/A100-Pcie.png)


# Getting Started
## Installation
### Build from source
Users can clone source code and build SimuMax from source:

1. Clone the repository.
```shell
git clone git@github.com:MooreThreads/SimuMax.git
cd SimuMax
```

2. Install the python package.
```shell
pip install -r requirements.txt
pip install -v -e .
```


## Example
Please refer to the [tutorial](./docs/tutorial.md) for more details.

```bash
cd ./examples
python perf_llama3_8b_tp1_pp2.py
# The results are stored in the llama3_8b_a100_pcie_bf16 directory
```
The output is as follows:
```
-------------SIMUMAX SUMMARY TP=1,EP=1,PP=2 -------------
- parallelism = seq4096.mbs1.mbc8.gbs32 tp1.ep1.pp2.dp4.etp1.edp4, world_size:8
- recompute = No Recompute
- dtype = bf16, grad_reduce = fp32
- system = a100_pcie_bf16
- model = dense
- mfu = 0.49
- TFLOPS = 151.59 (tflops=843.3426 T, duration=5.5632 s)
- TGS_per_gpu = 2945.052828675417
- peak_alloc_mem = {'first_stage': '50.7760 GB', 'last_stage': '45.1637 GB'}
```

# Unitest
after clone the repo, plz use "git config core.hooksPath git_hooks" to set commit hook, which will check the perf result in some config is still the same before commit automatically.

# TODO
SimuMax is in active development, may contain bugs or incomplete features. Contributions are welcome!
There are features to be added. Several significants are:
- Support context parallel
- More pipeline scheduler
- Overlap between computation and communication
- Offloading
- Strategy search
- More accurate memory-bound operator simulation


# Acknowledgements
Some aspects of the design and interfaces were inspired by [Calculon](https://github.com/calculon-ai/calculon). We appreciate the work done by the authors of that repository, which provided helpful references during development.


# Community

### Issue Reporting
If you find any problems for SimuMax, please open an issue.

### Contributions
Welcome any form of contribution of code, model implementation and document!


### Join Our Team
If you're passionate about:

Large-scale models for MoE, Reinforcement Learning, Multi-Modal
GPU/GPU-Cluster Training/Inference performance optimization

Feel free to reach out to xuerong.huang@mthreads.com.