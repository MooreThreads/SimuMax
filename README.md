# SimuMax: a static analytical model for LLM distributed training

* [Introduction](#introduction)
* [Installation](#installation)
* [Usage](#usage)
* [TODO](#todo)
* [Acknowledgements](#acknowledgements)

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
- [x] Recompute

## Installation
#### Build from source
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


## Usage

### Example
```python
# Define the system、strategy、model config
system_config_file = ...
strategy_config_file = ...
model_config_file = ...
# Setup perf model and config
perf_model = PerfLLM()
perf_model.configure(
    strategy_config=strategy_config_file, 
    model_config=model_config_file, 
    system_config=system_config_file
)

# Run simulate
perf_model.run_estimate()

# Based simulate result, run memory analysis
mem_result = perf_model.analysis_mem()

# Based simulate result, run cost analysis
cost_result = perf_model.analysis_cost()
```
In the above example, `system_config_file`, `strategy_config_file`, and `model_config_file` are paths to your configuration files.

The run_estimate method simulates the training process and estimates the performance.

The analysis_mem method analyzes the memory usage during the training process and returns a mem_result object. This object contains information about the memory usage of different parts of the model, such as the weight memory usage, gradient memory usage, and state memory usage.

The analysis_cost method analyzes the cost of the training process and returns a cost_result object. This object contains information about the compute usage of the model, such as the forward pass flops, backward pass flops, and memory accessed during each pass.

We also provide some examples of models. You can try the scripts in the examples directory. 
Note that performance analysis depends heavily on the system config, so an accurate config is important. We only provide a demo, not an accurate config.

```bash
cd ./examples
python perf_llama2-7b_4node.py
# The results are stored in the tmp directory
```


### Result Field Explanations
Here are explanations for each field in the `cost_result`:
- `comm_result`: each batch's communication cost. Currently, we assume that the communication cost of each batch is the same, and we will adjust it later.
- `compute_details`: each batch's compute cost details and whole training process's statistics.
- `breakdown_result`:  a dictionary that contains the breakdown of the cost of the training process.
- `chunk_time`: the time taken for each micro batch forward and backward pass.
- `all_tokens_per_iter`: the number of tokens processed per iteration.
- `duration_time_per_iter`: the time taken for each iteration.
- `mfu`:  simulated flops are used to calculate the MFU.
- `mfu_6nd_with_attn`: the 6ND MFU formula with attention which typically doesn't differ too much from `mfu`.
- `mfu_6nd`: the 6ND MFU formula without attention.
- `throughput_per_accelerator`: the throughput of each accelerator during the training process.

And here are explanations for each field in the `mem_result`(if pp_size is 1, the result is the memory analysis result of the first stage, otherwise, it will return the memory analysis result of the first stage and the last stage respectively.):
- `model_mem`: the memory usage of the model. including the weight memory usage, gradient memory usage, and state memory usage.
- `fwd_peak_allocated_mem`: the peak memory usage during the forward pass.
- `bwd_peak_allocated_mem`: the peak memory usage during the backward pass.
- `peak_cached_mem`: the peak memory usage of the cache.

## TODO
SimuMax is in active development, may contain bugs or incomplete features. Contributions are welcome!
There are features to be added. Several significants are:
- Support context parallel
- More pipeline scheduler
- Overlap between computation and communication
- Offloading
- FP8 training
- More accurate calculation/communication operator simulation

## Acknowledgements
Some aspects of the design and interfaces were inspired by [Calculon](https://github.com/calculon-ai/calculon). We appreciate the work done by the authors of that repository, which provided helpful references during development.