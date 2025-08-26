
# Guilded Tutorial
## Simple Example

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
python perf_llama3_8b_tp1_pp2.py
# The results are stored in the llama3_8b_a100_pcie_bf16 directory
```

Here are explanations for each field in the `cost_result`:
- `comm_result`: each batch's communication cost. Currently, we assume that the communication cost of each batch is the same, and we will adjust it later.
- `compute_details`: each batch's compute cost details and whole training process's statistics.
- `breakdown_result`:  a dictionary that contains the breakdown of the cost of the training process.
- `chunk_time`: the time taken for each micro batch forward and backward pass.
- `all_tokens_per_iter`: the number of tokens processed per iteration.
- `duration_time_per_iter`: the time taken for each iteration.
- `mfu_6nd_with_attn`: <b>It is the standard mfu, please refer to this value instead of "mfu"</b>. The 6ND MFU formula with attention which typically doesn't differ too much from `mfu`.
- `mfu`:  simulated flops are used to calculate the MFU.
    - `mfu` is not always same as `mfu_6nd_with_attn`, while it considers some extra op, such as a extra gemm in FA. 
- `throughput_per_accelerator`: the throughput of each accelerator during the training process.

And here are explanations for each field in the `mem_result`(if pp_size is 1, the result is the memory analysis result of the first stage, otherwise, it will return the memory analysis result of the first stage and the last stage respectively.):
- `model_mem`: the memory usage of the model. including the weight memory usage, gradient memory usage, and state memory usage.
- `fwd_activation_cache_per_micro_batch`: the memory usage of the activation cached for the backward during the forward pass for each micro batch.
- `peak_activation_mem_in_1F1B`: the peak memory usage of the activation during the forward and backward pass.
- `fwd_peak_allocated_mem`: the peak memory usage during the forward pass.
- `bwd_peak_allocated_mem`: the peak memory usage during the backward pass.
- `peak_mem`: the peak memory usage of the cache.
- `peak_mem_with_reserved`: the peak memory usage of the cache with reserved memory.
- `memory_reserved_ratio`: the ratio of the reserved memory to the total allocated memory.  
- `peak_path`: the peak memory usage path.
## Notes
- Currently, all Linear models are forced to perform gradient accumulation fusion.
## Features
### Set pp layers of the first and last stage
If the pipeline parallelism is used(pp_size > 1), the first and last stage can be set by the following variables in the strategy file(e.g. tp1_pp2_ep4_mbs1_mbc1.json). The layers of the middle stages are averaged based on the remaining layers.
```json
{
    "seq_len": 4096,
    "micro_batch_size": 1,
    "world_size": 8,
    "tp_size": 1,
    "pp_size": 4,
    "ep_size": 2,
    ...
    "num_layers_in_first_pipeline_stage": 9,
    "num_layers_in_last_pipeline_stage": 11,
}
```
In the above strategy file, the first stage contains 9 layers and the last stage contains 11 layers, if the layers of the model are 50, the layers of each pp stage are [9, 15, 15, 11].


###  Recompute

Set the follwing variables in the strategy file(e.g. tp1_pp2_ep4_mbs1_mbc1_selective_recompute_fp8.json) to enable selective recompute and fp8 mixed-precision training. Note that the llama only support mlp_recompute and mlp_rms_recompute.
- Full recompute  example:
```json
{
    "seq_len": 4096,
    "micro_batch_size": 1,
    "world_size": 8,
    "tp_size": 1,
    "pp_size": 1,
    "ep_size": 8,
    ...
    "recompute_granularity" : "full_block", // enable full recompute  
    "recompute_layer_num": 1, // the number of recomputed layers, 1 means the first layer is enabled to full-recompute.
}
```

- Selective recompute example:
```json
{
    "seq_len": 4096,
    "micro_batch_size": 1,
    "world_size": 8,
    "tp_size": 1,
    "pp_size": 1,
    "ep_size": 8,
    ...
    "recompute_granularity" : "selective_recompute", // enable selective recompute  
    "recompute_layer_num": 1, // the number of recomputed layers, 1 means the first layer is enabled to slelective recompute.
    "attn_recompute":true, // enable recompute for attention
    "mla_rms_recompute":true,   // enable recompute for rms before mla
    "mlp_recompute":true, // enable recompute for mlp
    "mlp_rms_recompute":true, // enable recompute for rms before mlp
}
```

Then please refer to the chapter "Example" to start the esimulation. We also provide some examples of models. You can try the scripts in the this directory.
Note that performance analysis depends heavily on the system config, so an accurate config is important. We only provide a demo, not an accurate config. 
<!-- The full example of recompute and fp8 is there: [perf_deepseek_1node_tp1pp2ep4_selective_recompute_fp8.py](perf_deepseek_1node_tp1pp2ep4_selective_recompute_fp8.py), [perf_llama_1node_tp4pp2ep1_selective_recompute_fp8.py](perf_llama_1node_tp4pp2ep1_selective_recompute_fp8.py) -->
```shell
cd ./examples
python perf_deepseekv2_layer4_ep4_pp2_selective_recompute.py # the result is saved in deepseek_v2_a100_pcie_bf16 directory
python perf_llama3_70b_layer12_tp2_full_recompute.py  # the result is saved in deepseek_v2_a100_pcie_bf16 directory
```




