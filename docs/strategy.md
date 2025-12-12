<p align="center">
  <a href="strategy.md">English</a>| 
  <a href="strategy-zh.md">中文版本</a> 
</p>


# Introduction
SimuMax relies on three core input files: system, strategy, and model. The strategy file defines the training parallel strategies such as TP/PP/EP, cluster GPU count, batch size, recomputation strategies, etc.

# Parameter Description
## Basic Training Parameters
### seq_len
Sequence length (number of tokens)
### micro_batch_size
Micro-batch size (number of samples processed per forward propagation pass)
### micro_batch_num
Number of micro-batches for gradient accumulation
### dtype
Computation data type (bf16 indicates half-precision floating-point), default is bf16
### fp8
Whether to use fp8 mixed precision training, default is false

## Distributed Strategy
### world_size
Total number of GPUs (default is 8)
### tp_size
Tensor Parallelism size, default is 1
### pp_size
Pipeline Parallelism size - vertically splits model layers, default is 1
### ep_size
Expert Parallelism size - used for MOE models, default is 1
### etp_size
Expert Tensor Parallelism size, default is 1
### moe_dispatcher_policy
Routing strategy for MOE models, default is "all2all"
### enable_sequence_parallel
Whether to enable sequence parallelism, default is true, effective when tp_size > 1
### num_layers_in_first_pipeline_stage & num_layers_in_last_pipeline_stage
Controls the number of layers contained in the first and last Pipeline Parallel stages, default is None
### interleaving_size
Reserved field, currently not used
### zero_state
ZeRO optimization configuration, currently only supports zero0 and zero1, default is 1

## Memory Optimization
### grad_reduce_in_bf16
Whether to use bf16 for gradient reduction, default is false
### use_accm_weight
Whether to use weight accumulation fusion (reduces temporary variables), default is true
### cache_groupgemm_col_fp8_inputs
Whether to cache FP8 inputs for groupgemm, default is false
### offload_groupgemm_col_inputs
Whether to offload groupgemm inputs to CPU, default is false

## Recompute Related
#### enable_recompute
Global switch for recompute, default is true
#### recompute_granularity
Granularity of recompute, options are "full_block" and "selective_recompute", default is None
#### recompute_layer_num
Number of layers for recompute, default is 0
#### attn_recompute
Recompute for attention module, default is false
#### mla_rms_recompute
Recompute for mla's rmsnorm and q/k up-projection, default is false
#### mlp_recompute
Recompute for MLP and groupedgemm, default is false
#### mlp_rms_recompute
Recompute for rmsnorm+router+sharedExpert, default is false
#### recompute_variance
Whether to remove redundant forward computation for the last module in recompute checkpoint, default is `false`.
When recompute_granularity is "`selective_recompute`", it is recommended to set this to `true` to save computation time.
## Computation Optimization
### attention_sparse_ratio
Attention sparse ratio (0.0 indicates dense attention), default is 0.0
### use_flash_sdp
Use FlashAttention acceleration
### use_fused_*
Various fused kernel optimizations
### enable_dropout
Whether to enable Dropout regularization, default is false

## Network Strategy
### tp_net, pp_net, dp_net, etc.
Network communication strategies for various parallel dimensions, default is "auto", automatically selected based on cluster scale and parallel strategy

## Other
### dispatch_probs
Megatron-LM related parameter (whether to dispatch probs), Megatron-LM version 0.14 requires this to be set to true
### mem_factor
Memory usage coefficient (0.94 means reserving 6% margin), used to estimate reserve_memory (=max_memory / mem_factor), default is 0.94