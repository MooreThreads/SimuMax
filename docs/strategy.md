<p align="center">
  <a href="strategy.md">English</a>| 
  <a href="strategy-zh.md">中文版本</a> 
</p>


# Strategy Config

SimuMax relies on three core input files: `system`, `strategy`, and `model`.
The `strategy` file defines training-side runtime choices such as TP / PP / EP, world size, batching, recompute, and VPP-related settings.

See also:

- [docs/README.md](./README.md)
- [model.md](./model.md)
- [system.md](./system.md)

The strategy file is where SimuMax most directly mirrors Megatron runtime choices.
If a real run and a strategy file disagree on PP / EP / TP, sequence parallelism, recompute, or VPP-related settings, both timing and memory can drift.

## Fastest way to start

Do not start from an empty file unless you have to.

Recommended path:

1. Copy the nearest existing JSON from [configs/strategy](../configs/strategy).
2. Keep `seq_len`, `micro_batch_size`, and `micro_batch_num` simple first.
3. Make the parallel sizes legal before touching recompute or VPP.
4. Only enable `interleaving_size > 1` after the non-VPP strategy is already working.

Examples:

- dense TP/PP baseline:
  [configs/strategy/tp1_pp2_dp4_mbs1.json](../configs/strategy/tp1_pp2_dp4_mbs1.json)
- MoE EP baseline:
  [configs/strategy/ep8_pp1_dp8_mbs1.json](../configs/strategy/ep8_pp1_dp8_mbs1.json)

## When to search instead of editing by hand

If you already have a reasonable starting strategy JSON, it is often easier to:

1. keep the parallel strategy fixed and search `micro_batch_size` / `micro_batch_num`
2. then search a small `tp/pp` space around the nearest existing config

Public references:

- [tutorial.md](./tutorial.md)
- [examples/search_strategy_llama3_8b.py](../examples/search_strategy_llama3_8b.py)

Search note:

- `gmi_error` is a simple per-rank memory margin in GiB for NCCL buffers and
  other runtime overheads that are not modeled explicitly
- start with a conservative value such as `10` on a new machine, then tighten
  it only after comparing against real memory usage

## Minimal viable strategy config

```json
{
    "seq_len": 4096,
    "micro_batch_size": 1,
    "micro_batch_num": 8,
    "dtype": "bf16",
    "world_size": 8,
    "tp_size": 1,
    "pp_size": 1,
    "ep_size": 1,
    "etp_size": 1,
    "enable_sequence_parallel": false,
    "interleaving_size": 1,
    "zero_state": 1,
    "enable_dropout": false,
    "use_flash_sdp": true,
    "enable_recompute": false,
    "mem_factor": 0.94
}
```

## First mental model for a single 8-GPU node

Start simple:

1. dense model, no VPP, no recompute
2. `world_size=8`, `tp=1`, `pp=1`, `ep=1`, `cp=1`
3. this means the remaining parallelism is pure data parallel

Then add complexity one step at a time:

1. increase `tp_size` if a single layer is too large
2. increase `pp_size` if the whole model is too large
3. use `ep_size` only for MoE models
4. use `interleaving_size > 1` only after ordinary PP works

## Required fields and common defaults

Fields you will usually set explicitly:

- `seq_len`
- `micro_batch_size`
- `micro_batch_num`
- `world_size`
- `tp_size`
- `pp_size`
- `ep_size`
- `etp_size`
- `dtype`

Fields many users can leave at the shipped defaults at first:

- `zero_state`
- `enable_dropout`
- `mem_factor`
- most `use_fused_*` toggles
- most recompute sub-switches

## How the parallel sizes relate

The most common dense relation is:

- `dp = world_size / (tp * pp * cp)`

So for a legal dense config:

- `world_size` must be divisible by `tp * pp * cp`

For MoE, SimuMax also checks:

- `world_size % (ep * etp * pp) == 0`

So a practical rule is:

- get a legal dense `tp/pp/cp` split first
- then add `ep`
- then check model-specific expert divisibility such as `expert_num % ep == 0`

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
Routing strategy for MOE models, default is "all2all". `all2all-seq` is deprecated and will be downgraded to `all2all` with a warning.
### enable_sequence_parallel
Whether to enable sequence parallelism, default is true, effective when tp_size > 1
### num_layers_in_first_pipeline_stage & num_layers_in_last_pipeline_stage
Controls the number of layers contained in the first and last Pipeline Parallel stages, default is None
### interleaving_size
Virtual pipeline size. Keep it at `1` for the first working strategy.
When `interleaving_size > 1`, `pp_size` must also be greater than `1`.
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

#### megatron_recompute
Megatron-LM 0.14 introduced selective recompute based on `discard_output`.
Enable this mode with `megatron_recompute=true` and list the modules whose
outputs are discarded in `megatron_recompute_modules`.

Example:

```json
{
    "enable_recompute": true,
    "recompute_granularity": "selective_recompute",
    "recompute_layer_num": 12,
    "megatron_recompute": true,
    "megatron_recompute_modules": ["layernorm", "mlp"]
}
```

Supported module names are `layernorm`, `mla_up_proj`, `moe_act`, `mlp`, and
`moe`. `core_attn` is reserved but not supported yet. This mode is mutually
exclusive with the legacy selective flags such as `attn_recompute` and
`mlp_recompute`; evaluate these strategies explicitly rather than through the
current search helper.
## Computation Optimization
### attention_sparse_ratio
Attention sparse ratio (0.0 indicates dense attention), default is 0.0
### use_flash_sdp
Use FlashAttention acceleration
### cross_entropy_loss_fusion
Whether to enable fused cross entropy in SimuMax, default is `false`.

Megatron mapping:

- SimuMax strategy field: `cross_entropy_loss_fusion=true`
- common shorthand in this repo: `ce_fusion`
- common case-name suffix in retained result tables: `_cef`

For Megatron real runs, this shorthand means enabling both:

- `--cross-entropy-loss-fusion`
- `--cross-entropy-fusion-impl te`

So `ce_fusion` / `_cef` in repo materials should be read as:

- `cross_entropy_loss_fusion=True`
- TE fused CE implementation
### use_fused_*
Various fused kernel optimizations
### enable_dropout
Whether to enable Dropout regularization, default is false

## Network Strategy
### tp_net, pp_net, dp_net, etc.
Network communication strategies for various parallel dimensions, default is "auto", automatically selected based on cluster scale and parallel strategy

## Other
### dispatch_probs
Megatron-LM-related parameter for the MoE probs ownership path.

- Megatron-LM 0.14 and later: use `dispatch_probs=true`
- Megatron-LM 0.12 and earlier: use `dispatch_probs=false`

For intermediate or locally patched runtimes, confirm the actual MoE path
before choosing the flag.
### mem_factor
Memory usage coefficient (0.94 means reserving 6% margin), used to estimate reserve_memory (=max_memory / mem_factor), default is 0.94
