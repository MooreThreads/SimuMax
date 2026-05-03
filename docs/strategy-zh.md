<p align="center">
  <a href="strategy.md">English</a>| 
  <a href="strategy-zh.md">中文版本</a> 
</p>

# Strategy 配置
SimuMax 依赖三个核心输入文件：`system`、`strategy`、`model`。`strategy` 文件定义训练运行时选择，例如 TP / PP / EP、总卡数、batch、recompute、VPP 等。

相关文档：

- [docs/README.md](./README.md)
- [model.md](./model.md)
- [system.md](./system.md)

strategy 是 SimuMax 和 Megatron 运行时语义最直接对齐的一层。若 real run 和 strategy 在 PP / EP / TP、sequence parallel、recompute、VPP 等设置上不一致，timing 和 memory 都可能明显漂移。

## 最快起步方式

除非非常特殊，否则不要从空文件开始写。

推荐路径：

1. 从 [configs/strategy](../configs/strategy) 复制最接近的已有 JSON。
2. 先把 `seq_len`、`micro_batch_size`、`micro_batch_num` 配成最简单的版本。
3. 先保证并行规模合法，再考虑 recompute 或 VPP。
4. 只有普通 PP 跑通后，再启用 `interleaving_size > 1`。

示例起点：

- dense TP/PP 基线：
  [configs/strategy/tp1_pp2_dp4_mbs1.json](../configs/strategy/tp1_pp2_dp4_mbs1.json)
- MoE EP 基线：
  [configs/strategy/ep8_pp1_dp8_mbs1.json](../configs/strategy/ep8_pp1_dp8_mbs1.json)

## 什么时候适合先做 search

如果你已经有一份接近的 strategy JSON，通常更推荐：

1. 先固定并行策略，搜索 `micro_batch_size` / `micro_batch_num`
2. 再围绕最近的已有配置，小范围搜索 `tp/pp`

公共入口可参考：

- [tutorial.md](./tutorial.md)
- [examples/search_strategy_llama3_8b.py](../examples/search_strategy_llama3_8b.py)

补充说明：

- `gmi_error` 是按卡预留的 GiB 级显存余量，用来粗略覆盖 NCCL buffer、
  allocator/runtime 开销，以及其他没有显式建模的组件
- 在新机器上第一次做 search 时，可以先用较保守的 `10`，之后再结合 real
  显存结果收紧

## 最小可用 strategy 示例

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

## 单机 8 卡的起步心智模型

先从最简单的开始：

1. dense 模型，不开 VPP，不开 recompute
2. `world_size=8`，`tp=1`，`pp=1`，`ep=1`，`cp=1`
3. 这时剩余的并行就是纯 data parallel

然后逐步增加复杂度：

1. 单层太大，再增大 `tp_size`
2. 整体模型太大，再增大 `pp_size`
3. 只有 MoE 模型才引入 `ep_size`
4. 只有普通 PP 跑通后，才引入 `interleaving_size > 1`

## 常用必填字段和常见默认

通常需要明确设置的字段：

- `seq_len`
- `micro_batch_size`
- `micro_batch_num`
- `world_size`
- `tp_size`
- `pp_size`
- `ep_size`
- `etp_size`
- `dtype`

很多用户一开始可以保持默认的字段：

- `zero_state`
- `enable_dropout`
- `mem_factor`
- 大多数 `use_fused_*`
- 大多数 recompute 子开关

## world_size / tp / pp / ep / dp 的关系

dense 场景下最常见的关系是：

- `dp = world_size / (tp * pp * cp)`

也就是说，dense 配置至少要满足：

- `world_size` 能被 `tp * pp * cp` 整除

MoE 场景还需要满足：

- `world_size % (ep * etp * pp) == 0`

实际建议是：

- 先把 dense 的 `tp/pp/cp` 配合法
- 再加入 `ep`
- 再检查模型侧 expert 相关整除，例如 `expert_num % ep == 0`

# 参数说明
## 基础训练参数
### seq_len
序列长度（token数量）
### micro_batch_size
微批次大小（单词每次前向传播处理的样本数）
### micro_batch_num
梯度累积的微批次数量
### dtype
计算数据类型（bf16表示半精度浮点数），默认为bf16
### fp8
是否使用fp8混合精度训练，默认为false
## 分布式策略
### world_size
总GPU数量（默认8）
### tp_size
张量并行大小（Tensor Parallelism），默认为1
### pp_size
流水线大小（Pipeline Parallelism），表示按层进行纵向切分，默认为1
### ep_size
专家大小（Expert Parallelism），仅用于MOE模型，默认为1
### etp_size
专家张量大小（Expert Tensor Parallelism），默认为1
### moe_dispatcher_policy
MOE模型的路由策略， 默认为"all2all"
### enable_sequence_parallel
是否启用序列并行，默认为true，当tp_size > 1时生效
### num_layers_in_first_pipeline_stage & num_layers_in_last_pipeline_stage
控制第一个和最后一个Pipeline Parallel stage包含的层数，默认为None
### interleaving_size
虚拟 pipeline 大小。第一次起步时保持 `1` 即可。`interleaving_size > 1` 时，`pp_size` 也必须大于 `1`。
### zero_state
ZeRO优化配置，目前只支持zero0和1，默认为1
## 内存优化
### grad_reduce_in_bf16
梯度归约是否使用bf16，默认为false
### use_accm_weight
是否使用累加权重融合（减少临时变量）, 默认为true
### cache_groupgemm_col_fp8_inputs
是否缓存groupgemm的FP8输入，默认为false
### offload_groupgemm_col_inputs
是否卸载groupgemm的输入到CPU，默认为false


## 重计算相关
#### enable_recompute
recompute全局开关，默认为true
#### recompute_granularity
recompute的粒度，可选为"full_block"和"selective_recompute"，默认为None
#### recompute_layer_num
recompute的层数，默认为0
#### attn_recompute
对attention模块进行重计算，默认为false
#### mla_rms_recompute
对mla的rmsnorm和q/k up-projection进行重计算，默认为false
#### mlp_recompute
对MLP和groupedgemm进行重计算，默认为false
#### mlp_rms_recompute
对rmsnorm+router+sharedExpert进行重计算，默认为false
#### recompute_variance
recompute checkpoint的最后一个module是否去掉冗余的前向计算，默认为false, 当recompute_granularity为"selective_recompute"时，建议设置为true以节省计算时间

#### megatron_recompute
Megatron-LM 0.14 引入了基于 `discard_output` 的 selective recompute。使用
`megatron_recompute=true` 开启，并在 `megatron_recompute_modules` 中列出
被 discard output 的模块。

示例：

```json
{
    "enable_recompute": true,
    "recompute_granularity": "selective_recompute",
    "recompute_layer_num": 12,
    "megatron_recompute": true,
    "megatron_recompute_modules": ["layernorm", "mlp"]
}
```

当前支持的模块名包括 `layernorm`、`mla_up_proj`、`moe_act`、`mlp`、`moe`。
`core_attn` 是预留名，但暂不支持。该模式不能和旧的 selective flags
（例如 `attn_recompute`、`mlp_recompute`）混用；目前也不通过 search helper
自动搜索，建议显式配置后单独评估。

## 计算优化
### attention_sparse_ratio
注意力稀疏比例（0.0为密集注意力），默认为0.0
### use_flash_sdp
使用FlashAttention加速
### cross_entropy_loss_fusion
是否启用 SimuMax 里的 fused cross entropy，默认是 `false`。

与 Megatron 的对应关系：

- SimuMax strategy 字段：`cross_entropy_loss_fusion=true`
- 本仓库常用简称：`ce_fusion`
- 结果表里常见的 case 后缀：`_cef`

对 Megatron real run，这个简称对应同时开启：

- `--cross-entropy-loss-fusion`
- `--cross-entropy-fusion-impl te`

因此仓库里的 `ce_fusion` / `_cef` 可以理解为：

- `cross_entropy_loss_fusion=True`
- 使用 TE 的 fused CE 实现
### use_fused_*
各种融合内核优化
### enable_dropout
是否启用Dropout正则化，默认为false


## 网络策略
### tp_net, pp_net, dp_net等
各种并行维度的网络通信策略，默认为"auto"，根据集群规模和并行策略自动选择

## 其他
### dispatch_probs
Megatron-LM 相关参数，用来决定 MoE 里 probs 的归属口径。

- Megatron-LM 0.14 及之后：用 `dispatch_probs=True`
- Megatron-LM 0.12 及更早：用 `dispatch_probs=False`

如果是中间过渡版本或本地 patch 版本，先确认实际走到的 MoE 路径，再决定这个开关。
### mem_factor
内存使用系数（0.94表示留6%的余量），用于估算reserve_memory（=max_memory / mem_factor），默认为0.94。
