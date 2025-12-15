<p align="center">
  <a href="strategy.md">English</a>| 
  <a href="strategy-zh.md">中文版本</a> 
</p>

# 介绍
SimuMax依赖于三个核心输入文件：system，strategy, model。strategy文件定义了训练的并行策略如tp/pp/ep，集群卡数， batchsize， 重计算策略等。

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
流水线大小（Pipeline Parallelism）### 纵向切分层数，默认为1
### ep_size
专家大小（Expert Parallelism）### 用于MOE模型，默认为1
### etp_size
专家张量大小（Expert Tensor Parallelism），默认为1
### moe_dispatcher_policy
MOE模型的路由策略， 默认为"all2all"
### enable_sequence_parallel
是否启用序列并行，默认为true，当tp_size > 1时生效
### num_layers_in_first_pipeline_stage & num_layers_in_last_pipeline_stage
控制第一个和最后一个Pipeline Parallel stage包含的层数，默认为None
### interleaving_size
保留字段，暂未使用  
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

## 计算优化
### attention_sparse_rati
注意力稀疏比例（0.0为密集注意力），默认为0.0
### use_flash_sdp
使用FlashAttention加速
### use_fused_*
各种融合内核优化
### enable_dropout
是否启用Dropout正则化，默认为false


## 网络策略
### tp_net, pp_net, dp_net等
各种并行维度的网络通信策略，默认为"auto"，根据集群规模和并行策略自动选择

## 其他
### dispatch_probs
Megatron-LM相关参数（是否分发probs）， 0.14版本Megatron-LM需要设置为true
### mem_factor
内存使用系数（0.94表示留6%的余量），用于估算reserve_memory（=max_memory / mem_factor），默认为0.94。