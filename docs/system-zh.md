<p align="center">
  <a href="system.md">English</a>| 
  <a href="system-zh.md">中文版本</a> 
</p>


# System 配置
SimuMax 依赖三个核心输入文件：`system`、`strategy`、`model`。`system` 文件描述机器侧能力：

- 加速器算力
- 访存带宽
- 机内 / 机间通信带宽与 latency
- shape 级别的算子效率

一个完整的 `system.json` 总是包含三部分：

- 基本信息：系统名、单机卡数
- `accelerator`：计算和访存侧描述
- `networks`：机内和机间通信描述

某些机器族还会带额外字段，比如 `FC8`，但第一次理解时先抓住上面三部分即可。

相关入口：

- 总览：[README.md](./README.md)
- model 字段：[model.md](./model.md)
- strategy 字段：[strategy.md](./strategy.md)
- 机器测速流程：[simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md)

实际使用时还有一个很重要的口径：

- 共享公共流程可以自动生成算子效率
- 在支持的 CUDA/MUSA 硬件上，共享流程也会尝试自动补上 `accelerator.backend`、当前可见 `num_per_node` 和 `accelerator.mem_gbs`
- 但通信拟合结果目前仍需要人工写回 `networks`
- `accelerator.bandwidth` 的默认值仍然只是起步模板，做 timing 分析前需要人工确认
- 所以新生成的 `system.json` 只有在你检查过机器侧字段并把 starter network 数值替换成实测通信参数之后，才适合做 timing 分析

## 什么时候需要自己实测

以下场景通常可以直接使用已有 system 配置：

- 目标机器和仓库中的示例机器接近
- 通信拓扑没有本质差别
- 目标 case 的主要算子 shape 已经有 `accurate_efficient_factor`

以下场景建议先做自己的机器实测：

- 机器是新的
- 机内或机间带宽 / latency 与已有配置差异明显
- `system.miss_efficiency` 不是空，而且你要解释 timing

经验规则：

- 如果只是做 OOM feasibility 判断，缺失 efficiency 还可以暂时容忍
- 如果要解释 `perf vs simulator` 或 `perf vs real` 的 timing，先补齐 efficiency

## 最快起步方式

除非非常特殊，否则不要从空文件开始写。

推荐路径：

1. 从 [configs/system](../configs/system) 复制最接近的已有配置。
2. 修改 `sys_name`。
3. 修改 `num_per_node`、`accelerator.backend`、`accelerator.mem_gbs`。
4. 先把 `networks` 改成最接近你机器拓扑的版本。
5. 最后再补 `accurate_efficient_factor` 和拟合得到的通信参数。

如果你只是做近似分析，复制最近的已有配置通常比从空文件新建更稳妥。

## 最小可用模板

下面这个例子是一个真正完整、可作为起点的最小模板，包含了 `networks` 骨架。

```json
{
    "sys_name": "my_system",
    "num_per_node": 8,
    "accelerator": {
        "backend": "cuda",
        "mem_gbs": 80,
        "op": {
            "default": {
                "tflops": 312,
                "efficient_factor": 0.75
            }
        },
        "bandwidth": {
            "default": {
                "efficient_factor": 0.9,
                "gbps": 1600,
                "latency_us": 40
            }
        },
        "mode": "roofline"
    },
    "networks": {
        "intra_with_pcie": false,
        "low_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 300,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        },
        "high_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 300,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        },
        "inter_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 200,
                "latency_us": 30
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        }
    }
}
```

## 必填字段与常见默认

建议视为必填的字段：

- `sys_name`
- `num_per_node`
- `accelerator.backend`
- `accelerator.mem_gbs`
- `accelerator.op.default`
- `accelerator.bandwidth.default`
- `networks.intra_with_pcie`
- `networks` 下对应的网络组

很多用户一开始可以沿用已有配置的字段：

- `accelerator.mode`（通常是 `roofline`）
- `processor_usage`（当前公共配置里基本都是保留字段）
- `accelerator.bandwidth` 下算子级访存微调
- `networks.*.op` 下算子级通信微调

如果只是做近似分析，可以：

- 先复制最近的已有 system config
- 暂时保留很多默认效率
- 用最接近机器的通信参数先跑通

如果要 timing 更准确，建议自己测：

- 主要 `matmul`、`group_matmul`、attention shape
- 机内 / 机间通信带宽与 latency

换句话说：

- `accelerator.op.*.accurate_efficient_factor` 用来补齐算子效率
- `networks.*` 用来补齐通信 timing
- `num_per_node`、`accelerator.mem_gbs`、`accelerator.bandwidth.*` 这些机器侧字段，在依赖 timing 结果前也需要人工确认

共享测速流程见 [simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md)。

## accelerator
accelerator部分包含了显存、访存带宽、算力、各个算子的计算效率等。

### backend 
后端描述，仅用于标识。


### mem_gbs
显存大小，单位为GB。

### op
该部分定义了各个算子使用的默认算力和不同shape下准确的计算效率。

SimuMax的核心之一是实现了shape级别计算效率建模，这是性能准确建模的关键，因此，SimuMax支持用户自定义多个核心算子在不同shape下的计算效率描述, 并且定义了一套shape表达规则， 用户需要按照该规则来新增算子不同shape的计算效率。

目前支持的算子列表和其shape表达规则为：


|key|算子|格式|示例|备注|
|---|---|---|---|---|
|matmul|矩阵乘法| b={batch_size}, m={m}, k={k}, n={n}, layout={layout}, accumulate={accumulate}, out_dtype={out_dtype}|`b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16`|accumulate：是否进行梯度累加，反向对w求导时该项为True|
|fp8_matmul|FP8矩阵乘法|同上|同上|同上|
|sdp_fwd|SDP前向计算|batch={batchh_size}, seq_len={seq_len}, head_num={head_num}, kv_head_num={kv_head_num}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}, qkv_contiguous={qkv_contiguous}|`batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0729673001633662`| qkv_contiguous：输入qkv是否在内存上连续，该项影响计算性能，因此单独描述，A100上一般为连续输入|
|sdp_bwd|SDP反向计算|同上|同上|同上|
|group_matmul| MOE模型的分组matmul|ng={num_groups}, M={fwd_M}, N={fwd_N}, K={fwd_k}, dtype={dtype}, out_dtype={out_dtype}, main_grad_dtype={main_grad_dtype}, stage={stage}, grad={grad}, accumulate={accumulate}, use_split_accumulator=False, single_output={single_output}| fwd stage：<br> `ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True": 0.6313438865579614`|1. fwd、bwd_grad_act、bwd_grad_w三个阶段的groupedgemm shape描述的M,N,K都等于fwd阶段的M,N,K，用stage来区分不同阶段<br>2. single_output只有fwd stage时为True<br>3. accumulate只有bwd_grad_w stage时为True<br>4. grad和use_split_accumulator只有在bwd_grad_w stage时为True|




例如，对于NVIDIA A100，其各个算子使用的算力和不同shape的计算效率描述为:

```json

"op": {
    "default" : {
        "tflops": 312,
        "efficient_factor": 0.75
    },
     "matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16": 0.7876672065615554,
                    "b=1, m=4096, k=1536, n=5120, layout=NN, accumulate=False, out_dtype=bf16": 0.737505124681297
                },
    },
     "fp8_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {},
     },
     "sdp_fwd" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0729673001633662,
                    "batch=1, seq_len=4096, head_num=64, kv_head_num=64, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0544285429372056
                },
     },
     "sdp_bwd" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 0.8018473732899901,
                    "batch=1, seq_len=4096, head_num=64, kv_head_num=64, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 0.7942592665301026
                },
     },
     "group_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True": 0.6313438865579614,
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=bwd_grad_act, grad=True, accumulate=False, use_split_accumulator=True, single_output=False": 0.6790978664070304,
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=bwd_grad_w, grad=True, accumulate=True, use_split_accumulator=True, single_output=False": 0.5196854178569805
                },
     },
     "fp8_group_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {},
     },
}
```
其中，`default`表示默认算力，不支持的算子类型使用该算力；每个算子下面，`tflops`表示标称算力，`efficient_factor`表示默认计算效率，`accurate_efficient_factor`表示各个算子在不同shape下的实际计算效率。


### bandwidth
访存带宽描述，包含各个访存类型的带宽。例如，对于NVIDIA A100，其访存带宽描述为:

```json
"bandwidth": {
    "default" : {
        "efficient_factor": 0.91,
        "gbps": 1600,
        "latency_us": 40
    },
    "permute_fwd":{
        "efficient_factor": 0.1879,
        "gbps": 1600,
        "latency_us": 40
    },
    "permute_bwd":{
        "efficient_factor": 0.1879,
        "gbps": 1600,
        "latency_us": 40
    },
    "ce":{
        "efficient_factor": 0.808,
        "gbps": 1600,
        "latency_us": 40
    }
}
```
其中，default表示默认访存带宽及其效率；除了默认访存带宽，我们新增3个memory bound算子的算子微调效率，用户可以自定义：
- permute_fwd表示permute前向的访存带宽及其效率
- permute_bwd表示permute反向的访存带宽及其效率
- ce表示cross entropy的访存带宽及其效率

## networks
### FC8
是否为FC8互联。
### intra_with_pcie
机内是否是pcie互连。
- intra_with_pcie=True，则表示机内是pcie互连，networks还需包含以下网络带宽配置
```json
"intra_node_pcie_8x": {
},
"intra_node_pcie_4x": {
},
"intra_node_pcie_2x": {
},
"inter_node": { 
}
```
- intra_with_pcie=False， 否则表示机内是nvlink高速互连， networks还需包含以下网络带宽配置。  
```json
"low_intra_node": {
},
"high_intra_node": {
},
"inter_node": { 
}
```

### intra_node_pcie_8x/intra_node_pcie_4x/intra_node_pcie_2x/low_intra_node/high_intra_node/inter_node   
每一种网络带宽配置，包含以下参数：
- processor_usage: unused, 保留字段     
- bandwidth: 网络带宽配置，包含以下参数
    - efficient_factor: 网络带宽效率
    - gbps: 网络带宽
    - latency_us: 网络延迟
- op: 网络带宽效率，包含以下参数
    - all_reduce: all_reduce操作的网络带宽效率
        - scale: 2，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - all_gather: all_gather操作的网络带宽效率
        - scale: 1，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - reduce_scatter: reduce_scatter操作的网络带宽效率
        - scale: 1，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - p2p: p2p操作的网络带宽效率
        - scale: 1，固定
        - offset: 0， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - all2all: all2all操作的网络带宽效率
        - scale: 1，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选 

例如A100_PCIE相邻两卡通信带宽详细配置：
```json
"intra_node_pcie_2x": {
            "processor_usage": 0.00,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 30,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1,
                    "efficient_factor": 0.6965,
                    "latency_us": 15.51
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6866,
                    "latency_us": 24.84
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6419,
                    "latency_us": 131.30
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6969,
                    "latency_us": 24.07
                }
            }

        },
```
