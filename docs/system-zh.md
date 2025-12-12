<p align="center">
  <a href="system.md">English</a>| 
  <a href="system-zh.md">中文版本</a> 
</p>


# 介绍
SimuMax依赖于三个核心输入文件：system，strategy, model。system文件定义了一个GPU集群系统的硬件性能参数，例如标称算力，访存带宽，机内/机间带宽，用于评估计算时间和网络通信时间。

system文件主要包含3部分描述：**基本信息(系统名称和单机卡数)**, **accelerator(算力和访存描述)**，**networks(网络配置和带宽描述)**。

一个基础模板为:

```json
system_template = {
    "sys_name": "A100",
    "num_per_node": 8,
    "accelerator": {
        "backend": "cuda",
        "mem_gbs": 80,
        "op" : {

        },
        "bandwidth": {
            
        },
        "mode": "roofline"
    },
    "FC8":true,
}
```

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
        - offset: -1， 固定
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