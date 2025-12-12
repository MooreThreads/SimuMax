<p align="center">
  <a href="system.md">English</a>| 
  <a href="system-zh.md">中文版本</a> 
</p>


# Introduction
SimuMax relies on three core input files: system, strategy, and model. The system file defines the hardware performance parameters of a GPU cluster system, such as nominal computing power, memory access bandwidth, intra-node/inter-node bandwidth, used for evaluating computation time and network communication time.

The system file primarily consists of three parts: Basic Information (system name and number of GPUs per node), accelerator (computing power and memory access description), and networks (network configuration and bandwidth description).

A basic template is:

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
The accelerator section includes GPU memory size, memory access bandwidth, computing power, and computational efficiency for various operators.

### backend 
Backend description, used for identification only.


### mem_gbs
GPU memory size, unit is GB.

### op
This section defines the default computing power used by various operators and the accurate computational efficiency under different shapes.

One of SimuMax's core features is implementing shape-level computational efficiency modeling, which is key to accurate performance modeling. Therefore, SimuMax supports user-defined descriptions of computational efficiency for multiple core operators under different shapes and defines a set of shape expression rules. Users need to follow these rules to add computational efficiency for different operator shapes.

The currently supported operator list and their shape expression rules are:




|key|Operator|Format|Example|Notes|
|---|---|---|---|---|
|matmul|Matrix Multiplication| b={batch_size}, m={m}, k={k}, n={n}, layout={layout}, accumulate={accumulate}, out_dtype={out_dtype}|`b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16`|`accumulate`: whether gradient accumulation is performed; True during backward pass for w gradient|
|fp8_matmul|	FP8 Matrix Multiplication|Same as above|Same as above|Same as above|
|sdp_fwd|SDP Forward Computation|batch={batchh_size}, seq_len={seq_len}, head_num={head_num}, kv_head_num={kv_head_num}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}, qkv_contiguous={qkv_contiguous}|`batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0729673001633662`| `qkv_contiguous`: whether input qkv is contiguous in memory; this affects performance, so described separately; generally contiguous input on A100|
|sdp_bwd|SDP Backward Computation|Same as above|Same as above|Same as above|
|group_matmul| Grouped matmul for MOE models|ng={num_groups}, M={fwd_M}, N={fwd_N}, K={fwd_k}, dtype={dtype}, out_dtype={out_dtype}, main_grad_dtype={main_grad_dtype}, stage={stage}, grad={grad}, accumulate={accumulate}, use_split_accumulator=False, single_output={single_output}| fwd stage：<br> `ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True": 0.6313438865579614`|1. The` M, N, K `shape descriptions for the fwd, bwd_grad_act, bwd_grad_w stages all equal the M, N, K of the fwd stage, differentiated by the stage parameter.<br>2. `single_output` is True only for the fwd stage.<br>3. `accumulate` is True only for the bwd_grad_w stage.<br>4. `grad` and `use_split_accumulator` are True only for the bwd_grad_w stage.|
|fp8_group_matmul|	Grouped matmul for MOE models|Same as above|Same as above|Same as above|



For example, for NVIDIA A100, the computing power used by its various operators and descriptions of computational efficiency under different shapes are:

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
Here, `default` represents the default computing power, which is used for unsupported operator types; under each operator, `tflops` represents the nominal computing power, `efficient_factor` represents the default computational efficiency, and `accurate_efficient_factor` also indicates the actual computational efficiency of each operator under different shapes.


### bandwidth
Memory access bandwidth description, including bandwidth for various memory access types. For example, for NVIDIA A100, its memory access bandwidth description is:
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
Here, `default` represents the default memory access bandwidth and its efficiency. Besides the default, we add operator-specific fine-tuned efficiency for 3 memory-bound operators that users can customize:
- `permute_fwd` represents memory access bandwidth and efficiency for the permute forward pass.
- `permute_bwd` represents memory access bandwidth and efficiency for the permute backward pass.
- `ce` represents memory access bandwidth and efficiency for cross entropy.

## networks
### FC8
Whether it is FC8 (Fully Connected 8) interconnect.
### intra_with_pcie
Whether intra-node connection uses PCIe.
- if intra_with_pcie=True, it means intra-node connection uses PCIe, and the networks section must also include the following network bandwidth configurations:
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
- if intra_with_pcie=False, it means intra-node connection uses high-speed NVLink, and the networks section must also include the following network bandwidth configurations:
```json
"low_intra_node": {
},
"high_intra_node": {
},
"inter_node": { 
}
```

### intra_node_pcie_8x/intra_node_pcie_4x/intra_node_pcie_2x/low_intra_node/high_intra_node/inter_node   
Each network bandwidth configuration includes the following parameters:
- processor_usage: unused, reserved field     
- bandwidth: Network bandwidth configuration, includes:
    - efficient_factor: Network bandwidth efficiency
    - gbps: Network bandwidth, GB/s
    - latency_us: Network latency
- op: Network bandwidth efficiency for specific operations, includes:
    - all_reduce: Network bandwidth efficiency for all_reduce operation
        - scale: 2, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional
    - all_gather: Network bandwidth efficiency for all_gather operation
        - scale: 1, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional
    - reduce_scatter: Network bandwidth efficiency for reduce_scatter operation
        - scale: 1, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional
    - p2p: Network bandwidth efficiency for p2p (point-to-point) operation
        - scale: 1, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional
    - all2all: Network bandwidth efficiency for all2all operation
        - scale: 1, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional

For example, detailed configuration of continuous two-card communication bandwidth for A100_PCIE:
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