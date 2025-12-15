<p align="center">
  <a href="README.md">English</a>| 
  <a href="README-zh.md">中文版本</a> 
</p>


# Introduction
SimuMax provides a set of benchmark performance testing scripts based on Megatron-LM and Transformer-Engine, which can be used to conduct single-node performance tests for llama and deepseek models on NVIDIA devices. These tests are used to calibrate performance results with SimuMax simulations.

# Benchmark Performance Testing
## llama3
Supported test model list:
- llama2-7b
- llama3-8b
- llama3-70b

First, add the test host IP to the hostfile, then run the following command:
```bash
bash run_llama3.sh ${mbs} ${mbc} ${tp_size} ${pp_size} ${model_type} ${test_type} ${num_layers}
```

The ${test_type} parameter has two options: test and profile. test indicates performance testing with mock-data for 10 iterations; profile indicates using torch.profiler to export trace files, running 6 iterations with 1 warm-up iteration.

For example, to run a performance test for a 12-layer llama3-70b, with log files generated in ./output_llama3_70b:

```bash
bash run_llama3.sh 1 32 1 2 llama3_70b test 12 
```

## deepseek
Supported test model list:
- deepseek-v2
- deepseek-v3

First, add the test host IP to the hostfile, then run the following command:
```bash
bash run_deepseekv2.sh ${mbs} ${mbc} ${ep_size} ${pp_size} ${model_type} ${test_type} ${num_layers}
```
The ${test_type} parameter has two options: test and profile. test indicates performance testing with mock-data for 10 iterations; profile indicates using torch.profiler to export trace files, running 6 iterations with 1 warm-up iteration.

For example, to run a performance test for a 4-layer deepseek-v2, with log files generated in ./output_deepseek_v2:

```bash
bash run_deepseekv2.sh 1 32 8 1 deepseek_v2 test 4
```
To run a performance profile for a 4-layer deepseek-v2, with log files generated in ./output_deepseek_v2/tboard:
```bash
bash run_deepseekv2.sh 1 32 8 1 deepseek_v2 profile 4
```

# A100 Test Environment
Framework:
- TransformerEngine: https://github.com/NVIDIA/TransformerEngine/tree/release_v2.1
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM/releases/tag/v0.12.0rc3

NVIDIA-related hardware/software environment:
- NCCL: nvidia-nccl-cu12 2.21.5
- CUDA 12.6
- NVIDIA A100 80GB PCIe

Using the community version of flash-mla on A100:
```bash
git clone -b head_dim_not_equal https://github.com/defei-coder/flash-attention.git
cd flash-attention
MAX_JOBS=8 python setup.py install --verbose > install.log
```
