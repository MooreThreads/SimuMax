<p align="center">
  <a href="README.md">English</a>| 
  <a href="README-zh.md">中文版本</a> 
</p>


# 介绍
SimuMax提供了一套基于Megatron-LM和TE的基准性能测试脚本，可以在英伟达设备上进行llama和deepseek的单机性能测试，用于和SimuMax仿真的性能结果进行校准。
# 基准性能测试
## llama3
支持的测试模型列表：
- llama2-7b
- llama3-8b
- llama3-70b


首先在hostfile中添加测试主机ip，然后运行以下命令：
```bash
bash run_llama3.sh ${mbs} ${mbc} ${tp_size} ${pp_size} ${model_type} ${test_type} ${num_layers}
```
`${test_type}`参数可选`test`和`profile`两个选项，`test`表示基于mock-data运行10个iter的测试训练性能； `profile`表示使用torch.profiler工具导出trace文件， 跑6个iter, 预热1个iter。

例如，运行12层llama3-70b的性能测试，生成的日志文件在./output_llama3_70b中：
```bash
bash run_llama3.sh 1 32 1 2 llama3_70b test 12 
```

## deepseek
支持的测试模型列表：
- deepseek-v2
- deepseek-v3

首先在hostfile中添加测试主机ip，然后运行以下命令：
```bash
bash run_deepseekv2.sh ${mbs} ${mbc} ${ep_size} ${pp_size} ${model_type} ${test_type} ${num_layers}
```
`${test_type}`参数可选`test`和`profile`两个选项，`test`表示基于mock-data运行10个iter的测试训练性能； `profile`表示使用torch.profiler工具导出trace文件， 跑6个iter, 预热1个iter。

例如，运行4层deepseek-v2性能测试，生成的日志文件在./output_deepseek_v2中：
```bash
bash run_deepseekv2.sh 1 32 8 1 deepseek_v2 test 4
```
运行4层deepseek-v2性能profile，生成的日志文件在./output_deepseek_v2/tboard中：
```bash
bash run_deepseekv2.sh 1 32 8 1 deepseek_v2 profile 4
```

# A100 测试环境说明

框架：
- te: https://github.com/NVIDIA/TransformerEngine/tree/release_v2.1
- megatron-lm: https://github.com/NVIDIA/Megatron-LM/releases/tag/v0.12.0rc3\


N卡相关硬件/软件环境：
- NCCL： nvidia-nccl-cu12 2.21.5
- CUDA 12.6 
- NVIDIA A100 80GB PCIe

A100上使用社区版本flash-mla:
```bash
git clone -b  head_dim_not_equal https://github.com/defei-coder/flash-attention.git
cd flash-attention
MAX_JOBS=8 python setup.py install --verbose > install.log 