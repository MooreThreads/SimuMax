<p align="center">
  <a href="README.md">English</a>| 
  <a href="README-zh.md">中文版本</a> 
</p>

# 自动生成system.json文件
## 介绍
SimuMax的性能评估准确性不仅得益于标准的模型定义、与训练框架一致的计算数据流，还得益于算子shape粒度的实测计算效率和逐通信原语的带宽效率+latency拟合。

SimuMax的system.json文件定义了设备的标称算力、各种算子在不同shape下的计算效率、机内/机间带宽等，其各个属性描述请查阅[system.md](../../docs/system.md)。

## 步骤
生成符合SimuMax标准的system.json文件包含两个步骤：
1. 测试算子shape级别的计算效率：首先批量运行给定的标准算子计算效率测试流程得到system.json初版
2. 拟合机内/机间通信带宽：运行例如<b>nccl_test</b>通信测试工具，得到各个通信原语在一个通信量区间的通信时间，基于该通信量区间的实测时间拟合各个通信算子的机内/机间通信效率和latency。将拟合的通信效率和latency填入system.json中，完成。

### 测试算子shape级别的计算效率
首先批量运行给定的标准算子计算效率测试流程得到system.json初版。

根据硬件实际配置，修改[run.sh](./run.sh)文件中的MAX_TFLOPS、SYS_NAME、PICE_INTRA_LINK等环境变量来指定设备的标称算力、系统名称、是否卡键为PCIE互联等。详细示例如下：
```shell
export MAX_TFLOPS=312 # 指定机器的标称算力
export SYS_NAME="A100_PCIE" #指定system名称
export PICE_INTRA_LINK=1 #指定机内卡间是否为PCIE互联
export FC8_MODE=0 #指定机内卡间通信连接是否为FC8模式
export PARAM_FILE="./run_params.json" # 测试超参数，指定测试模型的列表，mbs区间，seq_len区间等, 如果不指定，则使用默认超参。
python test_gemm_efficency.py
python test_grouped_gemm_efficency.py
python test_fa_efficency.py
python combine_efficency.py
```
其中run_params.json文件示例如下：
```json
{
    "model_list":["deepseekv2",
                  "deepseekv3",
                  "deepseek-32b",
                  "deepseek-16b",
                  "deepseek-1b",
                  "llama3-8b",
                  "llama3-70b",
                  "qwen3-32b",
                  "kimi-1T",
                  "mixtral-8x7b"],
    "mbs_list": [1, 2, 4],
    "seq_len_list":[4096],
    "tp_list": [1, 2, 4, 8],
    "ep_list": [1, 2, 4, 8, 16, 64]
}
```
运行：
```shell
bash run.sh
```
在运行目录下会生成{system_name}.json文件（例如实例中生成A100_PCIE.json文件）。每个模型的每种超参组合测试运行时间通常大约在1min左右。


生成的system文件中，"networks"部分的机内带宽数值为默认的经验值，如果PICE_INTRA_LINK=0，则为A100_pcie的经验值：30GB/s的机内标称带宽及其精调效率+200GB/s的机间带宽；如果PICE_INTRA_LINK=1，则为A100-sxm的50GB/s机内标称带宽+200GB/s的机间带宽。

注意：现在该测试框架基于简单的时间戳统计算子运行时间，因此测试结果可能存在误差，后续会考虑使用更精确的测试方法，例如 cuda events。

#### 效率自动化测试框架支持的模型限制
SimuMax目前可仿真的模型在./configs/models目录下，但不是所有模型的所有超参配置都可以基于现在这套框架进行自动化算子效率测试：
- Moe模型：对于MOE模型，仅支持TP=1配置下的效率测试，这意味着对于mixtral这类可开TP(ETP)的模型。
- Dense模型：全支持

### 拟合机内/机间通信带宽
如果设备为A100_PCIE，则<b>上述步骤生成的system文件中，默认"networks"部分的机内带宽为我们已经拟合过的带宽，可以直接使用，如果多机进行perf，仍需要进一步对机间带宽精调</b>。

如果为其它设备，则需要使用例如[nccl_test](https://github.com/NVIDIA/nccl-tests)工具来测试reduce-scatter、all2all、allgather等通信算子在某一区间（建议1M->8GB）的通信时间，随后通过线性回归拟合一个bw*eff和latency的带宽模型，得到各个通信原语的通信效率值eff和latency。

- nccl_test测试命令参考[nccl_test.sh](nccl_test.sh)：

    ```shell
    end=8G
    echo "run all_reduce_perf"
    ./build/all_reduce_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_all_reduce.txt
    echo "run all_gather_perf"
    ./build/all_gather_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_all_gather.txt
    echo "run reduce_scatter_perf"
    ./build/reduce_scatter_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_reduce_scatter.txt
    echo "run alltoall_perf"
    ./build/alltoall_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_alltoall_perf.txt
    echo "run sendrecv_perf"
    ./build/sendrecv_perf -n 1 -b 1M -e $end -f 2 -g 1 -t 1  -d bfloat16 > send_recv.txt
    ```

- 基于测试通信时间拟合通信效率和latency，请参考[nccl_fit.py](nccl_fit.py)。