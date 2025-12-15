<p align="center">
  <a href="README.md">English</a>| 
  <a href="README-zh.md">中文版本</a> 
</p>

# Automatic Generation of system.json File

## Introduction
The accuracy of SimuMax's performance evaluation benefits not only from the standard model definition and computation data flow consistent with training frameworks, but also from the measured computational efficiency at the operator-shape granularity and the bandwidth efficiency and latency fitting of each communication primitive.

The system.json file in SimuMax defines the device's nominal computing power, computational efficiency of various operators under different shapes, intra-node/inter-node bandwidth, etc. For descriptions of each attribute, please refer to [system.md](../../docs/system.md).

<b>We provide a standard testing process that can generate system files that meet SimuMax's requirements</b>.

## Steps
Generating a system.json file that complies with SimuMax standards consists of two steps:

Test computational efficiency at operator-shape level: First, batch-run the standard operator computational efficiency testing process to obtain an initial version of system.json.

Fit intra-node/inter-node communication bandwidth: Run tools such as nccl communication test tools to obtain communication times for each communication primitive within a communication volume range. Based on the measured times within this range, fit the intra-node/inter-node communication efficiency and latency for each communication operator. Fill the fitted communication efficiency and latency values into system.json to complete the process.

### Step 1: Test Computational Efficiency at Operator-Shape Level
First, batch-run the standard operator computational efficiency testing process to obtain an initial version of system.json.

According to the actual hardware configuration, modify environment variables in [run.sh](./run.sh) such as MAX_TFLOPS, SYS_NAME, PICE_INTRA_LINK, etc., to specify the device's nominal computing power, system name, whether cards are connected via PCIE, etc. A detailed example is as follows:

```shell
export MAX_TFLOPS=312 # Specify the machine's nominal computing power
export SYS_NAME="A100_PCIE" # Specify the system name
export PICE_INTRA_LINK=1 # Specify whether intra-machine cards are connected via PCIE
export FC8_MODE=0 # Specify whether intra-machine communication connections are in FC8 mode
export PARAM_FILE="./run_params.json" # Test hyperparameters, specify the list of test models, mbs interval, seq_len interval, etc. If not specified, the default hyperparameters will be used.
python test_gemm_efficency.py
python test_grouped_gemm_efficency.py
python test_fa_efficency.py
python combine_efficency.py
```
An example of the run_params.json file is as follows:
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

Run:
```shell
bash run.sh
```
A {system_name}.json file will be generated in the running directory (e.g., A100_PCIE.json in this example).

In the generated system file, the intra-machine bandwidth values in the "networks" section are default empirical values. 

If `PICE_INTRA_LINK=0`, it corresponds to the empirical values for A100_pcie: 
- 30 GB/s nominal intra-machine bandwidth with finely-tuned efficiency
- Plus 200 GB/s inter-machine bandwidth

If `PICE_INTRA_LINK=1`, it corresponds to:
- 50 GB/s nominal intra-machine bandwidth for A100-sxm
- Plus 200 GB/s inter-machine bandwidth

Note: The current testing framework calculates operator runtime based on simple timestamping, so the test results may contain errors. More precise testing methods, such as CUDA events, will be considered for future implementation.

#### Model Limitations Supported by the Efficiency Automation Testing Framework  
The models that SimuMax can currently simulate are located in the ./configs/models directory, but not all hyperparameter configurations of all models can be tested for operator efficiency automatically using the current framework:  
- MoE Models: For MoE models, efficiency testing is only supported under TP=1 configurations, which means for models like Mixtral that allow TP (ETP).  
- Dense Models: Fully supported

### Step 2: Fit intra-node/inter-node Communication Bandwidth
If the device is A100_PCIE, then **in the system file generated through the above steps, the default intra-machine bandwidth in the "networks" section uses our pre-calibrated bandwidth values and can be used directly. However, if performing multi-machine performance simulation, further fine-tuning of inter-machine bandwidth is still required.**

For other devices, tools such as [nccl_test](https://github.com/NVIDIA/nccl-tests) need to be used to test communication times for communication operators like reduce-scatter, all2all, allgather, etc., within a certain range (recommended: 1M to 8GB). Then, use linear regression to fit a bandwidth model with bw*eff and latency, obtaining the communication efficiency value eff and latency for each communication primitive.

Reference for nccl_test commands: [nccl_test.sh](nccl_test.sh):

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

For fitting communication efficiency and latency based on tested communication times, please refer to [nccl_fit.py](nccl_fit.py).