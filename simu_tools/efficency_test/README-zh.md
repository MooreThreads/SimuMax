<p align="center">
  <a href="README.md">English</a>|
  <a href="README-zh.md">中文版本</a>
</p>

# 自动生成 `system.json`

这个目录是 SimuMax 共享、对外安全的机器配置生成入口。

适合走这条路径的场景：

- 已经跑通过现成 example，现在需要给新机器生成 system 配置
- 最近的已有机器配置不足以支撑 timing 分析
- 目标 workload 命中了 `system.miss_efficiency`

不适合一上来就走这条路径的场景：

- 只是第一次 smoke test
- 只是做粗略 OOM feasibility 判断
- 目标机器已经和现有配置非常接近

SimuMax 的 timing 准确性主要依赖两类机器侧数据：

- 真实 shape 下的算子效率
- 目标拓扑下的通信带宽与 latency

如果只是做粗略 OOM 判断，已有 system 配置通常够用；如果要解释 `perf vs simulator` 或 `perf vs real` 的 timing，缺失 efficiency 应该优先补齐。

## 典型流程

生成符合 SimuMax 口径的 `system.json` 一般分两步：

1. 测试目标 shape 的算子效率
2. 拟合通信带宽与 latency，并写回 system 文件

system 文件字段说明见：

- [docs/system.md](../../docs/system.md)

## 运行前先确认

推荐顺序：

1. 先跑通现成 `perf` example
2. 确认目标模型和 shape 大致合理
3. 再进入机器测量流程

运行前检查表：

- 在本仓源码 checkout 中执行
- 使用已经装好 SimuMax 依赖的 Python 环境
- 确认 `torch` 能看到目标加速器（`cuda` 或 `musa`）
- 确认当前环境已经具备 `transformer_engine`、`flash_attn` 和对应 backend runtime
- 如果要做通信拟合，还需要准备 `nccl-tests` 或目标 backend 的等价工具

## 第一步：测试算子效率

先修改 [run.sh](./run.sh) 里的目标机器参数，然后运行：

```bash
bash run.sh
```

`run.sh` 会自动从仓库根目录补齐 `PYTHONPATH`，因此直接在源码 checkout 中运行即可，不要求先执行 `pip install -e .`。
如果要单独调用这些 Python 脚本，请先执行：

```bash
pip install -e .
```

或者：

```bash
PYTHONPATH=/path/to/SimuMax_dev python test_gemm_efficiency.py
```

主要脚本包括：

- `test_gemm_efficiency.py`
- `test_grouped_gemm_efficiency.py`
- `test_fa_efficiency.py`
- `combine_efficiency.py`

关于输出路径，需要特别注意：

- 结果会写到你执行 `bash run.sh` 时所在的当前目录
- 并不一定写在 `run.sh` 同目录下

### `run.sh` 里通常需要关注的参数

大部分用户只需要检查这几个环境变量：

- `MAX_TFLOPS`：目标加速器的标称峰值算力
- `SYS_NAME`：最终输出文件名，会写成 `<SYS_NAME>.json`
- `NUM_PER_NODE`：可选覆盖项；如果当前可见设备数不等于真实单机卡数，手动指定
- `MEM_GBS`：可选覆盖项；如果不想直接使用自动探测到的显存大小，可以手动指定
- `PICE_INTRA_LINK`：是否使用 PCIe 机内拓扑模板
- `FC8_MODE`：是否使用 FC8 风格的机内拓扑模板
- `PARAM_FILE`：指向 shape 扫描定义文件

仓库自带的 `run.sh` 默认不会设置 `NUM_PER_NODE` 和 `MEM_GBS`，因此共享流程会优先走自动探测；只有你显式取消注释并赋值时，才会覆盖自动探测结果。

### `run_params.json` 控制什么

`run_params.json` 控制 shape 扫描范围：

- model list
- `mbs`
- `seq_len`
- `tp`
- `ep`
- 可选 `cp`
- 可选 `dtype`

第一次建议先缩成一份很小的 sweep，例如：

```json
{
    "model_list": ["llama3-8b"],
    "mbs_list": [1],
    "seq_len_list": [4096],
    "tp_list": [1],
    "ep_list": [1],
    "cp_list": [1],
    "dtype": ["bf16"]
}
```

这样更容易先判断流程是否跑对。

### 运行中你会看到什么

正常现象包括：

- 屏幕上会打印大量 shape 级别测试输出，这是正常的
- 当前目录下会生成一些中间目录，例如：
  - `<detected_device>_gemm_efficiency/`
  - `<detected_device>_grouped_gemm_efficiency/`
  - `<detected_device>_fa_efficiency/`
- 如果最后的 merge 步骤成功，会写出 `<SYS_NAME>.json`

大致耗时：

- 很小的 sweep：几分钟
- 默认完整 sweep：会长很多，常见是几十分钟甚至更久，取决于机器和 shape 数量

### 什么状态说明第一步成功

第一步成功时应看到：

- 对当前模型族而言应出现的中间 `*_efficiency.json` 文件已经生成且非空
- `combine_efficiency.py` 已成功完成
- 在你执行 `bash run.sh` 的当前目录下看到了最终 `<SYS_NAME>.json`
- 最终文件里的 `accelerator.op.*.accurate_efficient_factor` 已写入测得的算子效率

模型类型说明：

- dense-only sweep 一般会产出 GEMM 和 FlashAttention 结果
- grouped GEMM 结果只有在所选模型和 shape 真正命中 MoE grouped-gemm 路径时才是必需的

需要特别注意：

- 最后的 `combine_efficiency.py` 成功后，`<SYS_NAME>.json` 才会包含合并后的算子效率结果
- 这个文件仍然只是起步 machine scaffold，不应直接视为完整的 timing-ready system
- 在支持的 CUDA/MUSA 硬件上，共享流程现在会尽量自动补上 `accelerator.backend`、当前可见 `num_per_node` 和 `accelerator.mem_gbs`
- 但在做 timing 分析前，仍然需要人工确认 `num_per_node`、`accelerator.backend`、`accelerator.mem_gbs`、`accelerator.bandwidth` 和 `networks`

另外，第一次小规模 sweep 时仍可能看到少量默认值 warning。这并不一定表示你的 `run_params.json` 写错了。

## 第二步：拟合通信带宽与 latency

使用 `nccl-tests` 或目标后端上的等价工具测试通信原语，再拟合 bandwidth + latency 模型。

本目录提供：

- [nccl_test.sh](./nccl_test.sh)
- [nccl_fit.py](./nccl_fit.py)

公共口径下，这一步的闭环是：

1. 用 `nccl-tests` 或等价工具测 collectives
2. 按 SimuMax 使用的同类线性模型拟合 bandwidth / latency
3. 把拟合结果手动写回 `<SYS_NAME>.json`

这里要注意：

- `nccl_test.sh` 只是一个示例命令文件
- `nccl_fit.py` 主要是展示拟合方法的 helper，不是完全通用的一键 CLI
- 所以当前共享流程里，“把拟合结果写回 system.json” 仍然是人工步骤

拟合完成后，主要需要更新 `<SYS_NAME>.json` 里的 `networks`：

- `networks.<group>.bandwidth.gbps`
- `networks.<group>.bandwidth.latency_us`
- `networks.<group>.bandwidth.efficient_factor`
- 如果你还拟合了不同 collective 的差异，再更新对应的 `networks.<group>.op.*`

通常最先需要补齐的是：

- 机内高速链路
- 机内 PCIe 链路（如果存在）
- 机间链路

最终可用于 timing 分析的 `system.json`，应该同时包含：

- 第一步生成的算子效率
- 第二步写回的通信参数
- 以及已经确认过的机器侧字段，例如 `num_per_node`、`accelerator.mem_gbs`、`accelerator.bandwidth`

## 什么时候需要重新实测

以下情况建议做自己的机器实测：

- 机器是新的
- 通信拓扑和最近的已有 system 差异明显
- 目标 workload 命中了缺失或 fallback 的算子效率

经验规则：

- 只看 OOM feasibility 时，可以先使用已有配置
- 要解释 timing 时，先补齐 missing efficiency
