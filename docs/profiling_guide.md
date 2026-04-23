# Profiling a real cluster to generate a SimuMax `system.json`

SimuMax ships one calibrated system config (`configs/system/a100_pcie.json`). To add your own hardware you run two independent calibration passes on a real node (or a small slice of the cluster) and merge the outputs. All tooling lives under `simu_tools/efficency_test/`.

## Overview

A `system.json` has two calibrated regions:

1. **`accelerator.op.*`** — per-operator (matmul / grouped matmul / flash-attention SDP) shape-level efficiency measured against the card's nominal TFLOPS. Example: a single `m=4096, k=5120, n=1536` BF16 GEMM achieving 0.79 of peak.
2. **`networks.<topology>.op.*`** — per-collective (`all_reduce`, `all_gather`, `reduce_scatter`, `all2all`, `p2p`) efficiency and fixed latency fitted from NCCL timings. Cost model: `time = latency_us + scale * (N-1)/N * bytes / (gbps * efficient_factor)`.

Everything else (`sys_name`, `num_per_node`, `mem_gbs`, fallback `default` blocks) is just metadata you set once.

## Stage 1 — Per-op compute efficiency

You need one machine with GPUs of the target type, PyTorch installed, and the SimuMax repo on it.

Edit the env vars in `simu_tools/efficency_test/run.sh`:

```shell
export MAX_TFLOPS=989            # nominal BF16 TFLOPS of the card (H100 SXM5 = 989, A100 = 312)
export SYS_NAME="h100_nvlink"    # becomes sys_name in the JSON
export PICE_INTRA_LINK=0         # 0 = NVLink intra-node, 1 = PCIe
export FC8_MODE=0                # FC8 PCIe topology flag (PCIe-only)
export PARAM_FILE="./run_params.json"   # which models/shapes to sweep
```

`run_params.json` controls the sweep — list only models you actually plan to simulate so calibration stays fast:

```json
{
  "model_list": ["llama3-70b", "qwen3-32b"],
  "mbs_list":   [1, 2, 4],
  "seq_len_list": [4096],
  "tp_list":    [1, 2, 4, 8],
  "ep_list":    [1]
}
```

Then run:

```shell
bash simu_tools/efficency_test/run.sh
```

Internally this invokes four scripts in order:

| Script | Writes |
|---|---|
| `test_gemm_efficency.py` | `<SYS_NAME>_gemm_efficency/gemm_efficency.json` |
| `test_grouped_gemm_efficency.py` | `<SYS_NAME>_grouped_gemm_efficency/grouped_gemm_efficency.json` |
| `test_fa_efficency.py` | `<SYS_NAME>_fa_efficency/fa_efficency_test.json` |
| `combine_efficency.py` | `<SYS_NAME>.json` (final merged file) |

`combine_efficency.py`:

- Aggregates the three per-op JSONs into `accelerator.op`.
- Computes an average BF16 efficiency across all measured shapes and writes it to `accelerator.op.default` — this is the fallback when a shape is not in the measured set.
- Attaches a **templated** `networks` block: when `PICE_INTRA_LINK=0` it includes `low_intra_node` / `high_intra_node` / `inter_node`; when `PICE_INTRA_LINK=1` it includes `intra_node_pcie_{2,4,8}x` / `inter_node`. The numbers in this block are defaults — Stage 2 replaces them for your specific cluster.

Known limits (`simu_tools/efficency_test/README.md:74-79`):

- MoE models: only TP=1 supported by the auto-tester.
- Dense models: full coverage.
- Timing uses Python-side timestamps; CUDA-event timing is a roadmap item, so treat numbers within a few percent.

## Stage 2 — Communication bandwidth & latency fit

NCCL efficiency depends on topology (NVLink vs PCIe switch, NDR vs HDR IB, rail count, switch generation), so defaults from Stage 1 must be replaced for any new cluster.

### 2.1 Run `nccl-tests`

Install the `nccl-tests` repo and run something like `simu_tools/efficency_test/nccl_test.sh`:

```shell
end=8G
./build/all_reduce_perf     -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_all_reduce.txt
./build/all_gather_perf     -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_all_gather.txt
./build/reduce_scatter_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_reduce_scatter.txt
./build/alltoall_perf       -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_alltoall.txt
./build/sendrecv_perf       -n  1 -b 1M -e $end -f 2 -g 1 -t 1  -d bfloat16 > send_recv.txt
```

Run once with all 8 GPUs of one node (that's your `high_intra_node` fit) and once across nodes with one GPU per node (that's your `inter_node` fit).

### 2.2 Fit efficiency and latency

Open `simu_tools/efficency_test/nccl_fit.py`. At the top, set:

```python
DEVICE_BW = 900   # nominal per-GPU bandwidth in GB/s (H100 NVLink = 900, A100 NVLink = 300)
```

Paste each nccl-tests output table into a `get_bws(...)` call and run. For every collective it prints:

```
all_reduce_nccl_test Estimated Bandwidth: 475.32 GB/s, Efficiency: 0.5281, Latency per P2P: 8.74 μs
```

The `scale` and `ranks` args must match SimuMax's cost formula:

| Collective | `scale` | `ranks` |
|---|---|---|
| all_reduce | 2 | comm group size (8 for intra-node) |
| all_gather | 1 | comm group size |
| reduce_scatter | 1 | comm group size |
| all2all | 1 | comm group size |
| sendrecv (p2p) | 1 | 2 |

### 2.3 Plug the numbers into `system.json`

For each topology block you want to populate, replace:

```json
"all_reduce": {
    "scale": 2,
    "offset": -1,
    "efficient_factor": <fitted eff>,
    "latency_us": <fitted latency>
}
```

Topology names SimuMax expects (picked automatically by `analysis_pcie_net` / `analysis_high_link_net` based on comm-group size):

| Label | When used |
|---|---|
| `high_intra_node` | NVLink / NVSwitch / NVLink-Switch |
| `low_intra_node` | fallback for smaller intra-node comm groups on NVLink boxes |
| `intra_node_pcie_8x` / `4x` / `2x` | PCIe boxes, keyed by comm-group size |
| `inter_node` | anything crossing NIC boundaries (IB, RoCE, Ethernet) |

For PCIe all-gather / reduce-scatter you can also attach `dp_fixed_bw` overrides keyed by comm-group size — see `configs/system/a100_pcie.json:308-323`.

## Stage 3 — Validate against a real run

Once the JSON is in place, compare simulated vs. measured end-to-end iter time on a known workload. SimuMax ships `simu_tools/megatron_scripts/` with matching Megatron-LM harnesses; that's how the bundled A100 numbers were validated. A ±10% gap on iter time is the usual target.

## Ship it

1. Copy your fitted `<SYS_NAME>.json` into `configs/system/`.
2. Reference it from a strategy run via `SystemConfig.init_from_config_file(get_simu_system_config("<sys_name>"))`.
3. Re-run your analysis script — the default fallback will still be used for any un-measured shape, so expanding `run_params.json` later only improves accuracy.

## Incremental calibration shortcut

If you can't run Stage 1 (no access, no PyTorch, short on time), you can still get a usable config by:

1. Copying an existing JSON (e.g. `configs/system/a100_pcie.json`).
2. Swapping `default` TFLOPS + average efficient factor for your card.
3. Replacing only the `networks` block with values from Stage 2 (or from spec sheets as a coarse starting point).

The shape-level `accurate_efficient_factor` entries will then be ignored when your model's shapes don't match, and SimuMax falls back to `default`. Expect larger errors on attention-heavy and MoE workloads than on dense GEMMs, since SDP and grouped-GEMM efficiency vary a lot by shape.
