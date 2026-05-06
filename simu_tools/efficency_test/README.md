<p align="center">
  <a href="README.md">English</a>|
  <a href="README-zh.md">中文版本</a>
</p>

# Automatic Generation of `system.json`

This directory is the shared entrypoint for building a SimuMax machine config.

Use this path when:

- you already ran shipped examples and now need a config for a new machine
- the nearest shipped machine config is not close enough for timing analysis
- `system.miss_efficiency` is non-empty for the target workload

Do not start here if you only need:

- a first smoke test
- a rough OOM screen
- a quick check on a machine that is already close to a shipped config

For the retained public B200 path, there is also a dedicated wrapper:

- `tools/b200/build_current_machine_system_config.py`

That helper uses this efficiency directory and keeps the B200 shape sweep and
CE/permute supplementation aligned.

SimuMax timing quality depends on two inputs from the machine side:

- operator efficiency under real shapes
- communication bandwidth / latency for the target topology

If you only need a rough OOM screen, an existing system config may be enough. If you want to explain `perf vs simulator` or `perf vs real` timing, missing efficiency should be treated as a blocker.

## Typical workflow

Generating a SimuMax-ready `system.json` usually has two steps:

1. Measure operator efficiency for the target shapes.
2. Fit communication bandwidth / latency and write them into the system file.

See the schema description in [docs/system.md](../../docs/system.md).

## Before you run anything

Recommended order:

1. run a shipped `perf` example first
2. confirm the target model and shapes are roughly understood
3. only then start machine measurement

Pre-run checklist:

- run from a source checkout of this repo
- use a Python environment where SimuMax dependencies are already installed
- make sure `torch` can see the target accelerator (`cuda` or `musa`)
- make sure the runtime required by `transformer_engine`, `flash_attn`, and your accelerator backend is available
- for communication fitting, also prepare `nccl-tests` or an equivalent backend tool

## Step 1: measure operator efficiency

Edit [run.sh](./run.sh) to match the target machine, then run:

```bash
bash run.sh
```

`run.sh` bootstraps `PYTHONPATH` from the repo root, so this checkout path works without a prior `pip install -e .`.
If you want to invoke the individual Python scripts directly, use either:

```bash
pip install -e .
```

or:

```bash
PYTHONPATH=/path/to/SimuMax_dev python test_gemm_efficiency.py
```

The main scripts are:

- `test_gemm_efficiency.py`
- `test_grouped_gemm_efficiency.py`
- `test_fa_efficiency.py`
- `combine_efficiency.py`

For current B200 CUDA benchmarking, prefer the restored TE-oriented FA path:

- `test_fa_efficiency.py`

Why:

- retained B200 training cases run the TransformerEngine-backed attention path
- MLA-sensitive model families make it more important to measure the actual TE
  path than a looser generic FA wall-time microbenchmark
- the restored B200 FA script uses the `trace_kernel` timing path so the
  efficiency result tracks the kernel path used by retained training more
  closely

Important path note:

- outputs are written to the directory from which you invoke `bash run.sh`
- they are not forced to live next to `run.sh`

### What you usually need to change in `run.sh`

Most users only need to check these variables:

- `MAX_TFLOPS`: nominal peak TFLOPS of the target accelerator
- `SYS_NAME`: final output file name, written as `<SYS_NAME>.json`
- `NUM_PER_NODE`: optional override if visible devices are not the true GPUs-per-node value
- `MEM_GBS`: optional override if you do not want to trust auto-detected device memory
- `PICE_INTRA_LINK`: whether the machine should use the PCIe intra-node topology template
- `FC8_MODE`: whether the machine should start from the alternative FC8-style intra-node template
- `PARAM_FILE`: which shape-sweep definition to use

In the shipped `run.sh`, `NUM_PER_NODE` and `MEM_GBS` are left unset by
default, so the shared workflow uses auto-detection unless you explicitly
uncomment and override them.

### What `run_params.json` controls

`run_params.json` controls the shape sweep:

- model list
- `mbs`
- `seq_len`
- `tp`
- `ep`
- optional `cp`
- optional `dtype`

For a first validation run, shrink it to one model and a very small sweep. For example:

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

### What you will see while it runs

Expected behavior:

- many shape-level timing lines will be printed; this is normal
- intermediate directories appear in the current working directory, for example:
  - `<detected_device>_gemm_efficiency/`
  - `<detected_device>_grouped_gemm_efficiency/`
  - `<detected_device>_fa_efficiency/`
- if the final merge step succeeds, the final merged file is written as `<SYS_NAME>.json`

Typical duration:

- a very small sweep: minutes
- the full default sweep: much longer, often tens of minutes or more depending on machine speed and shape count

### What counts as success

Step 1 is successful when:

- the expected intermediate files for your model family exist and are non-empty
- `combine_efficiency.py` completes successfully
- `<SYS_NAME>.json` appears in the directory from which you invoked `bash run.sh`
- the final file contains measured operator entries under `accelerator.op.*.accurate_efficient_factor`

Model-family note:

- dense-only sweeps usually produce GEMM and FlashAttention outputs
- grouped GEMM output is expected only when the selected models and shapes actually exercise MoE grouped-gemm paths

Important note:

- after the final `combine_efficiency.py` step succeeds, the generated `<SYS_NAME>.json` contains merged operator-efficiency data
- the generated file is still a starter machine scaffold, not a fully timing-ready system file
- on supported CUDA/MUSA hardware, the shared workflow now tries to auto-fill `accelerator.backend`, visible `num_per_node`, and `accelerator.mem_gbs`
- you should still review `num_per_node`, `accelerator.backend`, `accelerator.mem_gbs`, `accelerator.bandwidth`, and `networks` before treating the file as accurate for timing analysis

You may still see a few default-value warnings from bootstrap config construction. For a first small sweep, that does not by itself mean your `run_params.json` is wrong.

## Step 2: fit communication bandwidth and latency

Use `nccl-tests` or an equivalent tool for your backend to measure communication primitives over a reasonable size range, then fit a bandwidth + latency model.

This directory provides:

- [nccl_test.sh](./nccl_test.sh)
- [nccl_fit.py](./nccl_fit.py)

The intended measurement closure is:

1. run `nccl-tests` or an equivalent backend tool for the collectives you care about
2. fit bandwidth / latency with the same linear convention used by SimuMax
3. write the fitted values back into the generated `<SYS_NAME>.json`

Important note:

- `nccl_test.sh` is only an example command file
- `nccl_fit.py` is a helper that shows the fitting convention; it is not yet a fully generic one-command public CLI
- today, the final write-back into `system.json` is still a manual step in the shared workflow

After fitting, update the `networks` section of `<SYS_NAME>.json`, especially:

- `networks.<group>.bandwidth.gbps`
- `networks.<group>.bandwidth.latency_us`
- `networks.<group>.bandwidth.efficient_factor`
- and, when needed, the matching `networks.<group>.op.*` entries

For most users, the most important groups are:

- intra-node high-bandwidth links
- intra-node PCIe links, if present
- inter-node links

The final timing-ready `system.json` should therefore contain both:

- measured operator efficiency from step 1
- fitted communication values written back in step 2
- reviewed machine-side fields such as `num_per_node`, `accelerator.mem_gbs`, and `accelerator.bandwidth`

## When you should re-measure

Measure your own data when:

- the machine is new
- interconnect bandwidth / latency differs from the nearest shipped config
- the target workload hits missing or fallback operator efficiencies

Practical rule:

- OOM feasibility: shipped config may be enough
- timing interpretation: fill missing efficiency first
