# B200 Real Reproduction Guide

This guide captures the current B200 real-benchmark baseline used by the
public B200 result summary.

The public formal result table for this baseline is:
- `docs/b200/b200_release_v1.2_summary.md`

The goal is practical reproducibility:
- use the same container family;
- use the same external Megatron checkout;
- apply the same local patch(es);
- use the same canonical launcher scripts.

## 1. Current Local Baseline

The current machine confirms the following runtime baseline:

- OS: `Ubuntu 24.04.3 LTS`
- Python: `3.12.3`
- Torch: `2.10.0a0+a36e1d39eb.nv26.01.42222806`
- TransformerEngine: `2.11.0+c188b533`
- CUDA path used by launchers: `/usr/local/cuda-13.1`

Important:
- SimuMax B200 helper scripts set `te_version=2.11.0` for perf replay. This
  enables regular/grouped dummy-wgrad memory and the TE >= v2.8 CP A2A policy:
  attention saves pre-PostA2A O independently from the following W_o input
  cache, and backward only all-to-alls dO for the O side.
- The torch build string strongly indicates the local container belongs to the
  NVIDIA PyTorch `26.01` family.
- If your environment is managed outside the container, record the exact image
  tag from the host side as well. A common match for this runtime family is
  `nvcr.io/nvidia/pytorch:26.01-py3`, but this guide only treats the observed
  runtime above as guaranteed fact.

## 2. External Source Baseline

### Megatron-LM

Current local checkout:
- repo: `https://github.com/NVIDIA/Megatron-LM.git`
- commit: `23e00ed0963c35382dfe8a5a94fb3cda4d21e133`

Suggested reproducible checkout:

```bash
cd <repo-root>/simu_tools/megatron_scripts
rm -rf Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 23e00ed0963c35382dfe8a5a94fb3cda4d21e133
```

### TransformerEngine

Current local runtime:
- version string: `2.11.0+c188b533`

This repo does not vendor a separate TE source checkout. For real benchmark
reproduction, the minimum requirement is:
- TE importable in Python;
- TE version string matching `2.11.0+c188b533`.

Quick verification:

```bash
python - <<'PY'
import transformer_engine
print(transformer_engine.__version__)
PY
```

## 3. Required Patch(es)

### 3.1 B200 memory logging patch

Patch file:
- `simu_tools/megatron_scripts/patches/megatron_b200_memory_logging.patch`

Purpose:
- enables the env-driven all-rank / every-iter memory logging used by the
  retained formal memory contract:
  - `MEGATRON_LOG_MEMORY_ALL_RANKS`
  - `MEGATRON_REPORT_MEMORY_EVERY_ITER`
  - `MEGATRON_REPORT_MEMORY_MIN_ITER`
- keeps the memory-log behavior as a tracked external-checkout delta instead of
  relying on an undocumented dirty local Megatron tree

Apply manually:

```bash
cd <repo-root>/simu_tools/megatron_scripts/Megatron-LM
patch --quiet -p1 < ../patches/megatron_b200_memory_logging.patch
```

Revert manually:

```bash
cd <repo-root>/simu_tools/megatron_scripts/Megatron-LM
patch --quiet -R -p1 < ../patches/megatron_b200_memory_logging.patch
```

### 3.2 Fake PP warmup patch

Patch file:
- `simu_tools/megatron_scripts/patches/megatron_fake_pp_warmup.patch`

Helper script:
- `simu_tools/megatron_scripts/scripts/fake_pp_warmup_patch.sh`

Purpose:
- enables `MEGATRON_FAKE_PP_SIZE_FOR_WARMUP` for memory-only fake pipeline
  warmup experiments;
- does not change real PP communication topology.

Apply manually:

```bash
cd <repo-root>/simu_tools/megatron_scripts/Megatron-LM
patch --quiet -p1 < ../patches/megatron_fake_pp_warmup.patch
```

Revert manually:

```bash
cd <repo-root>/simu_tools/megatron_scripts/Megatron-LM
patch --quiet -R -p1 < ../patches/megatron_fake_pp_warmup.patch
```

### 3.3 Why `git apply` is used here

This repo keeps the public B200 launcher logic in tracked shell/python files,
but `Megatron-LM` itself is still an external checkout. We therefore use
checked-in patch files for external-checkout deltas that must be reproducible.

Current rule:

- use tracked launcher/wrapper scripts for repo-owned workflow behavior
- use `git apply` / `patch -p1` only for explicit external Megatron deltas

Also note the current public B200 scope intentionally avoids overlap-style real
paths:

- the retained formal results do not claim overlap timing support
- launcher defaults keep overlap-related paths off
- example knobs include:
  - `--disable-tp-comm-overlap-rs`
  - `--disable-tp-comm-overlap-ag`
  - `--no-overlap-p2p-communication`

So the public B200 workflow currently follows a non-overlap contract. This is
why the external Megatron checkout still carries tracked patch files instead of
assuming "clone upstream and run" is enough: we only claim reproducibility for
the retained non-overlap launcher path and its required external deltas.

In practice this means:

- formal retained B200 results do not depend on overlap timing behavior
- launcher-side overlap-related features are kept off by default
- when an external Megatron checkout still needs extra behavior to match the
  retained public workflow, we carry that delta as a tracked patch and apply it
  with `git apply` / `patch -p1`

Current examples are:

- all-rank / every-iter memory logging for the formal memory contract
- fake-PP warmup support for memory-only proxy experiments

### 3.4 Other local modifications

For the current retained B200 real runs, there is no second standalone
`git apply` patch file checked into this repo for the benchmark launcher path.
The remaining behavior is carried by the tracked launcher and wrapper scripts in
this repository itself.

If future work introduces another external patch dependency, add it to this
section and to `setup_b200_real_env.sh`.

## 4. Required Environment

Canonical real-benchmark launchers now assume these CUDA/Triton settings:

- `CUDA_HOME=/usr/local/cuda-13.1`
- `TRITON_PTXAS_PATH=/usr/local/cuda-13.1/bin/ptxas`
- `CPATH=/usr/local/cuda-13.1/include`
- `C_INCLUDE_PATH=/usr/local/cuda-13.1/include`
- `CPLUS_INCLUDE_PATH=/usr/local/cuda-13.1/include`

These are already exported by the canonical launcher scripts:
- `simu_tools/megatron_scripts/scripts/run_pretrain_llama3.sh`
- `simu_tools/megatron_scripts/scripts/run_pretrain_deepseekv2.sh`

Why this matters:
- `TRITON_PTXAS_PATH` is required for TE Triton fused CE in this environment;
- CUDA include paths are required because Triton launcher compilation may fail
  with `cuda.h: No such file or directory` otherwise.

## 5. Canonical Entry Points

### Real/perf orchestration

Use these repo-side wrappers as the canonical entry points:

- `tools/b200/build_current_machine_system_config.py`
- `tools/b200/run_megatron_perf_real_pipeline.py`
- `tools/b200/run_megatron_perf_real_batch.py`

### Megatron launchers

Use these launcher scripts for real runs:

- `simu_tools/megatron_scripts/run_llama3.sh`
- `simu_tools/megatron_scripts/run_deepseekv2.sh`
- `simu_tools/megatron_scripts/scripts/run_pretrain_llama3.sh`
- `simu_tools/megatron_scripts/scripts/run_pretrain_deepseekv2.sh`

These launchers carry the retained B200 workflow behavior for:
- CP launch shape and `dp` derivation;
- VPP / VP arguments;
- long sequence length arguments;
- Triton/CUDA environment hardening.
- retained non-overlap launch behavior for the current formal release scope

For single-node fresh repro:
- the default `hostfile` entry `127.0.0.1` is treated as a local launch target
  and no longer requires localhost passwordless SSH;
- launcher wrappers now propagate per-host launch failures back to the caller
  instead of returning `0` after a failed background SSH launch.

### Public B200 system config baseline

Retain one canonical public B200 system config:

- `configs/system/b200_bf16_ceperm.json`

Retain one canonical B200 one-click parameter seed:

- `tools/b200/run_params/b200_bf16_ceperm_sweep.json`

Notes:

- `tools/b200/build_current_machine_system_config.py` is the public B200
  system-config generation helper:
  - it regenerates compute efficiency with the public B200 shape sweep
  - it then applies the B200 CE/permute supplement

Important:
- the canonical `ceperm` config is not produced by one-click alone;
- it should be understood as:
  - one-click base config generation
  - plus CE/permute efficiency supplementation

## 6. Memory Measurement Contract

Current real memory results use the following contract:

- `real alloc` = maximum over all ranks of `torch.cuda.max_memory_allocated()`
- `real reserved` = maximum over all ranks of `torch.cuda.max_memory_reserved()`
- partial-rank memory logging is invalid for formal comparison

For memory refresh runs:
- use `2` steps;
- use `0` warmup steps;
- for the same topology, if only `mbc` changes, one representative real-memory
  run may be reused.

The parsing and defaults are implemented in:
- `tools/b200/run_megatron_perf_real_pipeline.py`

## 7. CP / VPP Scope Notes

### CP

Current formal CP scope keeps only:
- dense models;
- `cp_comm_type=a2a`;
- long-sequence cases (`32k`, `128k`).

`DeepSeekV2 MLA + CP a2a` is not supported by the current TE runtime.
`DeepSeekV2 MLA + CP p2p` is exploratory only and not part of formal
`perf vs real` results.

### VPP

Current agreement:
- `sync VPP` real/perf is valid;
- `async VPP` is analyzed through `real + simulator` only;
- `analysis_cost()` does not currently provide formal async-VPP perf timing.

## 8. B200 FA Efficiency Note

For B200 attention-efficiency refresh, do not treat FlashAttention as a generic
standalone wall-time microbenchmark.

Current B200 note:

- retained B200 real training uses TransformerEngine-backed attention paths
- MLA-sensitive model families make the "measure the actual TE path" requirement
  more important than on older dense-only sweeps
- the public FA benchmark path therefore uses the restored
  `simu_tools/efficency_test/test_fa_efficiency.py` workflow with
  `FA_TIMING_MODE=trace_kernel`
- CE/attention efficiency refresh for the retained B200 path therefore follows
  the actual TE runtime route used by training, instead of assuming a generic
  raw-FA wall-time proxy is good enough

Why this matters:

- the old host wall-time FA microbenchmark is too easy to skew with launcher
  overhead and does not necessarily match the retained B200 training kernel path
- the current B200 FA workflow was adjusted so the measured efficiency follows
  the actual TE-side path used by retained training cases, instead of a looser
  "raw FA only" approximation
- this is especially important for MLA-sensitive B200 cases, where "measure a
  generic FA kernel" is weaker than "measure the TE attention path we actually
  train with"

For the shared/public path, prefer the default TE-oriented route in
`test_fa_efficiency.py` unless you are explicitly debugging another backend.

## 9. Recommended Setup Procedure

If you want a fresh local setup in this repo layout:

```bash
cd <repo-root>
bash simu_tools/megatron_scripts/scripts/setup_b200_real_env.sh
```

That script:
- verifies the current runtime;
- clones/checks out Megatron-LM at the retained commit;
- applies the B200 memory-logging patch by default;
- optionally applies the fake-PP warmup patch;
- prints the next canonical launcher commands.

## 10. Canonical Result Entry

Canonical formal summary:
- `docs/b200/b200_release_v1.2_summary.md`
