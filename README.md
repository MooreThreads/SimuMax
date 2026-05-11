# SimuMax: a static analytical model for LLM distributed training

SimuMax is a simulator and analytical model for large-scale LLM training. It models compute, communication, and memory without launching a full training job, so users can estimate throughput, peak memory, traces, and pipeline behavior before running real workloads.

## Highlights

- v1.2 release notes:
  [docs/release_v1.2.md](docs/release_v1.2.md)
- B200 public benchmark materials include a formal v1.2 summary,
  reproduction guide, CP A2A result chart, and public B200 system config.
- Long-context context-parallel (CP) coverage is included in the public
  B200 release results.
- Megatron-LM 0.14 selective recompute with `discard_output` semantics is
  modeled through `megatron_recompute`.
- sync-VPP is available as Preview with a representative B200 preview check.
- Supports dense and MoE models, including TP / PP / EP / CP / SP / ZeRO-1 /
  recompute / MLA.
- Can be used in five core ways:
  - run `perf` with an existing system / strategy / model config
  - add a machine config for your own cluster
  - add a model config for your own architecture
  - search feasible batch settings or parallel strategies
  - generate simulator trace and memory snapshots for understanding and debugging

## Features

- Analytical `perf` modeling for throughput, MFU, and stage-level memory without launching a full training job.
- Context-parallel and long-context modeling, including B200 CP A2A validation
  in the v1.2 public benchmark set.
- sync-VPP support is available as Preview for representative validation and
  simulator/perf comparison workflows.
- Simulator support for pipeline schedules, traces, and memory snapshots, so modeled behavior can be compared against real runs.
- Coverage for dense and MoE training features, including TP / PP / EP / CP /
  SP / ZeRO-1 / recompute / MLA.
- Megatron-LM 0.14 style selective recompute can be represented with
  `megatron_recompute=true` and explicit `megatron_recompute_modules`.
- Config-driven workflow built around three inputs:
  - `system`: machine capability, bandwidth, latency, and operator efficiency
  - `strategy`: parallelism and runtime policy
  - `model`: architecture description
- Strategy-search helpers for micro-batch settings and small parallel-strategy sweeps.
- Public B200 and A100 benchmark summaries, plus shipped system configs and
  user-measured machine-config workflows.

## B200 Public Materials

- B200 CP A2A result graph:
  [assets/b200_cp_a2a_release_v1.2.png](assets/b200_cp_a2a_release_v1.2.png)
- Current canonical B200 summary:
  [docs/b200/b200_release_v1.2_summary.md](docs/b200/b200_release_v1.2_summary.md)
- Current B200 reproduction guide:
  [docs/b200/b200_real_repro_guide.md](docs/b200/b200_real_repro_guide.md)

Public benchmark summaries remain in [docs/FULL_RESULTS.md](docs/FULL_RESULTS.md).

## Getting Started

### Installation

We recommend using a dedicated virtual environment before installing SimuMax:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install the repo:

```bash
git clone git@github.com:MooreThreads/SimuMax.git
cd SimuMax
pip install -r requirements.txt
pip install -v -e .
```

### Recommended First-Time Path

If you are new to SimuMax, the shortest path is:

1. Run a shipped `perf` example.
2. Read [docs/tutorial.md](docs/tutorial.md) and try the `PerfLLM` API directly.
3. Only after that, start copying and editing your own `model`, `strategy`, and `system` configs.

If your goal is only smoke checking or rough OOM feasibility, you can usually stop after step 1 or 2.
The machine-measurement path is slower and is mainly for users who care about timing accuracy.

### Start With the App

- Public/default app entry: [app/README.md](app/README.md)

### User Docs

- [docs/README.md](docs/README.md)
- [docs/tutorial.md](docs/tutorial.md)
- [docs/system.md](docs/system.md)
- [docs/model.md](docs/model.md)
- [docs/strategy.md](docs/strategy.md)

## Usage Paths

### 1. Run perf with an existing config

The simplest path is still the shipped example set under [examples/](examples) and [configs/](configs).

Example:

```bash
cd examples
python perf_llama3_8b_tp1_pp2.py
```

### 2. Add your own machine/system config

The shared machine-measurement entrypoint is:

- [simu_tools/efficency_test/README.md](./simu_tools/efficency_test/README.md)

This path is the shared place for operator-efficiency and communication-fitting workflows.
This is a real measurement workflow, so expect it to take noticeably longer than running a shipped example. Use it when timing accuracy matters, not for the first smoke test.

### 3. Add your own model config

Model config files live under:

- [configs/models](configs/models)

Field-level guidance is documented in:

- [docs/model.md](docs/model.md)

### 4. Use simulator trace and optional memory artifacts

`simulate()` always exports `tracing_logs.json`. For example:

```python
perf.simulate("tmp/llama3_8b_a100_trace")
```

Depending on the strategy and current memory-timeline path, you may also see:

- `simu_memory_result.json`
- `simu_memory_snapshot.json`
- `simu_memory_viz_snapshot.pickle`

See:

- [examples/README.md](examples/README.md) for a runnable simulator example
  and trace / snapshot parsing notes
- [docs/tutorial.md](docs/tutorial.md) for the basic `simulate()` usage path

These artifacts are useful both for understanding SimuMax’s modeling logic and for comparing simulator behavior against real traces and real memory evidence.

### 5. Search batch settings or parallel strategies

If you already have a model / strategy / system triple and want SimuMax to
search feasible settings instead of guessing them by hand, start with:

- [examples/search_strategy_llama3_8b.py](examples/search_strategy_llama3_8b.py)
- [docs/tutorial.md](docs/tutorial.md)

Recommended order:

1. search `micro_batch_size` / `micro_batch_num` under a fixed `global_batch_size`
2. only then expand into a small `tp/pp` strategy sweep

## When You Must Measure Your Own Data

Using shipped configs is fine for demos, smoke tests, and some already-covered machines and shapes. But users should measure their own machine data when:

- the machine is new or materially different from the nearest shipped config
- interconnect bandwidth / latency is unknown or likely different
- the target model introduces dominant matmul / grouped-gemm / attention shapes not covered by the current system config
- `system.miss_efficiency` is non-empty and the goal is to explain timing, not just OOM feasibility

Practical rule:

- For OOM feasibility screening, missing efficiency may be acceptable.
- For `perf vs simulator` or `perf vs real` timing analysis, fill missing efficiency first.

## Benchmark Scripts

- Public/reference Megatron benchmark scripts:
  - [simu_tools/megatron_scripts/README.md](./simu_tools/megatron_scripts/README.md)

## Notes

- Currently, all Linear models are forced to perform gradient accumulation fusion.

## Testing

Typical local validation:

```bash
python -m compileall -q simumax app tools examples simu_tools/efficency_test simu_tools/megatron_scripts/*.py
PYTHONPATH=. python examples/perf_llama3_8b_tp1_pp2.py
```

## Roadmap

SimuMax is still evolving. Planned or ongoing areas include:

- more pipeline schedulers beyond the current sync-VPP Preview surface
- compute/communication overlap
- offloading
- larger public strategy-search examples
- tighter memory-bound operator modeling for more kernels and machine targets

## Acknowledgements

Some aspects of the design and interfaces were inspired by [Calculon](https://github.com/calculon-ai/calculon). We appreciate the work done by the authors of that repository, which provided helpful references during development.

## Community

### Issue Reporting

If you find any problems in SimuMax, please open an issue.

### Contributions

Contributions of code, model implementations, and documents are welcome.

### Join Our Team

If you're passionate about:

- large-scale MoE / RL / multimodal systems
- GPU and GPU-cluster training and inference optimization

feel free to reach out to `xuerong.huang@mthreads.com`.
