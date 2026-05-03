# SimuMax Docs

This folder holds the public documentation for SimuMax.

Recommended first-time order:

1. Run a shipped `perf` example in [tutorial.md](./tutorial.md).
2. Use the same tutorial to try the `PerfLLM` API directly.
3. Copy the nearest existing `model`, `strategy`, and `system` JSONs before creating anything from scratch.
4. If needed, search feasible batch settings or parallel strategies.
5. Only enter the machine-measurement workflow when you need timing accuracy on a new machine.

Use this index by task:

## 1. Run perf with an existing config

- [tutorial.md](./tutorial.md)
- [configs/models](../configs/models)
- [configs/strategy](../configs/strategy)
- [configs/system](../configs/system)

## 2. Add your own machine/system config

- [system.md](./system.md)
- [system-zh.md](./system-zh.md)
- [simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md)

Use the shipped config directly only when the target machine and dominant operator shapes are already covered. If the machine is new or `system.miss_efficiency` is non-empty for the target case, measure your own data before interpreting timing.

## 3. Add your own model config

- [model.md](./model.md)
- [model-zh.md](./model-zh.md)
- [strategy.md](./strategy.md)
- [strategy-zh.md](./strategy-zh.md)

## 4. Generate simulator trace and memory artifacts

- [tutorial.md](./tutorial.md)

Simulator artifacts are meant to help users understand stage-level timing, memory peaks, and cache lifetime, and to compare those modeled results against real traces or real memory evidence.

If you only need a smoke test or rough OOM estimate, you do not need to start with trace or snapshot generation.

## 5. Search batch settings or parallel strategies

- [tutorial.md](./tutorial.md)
- [strategy.md](./strategy.md)
- [examples/search_strategy_llama3_8b.py](../examples/search_strategy_llama3_8b.py)

Recommended order:

1. search `micro_batch_size` / `micro_batch_num` first
2. only then expand into a small `tp/pp` strategy sweep

## Benchmarks

- Release notes: [release_v1.2.md](./release_v1.2.md)
- Public benchmark summary: [FULL_RESULTS.md](./FULL_RESULTS.md)

## B200 Public Materials

- [docs/b200/README.md](./b200/README.md)
- [B200 formal summary](./b200/b200_release_v1.2_summary.md)
- [B200 real reproduction guide](./b200/b200_real_repro_guide.md)
