# Guided Tutorial

SimuMax relies on three input files:

- a **system** config
- a **strategy** config
- a **model** config

See also:

- [system.md](./system.md)
- [strategy.md](./strategy.md)
- [model.md](./model.md)

Recommended first-time order:

1. Run a shipped `perf` example.
2. Try the minimal `PerfLLM` API.
3. Copy the nearest existing `model`, `strategy`, and `system` configs.
4. If needed, search feasible batch settings or parallel strategies.
5. Only then move to machine measurement or simulator deep dives.

## 1. Run perf with an existing config

If you only want a quick smoke test or rough OOM feasibility, start here. You do not need to measure a new machine first.

## 2. Minimal `PerfLLM` API

```python
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM

perf = PerfLLM()
perf.configure(
    strategy_config=StrategyConfig.init_from_config_file("configs/strategy/tp1_pp2_dp4_mbs1.json"),
    model_config=ModelConfig.init_from_config_file("configs/models/llama3-8b.json"),
    system_config=SystemConfig.init_from_config_file("configs/system/a100_pcie.json"),
)

perf.run_estimate()
mem_result = perf.analysis_mem()
cost_result = perf.analysis_cost()
```

`run_estimate()` builds the modeled training graph and prepares both cost and memory analysis.

## 3. Run a shipped example

```bash
cd examples
python perf_llama3_8b_tp1_pp2.py
```

If you run this from `examples/`, the script creates a result directory in the current working directory.
Typical files include:

- core outputs:
  - `compute_result.json`
  - `mem_result.json`
- helper files for understanding or reproduction:
  - `base_info.json`
  - `model_arch`
  - `model_config.json`
  - `strategy_config.json`
  - `system_config.json`
  - `net_info.json`

Practical reading:

- start with `compute_result.json` and `mem_result.json`
- use the `*_config.json` files when you want to reproduce or compare the exact run
- use `base_info.json` and `model_arch` when you want to inspect the modeled architecture

## 4. Know when shipped configs are enough

You can usually use shipped configs directly when:

- the target machine is close to a provided example machine
- the model uses already-covered dominant operator shapes
- the goal is quick exploration, not benchmark-grade timing

You should measure your own machine or your own missing shapes when:

- the hardware is new
- communication bandwidth/latency is unknown
- the target model uses shapes not covered by the current system config
- `system.miss_efficiency` is non-empty and you want to explain timing

Practical rule:

- for OOM feasibility, missing efficiency may still be acceptable
- for `perf vs simulator` or `perf vs real` timing interpretation, fill missing efficiency first

## 5. Add your own machine config

The shared measurement workflow lives under:

- [simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md)

This path covers:

- operator-efficiency measurement
- communication fitting
- assembling a SimuMax-ready `system.json`

The final useful output of that workflow is a new `system.json` that contains:

- measured operator-efficiency entries
- fitted communication values written back into `networks`

If you only want smoke testing or OOM screening, do not start with this step. Reuse the nearest shipped system config first.

## 6. Add your own model config

Model config files live under:

- [configs/models](../configs/models)

See:

- [model.md](./model.md)

Recommended path:

1. copy the nearest existing JSON under `configs/models/`
2. change only the fields that are structurally different
3. pair it with a known-good strategy and system config first

## 7. Generate simulator trace and optional memory artifacts

`simulate()` exports simulator-side artifacts that help explain timing and memory behavior:

```python
perf.simulate("tmp/llama3_8b_a100_trace")
```

This always writes `tracing_logs.json`. Depending on the strategy and current
memory-timeline path, you may also see:

- `simu_memory_result.json`
- `simu_memory_snapshot.json`
- `simu_memory_viz_snapshot.pickle`

For a simple public example that usually includes the memory artifacts,
start from a shipped single-stage strategy such as
`configs/strategy/tp2_pp1_dp4_mbs1.json`:

```python
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM

perf = PerfLLM()
perf.configure(
    strategy_config=StrategyConfig.init_from_config_file("configs/strategy/tp2_pp1_dp4_mbs1.json"),
    model_config=ModelConfig.init_from_config_file("configs/models/llama3-8b.json"),
    system_config=SystemConfig.init_from_config_file("configs/system/a100_pcie.json"),
)
perf.run_estimate()
perf.simulate("tmp/llama3_8b_a100_trace")
```

This example uses `pp_size == 1`, so it will export the memory artifacts as
well as `tracing_logs.json`. More generally, the current implementation exports
memory artifacts when `pp_size == 1` or `pp_comm_async == false`.

Use these when:

- comparing simulator behavior against real traces
- understanding where peak memory comes from
- debugging pipeline / VPP / recompute lifetime issues

## 8. Search batch settings or parallel strategies

Start with the lighter search first:

1. fix `global_batch_size`
2. search feasible `micro_batch_size` / `micro_batch_num`
3. only then search a small `tp/pp` space

Public runnable example:

- [examples/search_strategy_llama3_8b.py](../examples/search_strategy_llama3_8b.py)

### Search feasible batch settings under a fixed global batch size

```python
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM

perf = PerfLLM()
perf.configure(
    strategy_config=StrategyConfig.init_from_config_file("configs/strategy/tp1_pp2_dp4_mbs1.json"),
    model_config=ModelConfig.init_from_config_file("configs/models/llama3-8b.json"),
    system_config=SystemConfig.init_from_config_file("configs/system/a100_pcie.json"),
)

perf.model_config.padded_vocab_size = True
perf.model_config.make_vocab_size_divisible_by = 128
perf.strategy.enable_recompute = False
perf.strategy.recompute_granularity = None
perf.strategy.recompute_layer_num = 0

all_mbs, all_mbn, all_peak_mem, all_cost = perf.search_max_micro_batch_size_fixed_gbs(
    pp_size=perf.strategy.pp_size,
    dp_size=perf.strategy.dp_size,
    global_batch_size=32,
    gmi_error=10,
    use_reserved_memory=True,
    save_all=False,
    verbose=False,
)
```

This is the best first search for most users, because it only changes batching
and keeps the parallel strategy fixed.

### Search a small parallel-strategy space

```python
all_search_result = {}
best_strategy = perf.search_best_parallel_strategy(
    world_size=8,
    gmi_error=10,
    micro_batch_size=1,
    global_batch_size=32,
    all_search_result=all_search_result,
    tp_search_list=[1, 2],
    ep_search_list=[1],
    pp_search_list=[2],
    recompute_search_type=["no_recompute"],
    use_reserved_memory=True,
    dump_path=None,
    verbose=False,
)
```

`gmi_error` is a simple per-rank memory safety margin in GiB. Use it to leave
room for NCCL buffers, allocator/runtime overhead, and other components that
are not modeled explicitly in the strategy search. For a first search on a new
machine, `10` is a reasonable conservative starting point.

Recommended practice:

- start from the nearest shipped strategy
- keep `ep=1` for dense models
- search a small legal space first
- expand recompute or VPP only after the basic search is stable
