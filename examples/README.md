# SimuMax Examples

This directory contains examples that use the shipped configs under `configs/`.

For the basic `PerfLLM` and `simulate()` API flow, see
[`docs/tutorial.md`](../docs/tutorial.md). This page only records the runnable
example entrypoints and the simulator artifact fields that users usually parse.

## Perf Examples

Run a standard perf example from this directory:

```bash
python perf_llama3_8b_tp1_pp2.py
```

The perf examples call `run_estimate()` and write analytical cost and memory
outputs such as `compute_result.json` and `mem_result.json`.

## Simulator Trace And Snapshot Example

Run the small simulator example:

```bash
python simulator_trace_snapshot.py
```

The example uses `llama2-tiny`, trims it to two layers for speed, runs
`run_estimate()`, then calls `simulate()` and prints a short summary of the
exported artifacts. Pass `--output DIR` to choose a different artifact
directory.

### Trace

`tracing_logs.json` is always exported. It is a Chrome trace style event list
that can be opened by Perfetto or Chrome tracing.

Common parsing fields:

- `ph == "X"`: duration event.
- `ts`, `dur`: start timestamp and duration in microseconds.
- `pid`: simulated rank, for example `rank0`.
- `tid`: display lane, such as forward compute, backward compute, comm, wait,
  or pipeline detail.
- `name`: operation name.
- `cat`: broad category when available.
- `args.call_stack`: model/module context.
- `args.base_name`, `args.direction`, `args.stream_type`: useful grouping keys.

Minimal trace parser:

```python
events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
slices = [event for event in events if event.get("ph") == "X"]
duration_us = max(event["ts"] + event.get("dur", 0.0) for event in slices)
comm_slices = [
    event for event in slices
    if event.get("cat") == "comm" or event.get("args", {}).get("stream_type") == "comm"
]
```

### Snapshot

The current simulator memory timeline exports memory artifacts for `pp_size == 1`
or for the sync pipeline memory path (`pp_comm_async == false`). This example
uses `pp_size == 1`, so it writes:

- `simu_memory_result.json`: compact static and peak allocated memory by rank.
- `simu_memory_snapshot.json`: detailed memory timeline.
- `simu_memory_viz_snapshot.pickle`: memory-visualization adapter.

`simu_memory_snapshot.json` currently uses schema
`simumax_memory_snapshot_v1`.

Common snapshot fields:

- `rank`, `ts_us`: simulated rank and timestamp.
- `static_bytes`: long-lived model memory.
- `cached_bytes`: live activation cache retained for backward.
- `temp_bytes`: op-local temporary peak memory.
- `allocated_bytes`: `static_bytes + cached_bytes + temp_bytes`.
- `phase`, `op_name`: lifecycle phase and associated operation.
- `microbatch`, `chunk`: data microbatch and VPP chunk metadata when present.
- `cache_tokens`: activation-cache lifetime records.

Minimal snapshot parser:

```python
events = snapshot["events"]
peak = max(events, key=lambda event: event["allocated_bytes"])
alloc_tokens = [
    token for token in snapshot.get("cache_tokens", [])
    if token.get("action") == "alloc"
]
```
