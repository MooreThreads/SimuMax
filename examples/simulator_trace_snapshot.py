import argparse
import json
from pathlib import Path

from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


def build_perf_model():
    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config("tp2_pp1_dp4_mbs1")
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config("llama2-tiny")
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config("a100_pcie")
        ),
    )

    # Keep the public simulator example quick while still exercising fwd/bwd,
    # recompute, trace export, and memory snapshot export.
    perf.model_config.model_name = "llama2-tiny-2layer"
    perf.model_config.layer_num = 2
    return perf


def load_trace_events(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("traceEvents", [])
    return payload


def summarize_trace(trace_path):
    events = load_trace_events(trace_path)
    slices = [event for event in events if event.get("ph") == "X"]
    ranks = sorted({event.get("pid") for event in slices})
    end_us = max(
        (event.get("ts", 0.0) + event.get("dur", 0.0) for event in slices),
        default=0.0,
    )
    comm_slices = [
        event
        for event in slices
        if event.get("cat") == "comm"
        or event.get("args", {}).get("stream_type") == "comm"
    ]
    compute_slices = [event for event in slices if event.get("cat") == "compute"]

    return {
        "event_count": len(events),
        "slice_count": len(slices),
        "compute_slice_count": len(compute_slices),
        "comm_slice_count": len(comm_slices),
        "rank_count": len(ranks),
        "duration_ms": end_us / 1000.0,
    }


def summarize_snapshot(snapshot_path):
    with open(snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    events = snapshot.get("events", [])
    peak_event = max(
        events, key=lambda event: event.get("allocated_bytes", 0), default={}
    )
    cache_tokens = snapshot.get("cache_tokens", [])
    alloc_tokens = [token for token in cache_tokens if token.get("action") == "alloc"]
    free_tokens = [token for token in cache_tokens if token.get("action") == "free"]

    return {
        "schema": snapshot.get("schema"),
        "event_count": len(events),
        "cache_token_alloc_count": len(alloc_tokens),
        "cache_token_free_count": len(free_tokens),
        "peak_rank": peak_event.get("rank"),
        "peak_op": peak_event.get("op_name"),
        "peak_phase": peak_event.get("phase"),
        "peak_allocated_gib": peak_event.get("allocated_bytes", 0) / (1024**3),
    }


def print_summary(save_path):
    trace_path = save_path / "tracing_logs.json"
    memory_result_path = save_path / "simu_memory_result.json"
    snapshot_path = save_path / "simu_memory_snapshot.json"
    memory_viz_path = save_path / "simu_memory_viz_snapshot.pickle"

    print(f"trace: {trace_path}")
    print(json.dumps(summarize_trace(trace_path), indent=2))

    if memory_result_path.exists():
        print(f"memory result: {memory_result_path}")
        print(memory_result_path.read_text(encoding="utf-8"))

    if snapshot_path.exists():
        print(f"memory snapshot: {snapshot_path}")
        print(json.dumps(summarize_snapshot(snapshot_path), indent=2))

    if memory_viz_path.exists():
        print(f"memory viz snapshot: {memory_viz_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small SimuMax simulator trace and memory snapshot example."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulator_llama2_tiny_a100"),
        help="Directory for simulator artifacts.",
    )
    parser.add_argument(
        "--no-merge-lanes",
        action="store_true",
        help="Export one simulator lane per world rank instead of PP representative ranks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    perf = build_perf_model()

    perf.run_estimate()
    perf.simulate(str(args.output), merge_lanes=not args.no_merge_lanes)
    print_summary(args.output)


if __name__ == "__main__":
    main()
