"""Simulator artifact export helpers."""

from __future__ import annotations

import json
import os


def should_enable_simu_memory_timeline(strategy, vp_size: int) -> bool:
    return strategy.pp_size == 1 or (not getattr(strategy, "pp_comm_async", True))


def append_memory_events_to_trace(output_json_path, memory_tracker):
    with open(output_json_path, "r", encoding="utf-8") as f:
        tracing_events = json.load(f)
    tracing_events.extend(memory_tracker.events)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(tracing_events, f, indent=4)


def export_simu_memory_artifacts(save_path, memory_tracker, pickle_module):
    memory_summary_path = os.path.join(save_path, "simu_memory_result.json")
    with open(memory_summary_path, "w", encoding="utf-8") as f:
        json.dump(memory_tracker.summary(), f, indent=4)

    memory_snapshot_path = os.path.join(save_path, "simu_memory_snapshot.json")
    with open(memory_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(memory_tracker.snapshot(), f, indent=4)

    memory_viz_snapshot_path = os.path.join(save_path, "simu_memory_viz_snapshot.pickle")
    with open(memory_viz_snapshot_path, "wb") as f:
        pickle_module.dump(memory_tracker.memory_viz_snapshot(), f)
