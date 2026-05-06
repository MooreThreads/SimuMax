"""Trace/export helpers shared by perf schedule exporters."""

from __future__ import annotations

import json
import os

from simumax.core.generate_tracing import convert_to_tracing_format


def normalize_schedule_records(schedules):
    normalized = []
    for rank, tasks in enumerate(schedules):
        rank_tasks = []
        for task in tasks:
            if isinstance(task, dict):
                row = dict(task)
            else:
                task_type, mb, start, duration, end = task
                row = {
                    "kind": task_type,
                    "mb": mb,
                    "start": start,
                    "duration": duration,
                    "end": end,
                    "label": task_type,
                }
            row.setdefault("rank", rank)
            rank_tasks.append(row)
        normalized.append(rank_tasks)
    return normalized


def serialize_comm_lane_for_trace(schedules):
    serialized = []
    for tasks in schedules:
        rank_tasks = [dict(task) for task in tasks]
        comm_cursor = 0.0
        comm_indices = []
        for idx, task in enumerate(rank_tasks):
            kind = task.get("kind", task.get("label", ""))
            is_comm = not (kind.startswith("F") or kind.startswith("B"))
            if is_comm:
                comm_indices.append(idx)
        comm_indices.sort(
            key=lambda idx: (
                float(rank_tasks[idx]["start"]),
                float(rank_tasks[idx]["end"]),
                rank_tasks[idx].get("label", rank_tasks[idx].get("kind", "")),
            )
        )
        for idx in comm_indices:
            task = rank_tasks[idx]
            orig_start = float(task["start"])
            orig_end = float(task["end"])
            orig_dur = max(0.0, orig_end - orig_start)
            start = max(orig_start, comm_cursor)
            end = max(start, orig_end)
            task["start"] = start
            task["end"] = end
            task["duration"] = end - start
            if task["duration"] <= 0.0 and orig_dur > 0.0:
                task["collapsed_for_trace"] = True
                task["orig_duration"] = orig_dur
            comm_cursor = end
        serialized.append(rank_tasks)
    return serialized


def schedule_records_to_tracing_logs(schedules):
    parsed_logs = []
    for rank, tasks in enumerate(schedules):
        pid = f"rank{rank}"
        for task in tasks:
            kind = task.get("kind", task.get("label", ""))
            if task.get("collapsed_for_trace"):
                continue
            label = task.get("label", kind)
            if kind.startswith("F"):
                operation = "fwd"
                call_stack = [label]
            elif kind.startswith("B"):
                operation = "bwd"
                call_stack = [label]
            else:
                operation = "fwd"
                call_stack = [label]
            parsed_logs.append(
                {
                    "rank": pid,
                    "call_stack": call_stack,
                    "gid": task.get("gid"),
                    "operation": operation,
                    "cost": float(task.get("duration") or 0.0),
                    "st": float(task["start"]),
                    "ed": float(task["end"]),
                    "post": None,
                    "order": None,
                }
            )
    return parsed_logs


def export_pipeline_schedule_trace(
    schedules,
    output_json_path,
    title="perf_pp_schedule",
    *,
    serialize_comm_lanes=True,
):
    output_json_path = os.path.abspath(output_json_path)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    schedules = normalize_schedule_records(schedules)
    if serialize_comm_lanes:
        schedules = serialize_comm_lane_for_trace(schedules)
    parsed_logs = schedule_records_to_tracing_logs(schedules)
    tracing_events = convert_to_tracing_format(parsed_logs)
    for event in tracing_events:
        if event.get("ph") != "X":
            continue
        args = event.setdefault("args", {})
        args["scheduler"] = title
    payload = {"traceEvents": tracing_events, "displayTimeUnit": "ms"}
    with open(output_json_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=4)
