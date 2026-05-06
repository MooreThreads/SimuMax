#!/usr/bin/env python3
import argparse
import copy
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.b200.run_megatron_perf_real_pipeline import (  # noqa: E402
    CASES_BY_NAME,
    ITER_RE,
    MEM_RE,
    REAL_MEMORY_ENV_DEFAULTS,
    case_name_aliases,
    configure_perf,
    dense_case,
    megatron_command,
    moe_case,
    pick_new_run_dir,
    run_dir_has_stdout_logs,
    rank_from_log,
    snapshot_dirs,
)


SYSTEM_REL = "configs/system/b200_bf16_ceperm.json"
SUMMARY_JSON = REPO_ROOT / "tools" / "b200" / "outputs" / "megatron_perf_real_batch_summary.json"
OUTPUT_MD = REPO_ROOT / "tools" / "b200" / "outputs" / "megatron_perf_real_batch_results.md"


FULL_RESULT_CASE_NAMES = [
    "llama3_8b_tp1_pp2_dp4_l32_mbc4",
    "llama3_8b_tp1_pp2_dp4_l32_mbc8",
    "llama3_8b_tp1_pp2_dp4_l32_mbc32",
    "llama3_8b_tp2_pp1_dp4_l32_mbc4",
    "llama3_8b_tp2_pp1_dp4_l32_mbc8",
    "llama3_8b_tp2_pp1_dp4_l32_mbc32",
    "llama3_70b_tp1_pp2_dp4_l12_mbc4",
    "llama3_70b_tp1_pp2_dp4_l12_mbc8",
    "llama3_70b_tp1_pp2_dp4_l12_mbc32",
    "llama3_70b_tp2_pp1_dp4_l12_mbc4",
    "llama3_70b_tp2_pp1_dp4_l12_mbc8",
    "llama3_70b_tp2_pp1_dp4_l12_mbc32",
    "deepseekv2_tp1_ep8_pp1_dp8_l4_mbc4",
    "deepseekv2_tp1_ep8_pp1_dp8_l4_mbc8",
    "deepseekv2_tp1_ep8_pp1_dp8_l4_mbc32",
    "deepseekv2_tp1_ep4_pp2_dp4_l4_mbc4",
    "deepseekv2_tp1_ep4_pp2_dp4_l4_mbc8",
    "deepseekv2_tp1_ep4_pp2_dp4_l4_mbc32",
]


def clone_case(case_name, section):
    case = copy.deepcopy(CASES_BY_NAME[case_name])
    case["section"] = section
    case["display_case"] = case["record_case_name"]
    return case


def build_active_cases():
    full_results = [clone_case(name, "Full Results") for name in FULL_RESULT_CASE_NAMES]

    validated_vpp = [
        dense_case(
            "llama3_8b_tp1_pp4_vp2_dp2_l32_mbc8_sync",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp1pp4vp2",
            layer_num=32,
            tp_size=1,
            pp_size=4,
            micro_batch_num=8,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_8b_l32_tp1_pp4_vp2_dp2_mbc8_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_8b_tp1_pp4_vp2_dp2_l32_mbc16_sync",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp1pp4vp2",
            layer_num=32,
            tp_size=1,
            pp_size=4,
            micro_batch_num=16,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_8b_l32_tp1_pp4_vp2_dp2_mbc16_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        clone_case("llama3_8b_tp1_pp4_vp2_dp2_l8_mbc8_sync", "Validated VPP"),
        clone_case("llama3_8b_tp1_pp4_vp2_dp2_l8_mbc16_sync", "Validated VPP"),
        clone_case("llama3_8b_tp1_pp4_vp2_dp2_l8_mbc32_sync", "Validated VPP"),
        dense_case(
            "llama3_8b_tp1_pp4_vp2_dp2_l8_mbc64_sync",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp1pp4vp2",
            layer_num=8,
            tp_size=1,
            pp_size=4,
            micro_batch_num=64,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_8b_l8_tp1_pp4_vp2_dp2_mbc64_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_70b_tp1_pp4_vp2_dp2_l8_mbc8_sync",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp1pp4vp2",
            layer_num=8,
            tp_size=1,
            pp_size=4,
            micro_batch_num=8,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_70b_l8_tp1_pp4_vp2_dp2_mbc8_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_70b_tp1_pp4_vp2_dp2_l8_mbc16_sync",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp1pp4vp2",
            layer_num=8,
            tp_size=1,
            pp_size=4,
            micro_batch_num=16,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_70b_l8_tp1_pp4_vp2_dp2_mbc16_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_70b_tp1_pp4_vp2_dp2_l8_mbc32_sync",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp1pp4vp2",
            layer_num=8,
            tp_size=1,
            pp_size=4,
            micro_batch_num=32,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_70b_l8_tp1_pp4_vp2_dp2_mbc32_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_70b_tp1_pp4_vp2_dp2_l8_mbc64_sync",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp1pp4vp2",
            layer_num=8,
            tp_size=1,
            pp_size=4,
            micro_batch_num=64,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_70b_l8_tp1_pp4_vp2_dp2_mbc64_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_8b_tp2_pp4_vp2_dp1_l32_mbc4_sync",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp2pp4vp2",
            layer_num=32,
            tp_size=2,
            pp_size=4,
            micro_batch_num=4,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_8b_l32_tp2_pp4_vp2_dp1_mbc4_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_8b_tp2_pp4_vp2_dp1_l32_mbc8_sync",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp2pp4vp2",
            layer_num=32,
            tp_size=2,
            pp_size=4,
            micro_batch_num=8,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_8b_l32_tp2_pp4_vp2_dp1_mbc8_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_8b_tp2_pp4_vp2_dp1_l32_mbc16_sync",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp2pp4vp2",
            layer_num=32,
            tp_size=2,
            pp_size=4,
            micro_batch_num=16,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_8b_l32_tp2_pp4_vp2_dp1_mbc16_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
        dense_case(
            "llama3_8b_tp2_pp4_vp2_dp1_l32_mbc32_sync",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
            parallel="tp2pp4vp2",
            layer_num=32,
            tp_size=2,
            pp_size=4,
            micro_batch_num=32,
            vp_size=2,
            pp_comm_async=False,
            record_case_name="llama3_8b_l32_tp2_pp4_vp2_dp1_mbc32_sync",
            extra_env={
                "VP_SIZE": "2",
                "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
            },
        ),
    ]
    for case in validated_vpp:
        case["section"] = "Validated VPP"
        case["display_case"] = case["record_case_name"]

    extra_tp = [
        dense_case(
            "llama3_8b_tp4_pp1_dp2_l32_mbc4",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp4_pp1_dp2_mbs1.json",
            parallel="tp4pp1",
            layer_num=32,
            tp_size=4,
            pp_size=1,
            micro_batch_num=4,
            record_case_name="llama3_8b_l32_tp4_pp1_dp2_mbc4",
        ),
        dense_case(
            "llama3_8b_tp4_pp1_dp2_l32_mbc8",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp4_pp1_dp2_mbs1.json",
            parallel="tp4pp1",
            layer_num=32,
            tp_size=4,
            pp_size=1,
            micro_batch_num=8,
            record_case_name="llama3_8b_l32_tp4_pp1_dp2_mbc8",
        ),
        dense_case(
            "llama3_8b_tp4_pp1_dp2_l32_mbc32",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp4_pp1_dp2_mbs1.json",
            parallel="tp4pp1",
            layer_num=32,
            tp_size=4,
            pp_size=1,
            micro_batch_num=32,
            record_case_name="llama3_8b_l32_tp4_pp1_dp2_mbc32",
        ),
        dense_case(
            "llama3_8b_tp8_pp1_dp1_l32_mbc4",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp8_pp1_dp1_mbs1.json",
            parallel="tp8pp1",
            layer_num=32,
            tp_size=8,
            pp_size=1,
            micro_batch_num=4,
            record_case_name="llama3_8b_l32_tp8_pp1_dp1_mbc4",
        ),
        dense_case(
            "llama3_8b_tp8_pp1_dp1_l32_mbc8",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp8_pp1_dp1_mbs1.json",
            parallel="tp8pp1",
            layer_num=32,
            tp_size=8,
            pp_size=1,
            micro_batch_num=8,
            record_case_name="llama3_8b_l32_tp8_pp1_dp1_mbc8",
        ),
        dense_case(
            "llama3_8b_tp8_pp1_dp1_l32_mbc32",
            model_rel="configs/models/llama3-8b.json",
            model_name="llama3-8b",
            model_type_arg="llama3_8b",
            strategy_rel="configs/strategy/tp8_pp1_dp1_mbs1.json",
            parallel="tp8pp1",
            layer_num=32,
            tp_size=8,
            pp_size=1,
            micro_batch_num=32,
            record_case_name="llama3_8b_l32_tp8_pp1_dp1_mbc32",
        ),
        dense_case(
            "llama3_70b_tp4_pp1_dp2_l12_mbc4",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp4_pp1_dp2_mbs1.json",
            parallel="tp4pp1",
            layer_num=12,
            tp_size=4,
            pp_size=1,
            micro_batch_num=4,
            record_case_name="llama3_70b_l12_tp4_pp1_dp2_mbc4",
        ),
        dense_case(
            "llama3_70b_tp4_pp1_dp2_l12_mbc8",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp4_pp1_dp2_mbs1.json",
            parallel="tp4pp1",
            layer_num=12,
            tp_size=4,
            pp_size=1,
            micro_batch_num=8,
            record_case_name="llama3_70b_l12_tp4_pp1_dp2_mbc8",
        ),
        dense_case(
            "llama3_70b_tp4_pp1_dp2_l12_mbc32",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp4_pp1_dp2_mbs1.json",
            parallel="tp4pp1",
            layer_num=12,
            tp_size=4,
            pp_size=1,
            micro_batch_num=32,
            record_case_name="llama3_70b_l12_tp4_pp1_dp2_mbc32",
        ),
        dense_case(
            "llama3_70b_tp8_pp1_dp1_l12_mbc4",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp8_pp1_dp1_mbs1.json",
            parallel="tp8pp1",
            layer_num=12,
            tp_size=8,
            pp_size=1,
            micro_batch_num=4,
            record_case_name="llama3_70b_l12_tp8_pp1_dp1_mbc4",
        ),
        dense_case(
            "llama3_70b_tp8_pp1_dp1_l12_mbc8",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp8_pp1_dp1_mbs1.json",
            parallel="tp8pp1",
            layer_num=12,
            tp_size=8,
            pp_size=1,
            micro_batch_num=8,
            record_case_name="llama3_70b_l12_tp8_pp1_dp1_mbc8",
        ),
        dense_case(
            "llama3_70b_tp8_pp1_dp1_l12_mbc32",
            model_rel="configs/models/llama3-70b.json",
            model_name="llama3-70b",
            model_type_arg="llama3_70b",
            strategy_rel="configs/strategy/tp8_pp1_dp1_mbs1.json",
            parallel="tp8pp1",
            layer_num=12,
            tp_size=8,
            pp_size=1,
            micro_batch_num=32,
            record_case_name="llama3_70b_l12_tp8_pp1_dp1_mbc32",
        ),
    ]
    for case in extra_tp:
        case["section"] = "Additional TP Checks"
        case["display_case"] = case["record_case_name"]

    return full_results + validated_vpp + extra_tp


def parse_duration_ms(value):
    if isinstance(value, str):
        parts = value.strip().split()
        token = float(parts[0])
        unit = parts[1].lower() if len(parts) > 1 else "ms"
        if unit in {"s", "sec", "secs", "second", "seconds"}:
            return token * 1000.0
        if unit in {"ms", "msec", "msecs", "millisecond", "milliseconds"}:
            return token
        if unit in {"us", "usec", "usecs", "microsecond", "microseconds"}:
            return token / 1000.0
        return token
    return float(value)


def miss_summary(miss_efficiency):
    if not miss_efficiency:
        return 0, []
    if isinstance(miss_efficiency, dict):
        ops = sorted(str(key) for key in miss_efficiency.keys())
        miss_count = sum(len(value) if isinstance(value, dict) else 1 for value in miss_efficiency.values())
        return miss_count, ops
    ops = []
    for item in miss_efficiency:
        if isinstance(item, dict):
            ops.append(item.get("op_name", "unknown"))
        else:
            ops.append(str(item))
    return len(miss_efficiency), sorted(set(ops))


def run_perf(case, system_rel):
    perf = configure_perf(case, system_rel)
    perf.strategy.enable_straggler_model = False
    perf.run_estimate()
    cost = perf.analysis_cost().data
    mem_result = perf.analysis_mem()
    peak_alloc_by_stage = perf.get_pp_stage_peak_mem(mem_result, "peak_mem", toG=True)
    peak_alloc_gib = max(float(value) for value in peak_alloc_by_stage.values())
    miss_count, miss_ops = miss_summary(perf.system.miss_efficiency)
    return {
        "perf_ms": round(parse_duration_ms(cost["duration_time_per_iter"]), 2),
        "perf_alloc_gib": round(peak_alloc_gib, 2),
        "miss_count": miss_count,
        "miss_ops": miss_ops,
        "peak_alloc_by_stage_gib": peak_alloc_by_stage,
    }


def run_real(case):
    launcher_base, cmd, extra_env = megatron_command(case)
    output_root = launcher_base / case["launcher"]["model_output_rel"]
    before = snapshot_dirs(output_root)
    proc_env = dict(subprocess.os.environ)
    proc_env.update(extra_env)
    for key, value in REAL_MEMORY_ENV_DEFAULTS.items():
        proc_env.setdefault(key, value)
    proc_env["PYTHONUNBUFFERED"] = "1"
    start_ts = time.time()
    completed = subprocess.run(cmd, cwd=str(launcher_base), env=proc_env, check=False)
    run_dir = pick_new_run_dir(output_root, before, start_ts)
    status = "ok" if completed.returncode == 0 and run_dir_has_stdout_logs(run_dir) else "failed"
    result = {
        "status": status,
        "returncode": completed.returncode,
        "run_dir": str(run_dir) if run_dir else "",
    }
    if completed.returncode == 0 and status != "ok":
        result["failure_reason"] = "launcher_returned_zero_but_no_stdout_logs_found"
    if run_dir:
        try:
            result.update(parse_fastest_run_logs(run_dir))
        except (FileNotFoundError, RuntimeError):
            pass
    return result


def parse_fastest_run_logs(run_dir: Path):
    stdout_logs = sorted(run_dir.rglob("stdout.log"), key=rank_from_log)
    if not stdout_logs:
        raise FileNotFoundError(f"No stdout.log found under {run_dir}")

    timing_by_rank = {}
    mem_by_rank = {}
    for log_path in stdout_logs:
        rank = rank_from_log(log_path)
        timings = []
        peak_allocated = 0.0
        peak_reserved = 0.0
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            iter_match = ITER_RE.search(line)
            if iter_match:
                timings.append(
                    {
                        "iteration": int(iter_match.group(1)),
                        "elapsed_ms": float(iter_match.group(3)),
                    }
                )
            mem_match = MEM_RE.search(line)
            if mem_match:
                peak_allocated = max(peak_allocated, float(mem_match.group(4)))
                peak_reserved = max(peak_reserved, float(mem_match.group(6)))
        if timings:
            timing_by_rank[rank] = timings
        if peak_allocated > 0.0:
            mem_by_rank[rank] = {
                "max_allocated_mb": peak_allocated,
                "max_reserved_mb": peak_reserved,
            }

    if not timing_by_rank:
        raise RuntimeError(f"No iteration timing lines found under {run_dir}")

    timing_rank = max(timing_by_rank)
    timings = timing_by_rank[timing_rank]
    steady = [item for item in timings if item["iteration"] >= 2] or timings
    fastest = min(item["elapsed_ms"] for item in steady)
    return {
        "timing_rank": timing_rank,
        "steady_iterations": [item["iteration"] for item in steady],
        "real_ms": round(fastest, 2),
        "peak_memory_by_rank_mb": mem_by_rank,
        "real_alloc_gib": round(
            max((item["max_allocated_mb"] for item in mem_by_rank.values()), default=0.0) / 1024,
            2,
        ),
        "real_reserved_gib": round(
            max((item["max_reserved_mb"] for item in mem_by_rank.values()), default=0.0) / 1024,
            2,
        ),
    }


def build_row(case, perf_result, real_result):
    row = {
        "section": case["section"],
        "case": case["display_case"],
        "status": real_result["status"],
        "run_dir": real_result.get("run_dir", ""),
        "perf_ms": perf_result["perf_ms"],
        "perf_alloc_gib": perf_result["perf_alloc_gib"],
        "miss_count": perf_result["miss_count"],
        "miss_ops": perf_result["miss_ops"],
    }
    if real_result["status"] == "ok":
        real_ms = real_result.get("real_ms")
        real_alloc = real_result.get("real_alloc_gib")
        if real_ms is None or real_alloc is None:
            row.update(
                {
                    "status": "failed",
                    "real_ms": None,
                    "rel_err_pct": None,
                    "real_alloc_gib": real_result.get("real_alloc_gib"),
                    "real_reserved_gib": real_result.get("real_reserved_gib"),
                    "alloc_err_pct": None,
                }
            )
            return row
        row.update(
            {
                "real_ms": real_ms,
                "rel_err_pct": round((perf_result["perf_ms"] - real_ms) / real_ms * 100, 2),
                "real_alloc_gib": real_alloc,
                "real_reserved_gib": real_result["real_reserved_gib"],
                "alloc_err_pct": round((perf_result["perf_alloc_gib"] - real_alloc) / real_alloc * 100, 2),
            }
        )
    else:
        row.update(
            {
                "real_ms": None,
                "rel_err_pct": None,
                "real_alloc_gib": real_result.get("real_alloc_gib"),
                "real_reserved_gib": real_result.get("real_reserved_gib"),
                "alloc_err_pct": None,
            }
        )
    return row


def format_signed_pct(value):
    return "-" if value is None else f"{value:+.2f}%"


def format_float(value):
    return "-" if value is None else f"{value:.2f}"


def write_markdown(rows, output_md: Path, system_rel: str):
    output_md.parent.mkdir(parents=True, exist_ok=True)
    sections = ["Full Results", "Validated VPP", "Additional TP Checks"]
    lines = [
        f"# Megatron Perf vs Real Batch Results ({datetime.now().strftime('%Y-%m-%d')})",
        "",
        "口径统一如下:",
        "",
        "- `real ms`: timing rank 上 `iteration>=2` 的最快一次 iteration",
        "- `perf ms`: 当前代码 fresh recompute 的 `analysis_cost().duration_time_per_iter`",
        f"- `perf system`: `{system_rel}`",
        "- `real` 显存统一记 Megatron 日志里 `all ranks` 的全局最大 `max allocated / max reserved`",
        "- `perf alloc`: `analysis_mem()` 的 `peak_mem`（不含 reserved）",
        "- `rel err`: `(perf ms - real ms) / real ms`",
        "- `alloc err`: `(perf alloc - real alloc) / real alloc`",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"## {section}",
                "",
                "| case | status | real ms | perf ms | rel err | real max alloc GiB | real max reserved GiB | perf alloc GiB | alloc err |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["section"] != section:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["case"],
                        row["status"],
                        format_float(row["real_ms"]),
                        format_float(row["perf_ms"]),
                        format_signed_pct(row["rel_err_pct"]),
                        format_float(row["real_alloc_gib"]),
                        format_float(row["real_reserved_gib"]),
                        format_float(row["perf_alloc_gib"]),
                        format_signed_pct(row["alloc_err_pct"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run active Megatron perf-vs-real matrix with the current perf config.")
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument(
        "--system-rel",
        default=SYSTEM_REL,
        help="Override the system config for perf recompute.",
    )
    parser.add_argument(
        "--reuse-real-summary-json",
        default="",
        help="Reuse real results from an existing summary json instead of rerunning Megatron.",
    )
    return parser.parse_args()


def build_real_reuse_map(summary_json: Path):
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    result = {}
    for row in payload["rows"]:
        case_name = row.get("case")
        if not case_name:
            continue
        for alias in case_name_aliases(case_name):
            result.setdefault(alias, row)
    return result


def reuse_real(row):
    return {
        "status": row["status"],
        "run_dir": row.get("run_dir", ""),
        "real_ms": row.get("real_ms"),
        "real_alloc_gib": row.get("real_alloc_gib"),
        "real_reserved_gib": row.get("real_reserved_gib"),
    }


def main():
    args = parse_args()
    cases = build_active_cases()
    real_reuse = {}
    if args.reuse_real_summary_json:
        reuse_path = Path(args.reuse_real_summary_json)
        if not reuse_path.is_absolute():
            reuse_path = REPO_ROOT / reuse_path
        real_reuse = build_real_reuse_map(reuse_path)
    rows = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['display_case']}", flush=True)
        perf_result = run_perf(case, args.system_rel)
        reused = real_reuse.get(case["display_case"])
        real_result = reuse_real(reused) if reused else run_real(case)
        rows.append(build_row(case, perf_result, real_result))

    summary = {
        "generated_at": datetime.now().isoformat(),
        "system_rel": args.system_rel,
        "real_ms_rule": "fastest steady iteration on timing rank (iteration>=2)",
        "rows": rows,
    }

    summary_json = Path(args.summary_json)
    if not summary_json.is_absolute():
        summary_json = REPO_ROOT / summary_json
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = REPO_ROOT / output_md
    write_markdown(rows, output_md, args.system_rel)
    print(json.dumps({"summary_json": str(summary_json), "output_md": str(output_md)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
