import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simumax.core.perf_llm import PerfLLM


TOOLS_OUTPUT_DIR = REPO_ROOT / "tools" / "b200" / "outputs"
SCREEN_JSON = TOOLS_OUTPUT_DIR / "megatron_perf_real_pipeline_screen.json"
STATE_JSON = TOOLS_OUTPUT_DIR / "megatron_perf_real_pipeline_state.json"
SUMMARY_JSON = TOOLS_OUTPUT_DIR / "megatron_perf_real_pipeline_summary.json"
CURRENT_SUMMARY_RECORD = REPO_ROOT / "docs" / "b200" / "b200_release_v1.2_summary.json"

SYSTEMS = [
    {
        "config": "b200_ceperm",
        "system_rel": "configs/system/b200_bf16_ceperm.json",
    },
]
B200_TE_VERSION = "2.11.0"

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
MEM_RE = re.compile(
    r"\[Rank\s+(\d+)\]\s+\(after\s+(\d+)\s+iterations\)\s+memory \(MB\) \| "
    r"allocated:\s*([0-9.]+)\s*\|\s*max allocated:\s*([0-9.]+)\s*\| "
    r"reserved:\s*([0-9.]+)\s*\|\s*max reserved:\s*([0-9.]+)"
)

REAL_MATCH_ABS_TOL_MS = 10.0
REAL_MATCH_REL_TOL = 0.005
# Memory parsing contract for formal comparison:
# - real alloc/reserved must be taken as the max over all ranks
# - partial-rank memory logs are invalid
# - dedicated memory refresh reruns use 2 train steps / 0 warmup
REAL_MEMORY_ENV_DEFAULTS = {
    "MEGATRON_LOG_MEMORY_ALL_RANKS": "1",
    "MEGATRON_REPORT_MEMORY_EVERY_ITER": "1",
    "MEGATRON_REPORT_MEMORY_MIN_ITER": "1",
    "MEGATRON_TRAIN_ITERS": "3",
    "MEGATRON_WARMUP_ITERS": "1",
    "CUDA_HOME": "/usr/local/cuda-13.1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda-13.1/bin/ptxas",
    "PATH_PREFIX": "/usr/local/cuda-13.1/bin",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def abs_path(rel_path: str) -> Path:
    return REPO_ROOT / rel_path


def apply_real_env_defaults(env: dict) -> dict:
    merged = dict(env)
    path_prefix = REAL_MEMORY_ENV_DEFAULTS.get("PATH_PREFIX", "")
    for key, value in REAL_MEMORY_ENV_DEFAULTS.items():
        if key == "PATH_PREFIX":
            continue
        merged.setdefault(key, value)
    if path_prefix:
        current_path = merged.get("PATH") or os.environ.get("PATH", "")
        parts = current_path.split(":") if current_path else []
        if path_prefix not in parts:
            merged["PATH"] = f"{path_prefix}:{current_path}" if current_path else path_prefix
    return merged


def dense_case(
    name,
    model_rel,
    model_name,
    model_type_arg,
    strategy_rel,
    parallel,
    layer_num,
    tp_size,
    pp_size,
    micro_batch_num,
    *,
    vp_size=1,
    cp_size=1,
    pp_comm_async=False,
    launcher_base_rel="simu_tools/megatron_scripts",
    extra_env=None,
    record_case_name=None,
    suites=None,
):
    return {
        "name": name,
        "record_case_name": record_case_name or name,
        "kind": "dense",
        "model_rel": model_rel,
        "model_name": model_name,
        "model_type_arg": model_type_arg,
        "strategy_rel": strategy_rel,
        "parallel": parallel,
        "layer_num": layer_num,
        "world_size": 8,
        "micro_batch_size": 1,
        "micro_batch_num": micro_batch_num,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "ep_size": 1,
        "cp_size": cp_size,
        "vp_size": vp_size,
        "pp_comm_async": pp_comm_async,
        "strategy_overrides": {},
        "model_overrides": {
            "layer_num": layer_num,
            "padded_vocab_size": True,
            "make_vocab_size_divisible_by": 128,
        },
        "launcher": {
            "base_rel": launcher_base_rel,
            "script_rel": "run_llama3.sh",
            "model_output_rel": f"output_{model_type_arg}",
        },
        "extra_env": extra_env or {},
        "suites": suites or [],
    }


def moe_case(
    name,
    *,
    strategy_rel,
    parallel,
    layer_num,
    micro_batch_num,
    pp_size,
    ep_size,
    record_case_name=None,
    suites=None,
):
    return {
        "name": name,
        "record_case_name": record_case_name or name,
        "kind": "moe",
        "model_rel": "configs/models/deepseekv2.json",
        "model_name": "deepseekv2",
        "model_type_arg": "deepseek_v2",
        "strategy_rel": strategy_rel,
        "parallel": parallel,
        "layer_num": layer_num,
        "world_size": 8,
        "micro_batch_size": 1,
        "micro_batch_num": micro_batch_num,
        "tp_size": 1,
        "pp_size": pp_size,
        "ep_size": ep_size,
        "cp_size": 1,
        "vp_size": 1,
        "pp_comm_async": False,
        "strategy_overrides": {
            "dispatch_probs": True,
        },
        "model_overrides": {
            "layer_num": layer_num,
            "moe_pad_expert_input_to_capacity": True,
            "capacity": 1,
            "padded_vocab_size": True,
            "make_vocab_size_divisible_by": 128,
        },
        "launcher": {
            "base_rel": "simu_tools/megatron_scripts",
            "script_rel": "run_deepseekv2.sh",
            "model_output_rel": "output_deepseek_v2",
        },
        "extra_env": {},
        "suites": suites or [],
    }


CASES = [
    dense_case(
        "llama3_8b_tp1_pp1_dp8_l8_mbc8",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp1_dp8_mbs1.json",
        parallel="tp1pp1",
        layer_num=8,
        tp_size=1,
        pp_size=1,
        micro_batch_num=8,
        record_case_name="llama3_8b_l8_tp1_pp1_dp8_mbc8",
        suites=["baseline"],
    ),
    dense_case(
        "llama3_8b_tp1_pp2_dp4_l8_mbc8",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp2_dp4_mbs1.json",
        parallel="tp1pp2",
        layer_num=8,
        tp_size=1,
        pp_size=2,
        micro_batch_num=8,
        record_case_name="llama3_8b_l8_tp1_pp2_dp4_mbc8",
        suites=["baseline"],
    ),
    dense_case(
        "llama3_8b_tp1_pp4_vp2_dp2_l8_mbc8_sync",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
        parallel="tp1pp4vp2",
        layer_num=8,
        tp_size=1,
        pp_size=4,
        micro_batch_num=8,
        vp_size=2,
        pp_comm_async=False,
        record_case_name="llama3_8b_l8_tp1_pp4_vp2_dp2_mbc8_sync",
        suites=["vpp", "vpp-existing"],
        extra_env={
            "VP_SIZE": "2",
            "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
        },
    ),
    dense_case(
        "llama3_8b_tp1_pp4_vp2_dp2_l8_mbc16_sync",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
        parallel="tp1pp4vp2",
        layer_num=8,
        tp_size=1,
        pp_size=4,
        micro_batch_num=16,
        vp_size=2,
        pp_comm_async=False,
        record_case_name="llama3_8b_l8_tp1_pp4_vp2_dp2_mbc16_sync",
        suites=["vpp", "vpp-existing"],
        extra_env={
            "VP_SIZE": "2",
            "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
        },
    ),
    dense_case(
        "llama3_8b_tp1_pp4_vp2_dp2_l8_mbc32_sync",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
        parallel="tp1pp4vp2",
        layer_num=8,
        tp_size=1,
        pp_size=4,
        micro_batch_num=32,
        vp_size=2,
        pp_comm_async=False,
        record_case_name="llama3_8b_l8_tp1_pp4_vp2_dp2_mbc32_sync",
        suites=["vpp", "vpp-existing"],
        extra_env={
            "VP_SIZE": "2",
            "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
        },
    ),
    dense_case(
        "llama3_8b_tp1_pp2_dp4_l32_mbc4",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp2_dp4_mbs1.json",
        parallel="tp1pp2",
        layer_num=32,
        tp_size=1,
        pp_size=2,
        micro_batch_num=4,
        record_case_name="llama3_8b_l32_tp1_pp2_dp4_mbc4",
        suites=["full-results"],
    ),
    dense_case(
        "llama3_8b_tp1_pp2_dp4_l32_mbc8",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp2_dp4_mbs1.json",
        parallel="tp1pp2",
        layer_num=32,
        tp_size=1,
        pp_size=2,
        micro_batch_num=8,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_8b_tp1_pp2_dp4_l32_mbc32",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp2_dp4_mbs1.json",
        parallel="tp1pp2",
        layer_num=32,
        tp_size=1,
        pp_size=2,
        micro_batch_num=32,
        record_case_name="llama3_8b_l32_tp1_pp2_dp4_mbc32",
        suites=["full-results"],
    ),
    dense_case(
        "llama3_8b_tp2_pp1_dp4_l32_mbc4",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp2_pp1_dp4_mbs1.json",
        parallel="tp2pp1",
        layer_num=32,
        tp_size=2,
        pp_size=1,
        micro_batch_num=4,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_8b_tp2_pp1_dp4_l32_mbc8",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp2_pp1_dp4_mbs1.json",
        parallel="tp2pp1",
        layer_num=32,
        tp_size=2,
        pp_size=1,
        micro_batch_num=8,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_8b_tp2_pp1_dp4_l32_mbc32",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp2_pp1_dp4_mbs1.json",
        parallel="tp2pp1",
        layer_num=32,
        tp_size=2,
        pp_size=1,
        micro_batch_num=32,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_70b_tp1_pp2_dp4_l12_mbc4",
        model_rel="configs/models/llama3-70b.json",
        model_name="llama3-70b",
        model_type_arg="llama3_70b",
        strategy_rel="configs/strategy/tp1_pp2_dp4_mbs1.json",
        parallel="tp1pp2",
        layer_num=12,
        tp_size=1,
        pp_size=2,
        micro_batch_num=4,
        record_case_name="llama3_70b_l12_tp1_pp2_dp4_mbc4",
        suites=["full-results"],
    ),
    dense_case(
        "llama3_70b_tp1_pp2_dp4_l12_mbc8",
        model_rel="configs/models/llama3-70b.json",
        model_name="llama3-70b",
        model_type_arg="llama3_70b",
        strategy_rel="configs/strategy/tp1_pp2_dp4_mbs1.json",
        parallel="tp1pp2",
        layer_num=12,
        tp_size=1,
        pp_size=2,
        micro_batch_num=8,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_70b_tp1_pp2_dp4_l12_mbc32",
        model_rel="configs/models/llama3-70b.json",
        model_name="llama3-70b",
        model_type_arg="llama3_70b",
        strategy_rel="configs/strategy/tp1_pp2_dp4_mbs1.json",
        parallel="tp1pp2",
        layer_num=12,
        tp_size=1,
        pp_size=2,
        micro_batch_num=32,
        record_case_name="llama3_70b_l12_tp1_pp2_dp4_mbc32",
        suites=["full-results"],
    ),
    dense_case(
        "llama3_70b_tp2_pp1_dp4_l12_mbc4",
        model_rel="configs/models/llama3-70b.json",
        model_name="llama3-70b",
        model_type_arg="llama3_70b",
        strategy_rel="configs/strategy/tp2_pp1_dp4_mbs1.json",
        parallel="tp2pp1",
        layer_num=12,
        tp_size=2,
        pp_size=1,
        micro_batch_num=4,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_70b_tp2_pp1_dp4_l12_mbc8",
        model_rel="configs/models/llama3-70b.json",
        model_name="llama3-70b",
        model_type_arg="llama3_70b",
        strategy_rel="configs/strategy/tp2_pp1_dp4_mbs1.json",
        parallel="tp2pp1",
        layer_num=12,
        tp_size=2,
        pp_size=1,
        micro_batch_num=8,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_70b_tp2_pp1_dp4_l12_mbc32",
        model_rel="configs/models/llama3-70b.json",
        model_name="llama3-70b",
        model_type_arg="llama3_70b",
        strategy_rel="configs/strategy/tp2_pp1_dp4_mbs1.json",
        parallel="tp2pp1",
        layer_num=12,
        tp_size=2,
        pp_size=1,
        micro_batch_num=32,
        suites=["full-results"],
    ),
    moe_case(
        "deepseekv2_tp1_ep8_pp1_dp8_l4_mbc8",
        strategy_rel="configs/strategy/ep8_pp1_dp8_mbs1.json",
        parallel="tp1ep8pp1",
        layer_num=4,
        micro_batch_num=8,
        pp_size=1,
        ep_size=8,
        suites=["full-results"],
    ),
    moe_case(
        "deepseekv2_tp1_ep8_pp1_dp8_l4_mbc4",
        strategy_rel="configs/strategy/ep8_pp1_dp8_mbs1.json",
        parallel="tp1ep8pp1",
        layer_num=4,
        micro_batch_num=4,
        pp_size=1,
        ep_size=8,
        suites=["full-results"],
    ),
    moe_case(
        "deepseekv2_tp1_ep8_pp1_dp8_l4_mbc32",
        strategy_rel="configs/strategy/ep8_pp1_dp8_mbs1.json",
        parallel="tp1ep8pp1",
        layer_num=4,
        micro_batch_num=32,
        pp_size=1,
        ep_size=8,
        suites=["full-results"],
    ),
    moe_case(
        "deepseekv2_tp1_ep4_pp2_dp4_l4_mbc4",
        strategy_rel="configs/strategy/ep4_pp2_dp4_mbs1.json",
        parallel="tp1ep4pp2",
        layer_num=4,
        micro_batch_num=4,
        pp_size=2,
        ep_size=4,
        suites=["full-results"],
    ),
    moe_case(
        "deepseekv2_tp1_ep4_pp2_dp4_l4_mbc8",
        strategy_rel="configs/strategy/ep4_pp2_dp4_mbs1.json",
        parallel="tp1ep4pp2",
        layer_num=4,
        micro_batch_num=8,
        pp_size=2,
        ep_size=4,
        suites=["full-results"],
    ),
    moe_case(
        "deepseekv2_tp1_ep4_pp2_dp4_l4_mbc32",
        strategy_rel="configs/strategy/ep4_pp2_dp4_mbs1.json",
        parallel="tp1ep4pp2",
        layer_num=4,
        micro_batch_num=32,
        pp_size=2,
        ep_size=4,
        suites=["full-results"],
    ),
    dense_case(
        "llama3_8b_tp1_pp4_vp2_dp2_l8_mbc4_sync",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
        parallel="tp1pp4vp2",
        layer_num=8,
        tp_size=1,
        pp_size=4,
        micro_batch_num=4,
        vp_size=2,
        pp_comm_async=False,
        record_case_name="llama3_8b_l8_tp1_pp4_vp2_dp2_mbc4_sync",
        suites=["vpp", "vpp-expand"],
        extra_env={
            "VP_SIZE": "2",
            "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
        },
    ),
    dense_case(
        "llama3_8b_tp1_pp4_vp2_dp2_l8_mbc2_sync",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
        parallel="tp1pp4vp2",
        layer_num=8,
        tp_size=1,
        pp_size=4,
        micro_batch_num=2,
        vp_size=2,
        pp_comm_async=False,
        record_case_name="llama3_8b_l8_tp1_pp4_vp2_dp2_mbc2_sync",
        suites=["vpp", "vpp-more"],
        extra_env={
            "VP_SIZE": "2",
            "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
        },
    ),
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
        suites=["vpp", "vpp-expand"],
        extra_env={
            "VP_SIZE": "2",
            "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
        },
    ),
    dense_case(
        "llama3_8b_tp1_pp4_vp2_dp2_l8_mbc128_sync",
        model_rel="configs/models/llama3-8b.json",
        model_name="llama3-8b",
        model_type_arg="llama3_8b",
        strategy_rel="configs/strategy/tp1_pp4_vp2_sync_mbs1_mbc8_no_ckpt.json",
        parallel="tp1pp4vp2",
        layer_num=8,
        tp_size=1,
        pp_size=4,
        micro_batch_num=128,
        vp_size=2,
        pp_comm_async=False,
        record_case_name="llama3_8b_l8_tp1_pp4_vp2_dp2_mbc128_sync",
        suites=["vpp", "vpp-more"],
        extra_env={
            "VP_SIZE": "2",
            "EXTRA_ARGS": "--transformer-impl transformer_engine --no-overlap-p2p-communication",
        },
    ),
]

CASES_BY_NAME = {case["name"]: case for case in CASES}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perf-first Megatron perf-vs-real pipeline for the current repo."
    )
    parser.add_argument(
        "--phase",
        choices=["perf-screen", "megatron-run", "summarize", "all"],
        default="all",
    )
    parser.add_argument(
        "--perf-source",
        choices=["auto", "recorded", "recompute"],
        default="auto",
        help=(
            "Prefer checked-in perf records when available. "
            "'recorded' now falls back to recompute for cases not covered by the "
            "checked-in formal summary."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Run only the named suite(s), e.g. full-results or vpp-expand.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named case(s). Defaults to all curated cases.",
    )
    parser.add_argument(
        "--screen-json",
        default=str(SCREEN_JSON.relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--state-json",
        default=str(STATE_JSON.relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--summary-json",
        default=str(SUMMARY_JSON.relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--system-rel",
        default=None,
        help="Override the system config for perf recompute without editing the curated case table.",
    )
    parser.add_argument(
        "--reuse-state-runs",
        action="store_true",
        help="Skip Megatron launch when the state file already records a successful run for the case.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Megatron commands without launching them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun perf screen and Megatron even if cached state exists.",
    )
    return parser.parse_args()


def load_json(path: Path, default):
    if not path.exists():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def selected_cases(case_names):
    if not case_names:
        return CASES
    missing = sorted(set(case_names) - set(CASES_BY_NAME))
    if missing:
        raise SystemExit(f"Unknown case(s): {missing}")
    return [CASES_BY_NAME[name] for name in case_names]


def filter_cases_by_suite(cases, suites):
    if not suites:
        return cases
    suite_set = set(suites)
    return [case for case in cases if suite_set & set(case.get("suites", []))]


def parse_duration_ms(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith(" ms"):
        return float(text.split()[0])
    if text.endswith(" s"):
        return float(text.split()[0]) * 1000.0
    return float(text)


def miss_summary(miss_efficiency):
    miss_count = 0
    miss_ops = []
    for op_name, payload in miss_efficiency.items():
        miss_ops.append(op_name)
        if isinstance(payload, dict):
            miss_count += len(payload)
        elif isinstance(payload, list):
            miss_count += len(payload)
        else:
            miss_count += 1
    return miss_count, sorted(set(miss_ops))


def load_recorded_perf_map():
    recorded = {}
    if not CURRENT_SUMMARY_RECORD.exists():
        return recorded
    payload = json.loads(CURRENT_SUMMARY_RECORD.read_text(encoding="utf-8"))
    items = payload["rows"] if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else payload
    for item in items:
        if not isinstance(item, dict):
            continue
        case_name = item.get("case")
        if case_name:
            for alias in case_name_aliases(case_name):
                recorded.setdefault(alias, copy.deepcopy(item))
    return recorded


def case_name_aliases(case_name: str):
    aliases = []
    if not case_name:
        return aliases
    aliases.append(case_name)
    if case_name.endswith("_cef"):
        aliases.append(case_name[: -len("_cef")])
    else:
        aliases.append(f"{case_name}_cef")
    return aliases


def system_rel_from_filename(filename):
    for system in SYSTEMS:
        if Path(system["system_rel"]).name == filename:
            return system["system_rel"]
    return filename


def configure_perf(case, system_rel):
    perf = PerfLLM()
    perf.configure(
        strategy_config=str(abs_path(case["strategy_rel"])),
        model_config=str(abs_path(case["model_rel"])),
        system_config=str(abs_path(system_rel)),
        debug_points=[],
    )
    perf.strategy.world_size = case["world_size"]
    perf.strategy.tp_size = case["tp_size"]
    perf.strategy.pp_size = case["pp_size"]
    perf.strategy.ep_size = case["ep_size"]
    perf.strategy.cp_size = case["cp_size"]
    perf.strategy.micro_batch_size = case["micro_batch_size"]
    perf.strategy.micro_batch_num = case["micro_batch_num"]
    perf.strategy.interleaving_size = case["vp_size"]
    perf.strategy.te_version = case.get("te_version", B200_TE_VERSION)
    if case["pp_comm_async"] is not None:
        perf.strategy.pp_comm_async = case["pp_comm_async"]
    for key, value in case["strategy_overrides"].items():
        setattr(perf.strategy, key, value)
    for key, value in case["model_overrides"].items():
        setattr(perf.model_config, key, value)
    return perf


def perf_screen_case(case):
    system_results = []
    fit_all = True
    max_peak_mem_gb = 0.0
    for system in SYSTEMS:
        perf = configure_perf(case, system["system_rel"])
        perf.strategy.enable_straggler_model = False
        perf.run_estimate()
        mem_result = perf.analysis_mem()
        peak_by_stage = perf.get_pp_stage_peak_mem(mem_result, "peak_mem_with_reserved", toG=True)
        peak_mem_gb = max(float(value) for value in peak_by_stage.values())
        max_peak_mem_gb = max(max_peak_mem_gb, peak_mem_gb)
        accelerator_mem_gb = float(perf.system.accelerator.mem_gbs)
        fit = peak_mem_gb <= accelerator_mem_gb + 1e-9
        fit_all = fit_all and fit

        runs = []
        for straggler in (False, True):
            perf = configure_perf(case, system["system_rel"])
            perf.strategy.enable_straggler_model = straggler
            perf.run_estimate()
            cost = perf.analysis_cost().data
            miss_count, miss_ops = miss_summary(perf.system.miss_efficiency)
            runs.append(
                {
                    "straggler": straggler,
                    "perf_ms": round(parse_duration_ms(cost["duration_time_per_iter"]), 1),
                    "straggle_ratio": float(cost.get("straggle_ratio", 1.0)),
                    "miss_count": miss_count,
                    "miss_ops": miss_ops,
                }
            )

        system_results.append(
            {
                "config": system["config"],
                "system_rel": system["system_rel"],
                "fit": fit,
                "accelerator_mem_gb": accelerator_mem_gb,
                "peak_mem_with_reserved_gb": round(peak_mem_gb, 4),
                "peak_mem_by_stage_gb": peak_by_stage,
                "runs": runs,
            }
        )

    return {
        "name": case["name"],
        "model_name": case["model_name"],
        "parallel": case["parallel"],
        "layer_num": case["layer_num"],
        "micro_batch_num": case["micro_batch_num"],
        "status": "fit" if fit_all else "oom",
        "max_peak_mem_with_reserved_gb": round(max_peak_mem_gb, 4),
        "systems": system_results,
    }


def recorded_screen_case(case, recorded_item):
    legacy_systems = recorded_item.get("systems")
    if isinstance(legacy_systems, list):
        system_results = []
        for system in legacy_systems:
            system_results.append(
                {
                    "config": system["config"],
                    "system_rel": system_rel_from_filename(system["system"]),
                    "fit": True,
                    "peak_mem_with_reserved_gb": None,
                    "peak_mem_by_stage_gb": {},
                    "runs": copy.deepcopy(system["runs"]),
                }
            )
        return {
            "name": case["name"],
            "record_case_name": case["record_case_name"],
            "model_name": case["model_name"],
            "parallel": case["parallel"],
            "layer_num": case["layer_num"],
            "micro_batch_num": case["micro_batch_num"],
            "status": "fit",
            "max_peak_mem_with_reserved_gb": None,
            "source": "recorded",
            "record_real_ms": recorded_item.get("real_ms"),
            "systems": system_results,
        }

    system_rel = recorded_item.get(
        "system_rel",
        "configs/system/b200_bf16_ceperm.json",
    )
    perf_ms = recorded_item.get("perf_ms")
    miss_count = int(recorded_item.get("miss_count", 0) or 0)
    miss_ops = sorted(set(recorded_item.get("miss_ops", []) or []))
    system_results = []
    system_results.append(
        {
            "config": Path(str(system_rel)).stem,
            "system_rel": system_rel,
            "fit": True,
            "peak_mem_with_reserved_gb": recorded_item.get("perf_peak_reserved_gib"),
            "peak_mem_by_stage_gb": recorded_item.get("peak_alloc_by_stage_gib", {}) or {},
            "runs": [
                {
                    "straggler": False,
                    "perf_ms": round(float(perf_ms), 1) if perf_ms is not None else None,
                    "straggle_ratio": 1.0,
                    "miss_count": miss_count,
                    "miss_ops": miss_ops,
                }
            ],
        }
    )
    return {
        "name": case["name"],
        "record_case_name": case["record_case_name"],
        "model_name": case["model_name"],
        "parallel": case["parallel"],
        "layer_num": case["layer_num"],
        "micro_batch_num": case["micro_batch_num"],
        "status": "fit",
        "max_peak_mem_with_reserved_gb": None,
        "source": "recorded",
        "record_real_ms": recorded_item.get("real_ms"),
        "systems": system_results,
    }


def megatron_command(case):
    launcher = case["launcher"]
    launcher_base = abs_path(launcher["base_rel"])
    script_path = launcher_base / launcher["script_rel"]
    if case["kind"] == "dense":
        cmd = [
            "bash",
            str(script_path),
            str(case["micro_batch_size"]),
            str(case["micro_batch_num"]),
            str(case["tp_size"]),
            str(case["pp_size"]),
            case["model_type_arg"],
            "test",
            str(case["layer_num"]),
            str(case["cp_size"]),
            "bf16",
        ]
    else:
        cmd = [
            "bash",
            str(script_path),
            str(case["micro_batch_size"]),
            str(case["micro_batch_num"]),
            str(case["ep_size"]),
            str(case["pp_size"]),
            case["model_type_arg"],
            "test",
            str(case["layer_num"]),
            str(case["cp_size"]),
            "4096",
            "bf16",
        ]
    env = {}
    env.update(case["extra_env"])
    return launcher_base, cmd, env


def snapshot_dirs(root: Path):
    if not root.exists():
        return {}
    return {
        child.name: child.stat().st_mtime
        for child in root.iterdir()
        if child.is_dir()
    }


def pick_new_run_dir(root: Path, before, start_ts):
    if not root.exists():
        return None
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        mtime = child.stat().st_mtime
        if child.name not in before or mtime >= start_ts - 1.0:
            candidates.append((mtime, child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_dir_has_stdout_logs(run_dir: Path | None) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    return any(path.is_file() for path in run_dir.rglob("stdout.log"))


def run_megatron_case(case, state_case, dry_run=False):
    launcher_base, cmd, extra_env = megatron_command(case)
    output_root = launcher_base / case["launcher"]["model_output_rel"]
    before = snapshot_dirs(output_root)
    env = apply_real_env_defaults(dict(**extra_env))
    run_record = {
        "status": "dry_run" if dry_run else "launched",
        "command": cmd,
        "cwd": str(launcher_base),
        "env": env,
        "started_at": utc_now(),
    }
    if dry_run:
        return run_record

    real_env = dict(**env)
    proc_env = dict(**real_env)
    proc_env.update({"PYTHONUNBUFFERED": "1"})
    start_ts = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(launcher_base),
        env={**dict(**subprocess.os.environ), **proc_env},
        check=False,
    )
    run_dir = pick_new_run_dir(output_root, before, start_ts)
    status = "ok" if completed.returncode == 0 and run_dir_has_stdout_logs(run_dir) else "failed"
    run_record.update(
        {
            "ended_at": utc_now(),
            "returncode": completed.returncode,
            "status": status,
            "run_dir": str(run_dir) if run_dir else "",
        }
    )
    if completed.returncode == 0 and status != "ok":
        run_record["failure_reason"] = "launcher_returned_zero_but_no_stdout_logs_found"
    state_case["run"] = run_record
    return run_record


def rank_from_log(path: Path):
    try:
        return int(path.parent.name)
    except ValueError:
        return -1


def parse_run_logs(run_dir: Path):
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
                        "total_iterations": int(iter_match.group(2)),
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
    steady = [item for item in timings if item["iteration"] >= 2]
    steady = steady or timings
    real_ms = min(item["elapsed_ms"] for item in steady)
    return {
        "run_dir": str(run_dir),
        "timing_rank": timing_rank,
        "timing_samples": timings,
        "steady_iterations": [item["iteration"] for item in steady],
        "real_ms": real_ms,
        "peak_memory_by_rank_mb": mem_by_rank,
        "max_peak_allocated_mb": max(
            (item["max_allocated_mb"] for item in mem_by_rank.values()),
            default=0.0,
        ),
        "max_peak_reserved_mb": max(
            (item["max_reserved_mb"] for item in mem_by_rank.values()),
            default=0.0,
        ),
    }


def case_summary(case, perf_screen, real_run):
    systems = []
    for system in perf_screen["systems"]:
        runs = []
        for run in system["runs"]:
            runs.append(
                {
                    "straggler": run["straggler"],
                    "perf_ms": run["perf_ms"],
                    "real_minus_perf_ms": round(real_run["real_ms"] - run["perf_ms"], 6),
                    "real_over_perf": round(real_run["real_ms"] / run["perf_ms"], 9),
                    "miss_count": run["miss_count"],
                    "miss_ops": run["miss_ops"],
                }
            )
        systems.append(
            {
                "config": system["config"],
                "system": Path(str(system["system_rel"])).name,
                "fit": system["fit"],
                "peak_mem_with_reserved_gb": system["peak_mem_with_reserved_gb"],
                "runs": runs,
            }
        )
    record_real_ms = perf_screen.get("record_real_ms")
    record_real_match = None
    if record_real_ms is not None:
        record_real_ms = float(record_real_ms)
        record_real_delta = real_run["real_ms"] - record_real_ms
        record_real_match = abs(record_real_delta) <= max(
            REAL_MATCH_ABS_TOL_MS, abs(record_real_ms) * REAL_MATCH_REL_TOL
        )
    else:
        record_real_delta = None
    return {
        "case": case["record_case_name"],
        "pipeline_case_name": case["name"],
        "model": case["model_name"],
        "parallel": case["parallel"],
        "layer_num": case["layer_num"],
        "mbc": case["micro_batch_num"],
        "real_ms": round(real_run["real_ms"], 6),
        "record_real_ms": record_real_ms,
        "record_real_delta_ms": round(record_real_delta, 6) if record_real_delta is not None else None,
        "record_real_match": record_real_match,
        "timing_rank": real_run["timing_rank"],
        "steady_iterations": real_run["steady_iterations"],
        "peak_memory_by_rank_mb": real_run["peak_memory_by_rank_mb"],
        "max_peak_allocated_mb": round(real_run["max_peak_allocated_mb"], 6),
        "systems": systems,
    }


def find_case_screen(screen_payload, case_name):
    for item in screen_payload["cases"]:
        if item["name"] == case_name:
            return item
    return None


def perf_screen_phase(cases, screen_path, state_path, perf_source, force=False):
    screen_payload = {
        "generated_at": utc_now(),
        "perf_source": perf_source,
        "cases": [],
    }
    if force:
        state_payload = {"generated_at": utc_now(), "cases": {}}
    else:
        state_payload = load_json(state_path, {"generated_at": utc_now(), "cases": {}})
    recorded_map = load_recorded_perf_map() if perf_source in {"auto", "recorded"} else {}
    for case in cases:
        print(f"[perf-screen] {case['name']}", flush=True)
        recorded_item = recorded_map.get(case["record_case_name"]) if perf_source in {"auto", "recorded"} else None
        if perf_source == "recorded" and recorded_item is not None:
            result = recorded_screen_case(case, recorded_item)
        elif perf_source == "auto" and recorded_item is not None:
            result = recorded_screen_case(case, recorded_item)
        else:
            result = perf_screen_case(case)
        screen_payload["cases"].append(result)
        state_payload["cases"].setdefault(case["name"], {})
        state_payload["cases"][case["name"]]["perf_screen"] = result
    dump_json(screen_path, screen_payload)
    dump_json(state_path, state_payload)
    runnable = [item["name"] for item in screen_payload["cases"] if item["status"] == "fit"]
    print(json.dumps({"runnable_cases": runnable, "count": len(runnable)}, indent=2, ensure_ascii=False))
    return screen_payload, state_payload


def megatron_run_phase(cases, screen_payload, state_path, dry_run=False, reuse_state_runs=False, force=False):
    state_payload = load_json(state_path, {"generated_at": utc_now(), "cases": {}})
    for case in cases:
        screen_case = find_case_screen(screen_payload, case["name"])
        if not screen_case:
            raise RuntimeError(f"Missing perf screen result for {case['name']}")
        if screen_case["status"] != "fit":
            print(f"[megatron-run] skip oom case {case['name']}", flush=True)
            continue
        state_case = state_payload["cases"].setdefault(case["name"], {})
        if reuse_state_runs and not force and state_case.get("run", {}).get("status") == "ok":
            print(f"[megatron-run] reuse {case['name']}", flush=True)
            continue
        print(f"[megatron-run] {case['name']}", flush=True)
        state_case["run"] = run_megatron_case(case, state_case, dry_run=dry_run)
        dump_json(state_path, state_payload)
    return state_payload


def summarize_phase(cases, state_path, summary_path, perf_source):
    state_payload = load_json(state_path, {"generated_at": utc_now(), "cases": {}})
    recorded_map = load_recorded_perf_map() if perf_source in {"auto", "recorded"} else {}
    summary_cases = []
    missing = []
    for case in cases:
        state_case = state_payload["cases"].setdefault(case["name"], {})
        perf_screen = state_case.get("perf_screen")
        if perf_screen is None:
            recorded_item = recorded_map.get(case["record_case_name"]) if perf_source in {"auto", "recorded"} else None
            if perf_source == "recorded" and recorded_item is not None:
                perf_screen = recorded_screen_case(case, recorded_item)
            elif perf_source == "auto" and recorded_item is not None:
                perf_screen = recorded_screen_case(case, recorded_item)
            else:
                perf_screen = perf_screen_case(case)
            state_case["perf_screen"] = perf_screen
        run_dir_text = state_case.get("run", {}).get("run_dir", "")
        run_dir = Path(run_dir_text) if run_dir_text else None
        if run_dir and not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        if not (run_dir and run_dir.exists()):
            run_dir = None
        if run_dir is None:
            missing.append(case["name"])
            continue
        print(f"[summarize] {case['name']} <- {run_dir}", flush=True)
        real_run = parse_run_logs(run_dir)
        state_case["summary_run_source"] = str(run_dir)
        state_case["real_run"] = real_run
        summary_cases.append(case_summary(case, perf_screen, real_run))

    summary_payload = {
        "generated_at": utc_now(),
        "case_count": len(summary_cases),
        "missing_cases": missing,
        "record_match_failures": [
            item["case"]
            for item in summary_cases
            if item.get("record_real_match") is False
        ],
        "cases": summary_cases,
    }
    dump_json(state_path, state_payload)
    dump_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return summary_payload


def main():
    args = parse_args()
    if args.system_rel:
        override_name = Path(args.system_rel).stem
        SYSTEMS[:] = [
            {
                "config": override_name,
                "system_rel": args.system_rel,
            },
        ]
    cases = filter_cases_by_suite(selected_cases(args.case), args.suite)
    if not cases:
        raise SystemExit("No cases matched the requested filters.")
    screen_path = abs_path(args.screen_json)
    state_path = abs_path(args.state_json)
    summary_path = abs_path(args.summary_json)

    if args.phase == "perf-screen":
        perf_screen_phase(cases, screen_path, state_path, perf_source=args.perf_source, force=args.force)
        return 0

    if args.phase == "megatron-run":
        screen_payload = load_json(screen_path, {"generated_at": utc_now(), "cases": []})
        if not screen_payload["cases"] or args.force:
            screen_payload, _ = perf_screen_phase(
                cases,
                screen_path,
                state_path,
                perf_source=args.perf_source,
                force=args.force,
            )
        megatron_run_phase(
            cases,
            screen_payload,
            state_path,
            dry_run=args.dry_run,
            reuse_state_runs=args.reuse_state_runs,
            force=args.force,
        )
        return 0

    if args.phase == "summarize":
        summarize_phase(
            cases,
            state_path,
            summary_path,
            perf_source=args.perf_source,
        )
        return 0

    screen_payload, _ = perf_screen_phase(
        cases,
        screen_path,
        state_path,
        perf_source=args.perf_source,
        force=args.force,
    )
    megatron_run_phase(
        cases,
        screen_payload,
        state_path,
        dry_run=args.dry_run,
        reuse_state_runs=args.reuse_state_runs,
        force=args.force,
    )
    if not args.dry_run:
        summarize_phase(
            cases,
            state_path,
            summary_path,
            perf_source=args.perf_source,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
