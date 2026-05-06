#!/usr/bin/env python3
"""Benchmark CE and TE MoE permute kernels for SimuMax bandwidth factors.

The measured time is CUDA-event elapsed time around the logical op span.  For
CE, communication is intentionally not included in the efficiency factor:
SimuMax models the TP all-reduces separately in ``ParallelCE``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "simu_tools/megatron_scripts/Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))


CUDA_ROOT_CANDIDATES = (
    Path("/usr/local/cuda"),
    Path("/usr/local/cuda-13.1"),
    Path("/usr/local/cuda-13"),
)


def prepend_env_paths(current_value: str, new_paths: list[Path]) -> str:
    items = [str(path) for path in new_paths if path]
    if current_value:
        items.extend(part for part in current_value.split(":") if part)
    deduped = []
    seen = set()
    for item in items:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return ":".join(deduped)


def apply_cuda_toolchain_env() -> str:
    for root in CUDA_ROOT_CANDIDATES:
        if (root / "bin/ptxas").exists() and (root / "include/cuda.h").exists():
            include_dirs = [root / "include", root / "targets/x86_64-linux/include"]
            os.environ.setdefault("CUDA_HOME", str(root))
            os.environ.setdefault("TRITON_PTXAS_PATH", str(root / "bin/ptxas"))
            os.environ["PATH"] = prepend_env_paths(os.environ.get("PATH", ""), [root / "bin"])
            for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
                os.environ[key] = prepend_env_paths(os.environ.get(key, ""), include_dirs)
            return str(root)
    return ""


def gpu_compute_apps() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    apps = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            apps.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]) if parts[1].isdigit() else parts[1],
                    "process_name": parts[2],
                    "used_memory_mib": parts[3],
                }
            )
    return apps


def require_no_unknown_gpu_apps(label: str):
    current_pid = os.getpid()
    apps = gpu_compute_apps()
    unknown = [app for app in apps if app.get("pid") != current_pid]
    if unknown:
        raise RuntimeError(f"unknown GPU apps at {label}: {unknown}")
    print(f"GPU apps at {label}: {apps}", flush=True)


def load_bandwidth(system_path: Path, op_name: str) -> tuple[float, float]:
    data = json.loads(system_path.read_text(encoding="utf-8"))
    bw = data["accelerator"]["bandwidth"].get(op_name) or data["accelerator"]["bandwidth"]["default"]
    return float(bw["gbps"]), float(bw["latency_us"])


def efficiency_from_bytes_time(mem_bytes: int, ms: float, gbps: float, latency_us: float, launches: int) -> float:
    effective_seconds = (ms / 1000.0) - (latency_us * launches / 1e6)
    if effective_seconds <= 0:
        return float("inf")
    return mem_bytes / (gbps * (1024**3) * effective_seconds)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def pmin(values: list[float]) -> float:
    return float(min(values))


def ce_theoretical_bytes(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    elem_bytes: int = 2,
    fusion: bool = False,
) -> tuple[int, int]:
    # Keep this exactly aligned with ParallelCE._comp_leaf_mem_accessed_info.
    logits_size = batch_size * seq_len * vocab_size
    if fusion:
        loss_size = batch_size * seq_len
        fwd = 2 * logits_size * elem_bytes + loss_size * 4
        bwd = 2 * logits_size * elem_bytes + loss_size * 4
        return int(fwd), int(bwd)
    fwd = logits_size * 4 + logits_size * elem_bytes
    fwd += (logits_size + batch_size * seq_len) * 4
    fwd += (logits_size + batch_size * seq_len + logits_size) * 4
    fwd += (logits_size * 2) * 4
    fwd += (logits_size + batch_size) * 4
    fwd += (logits_size + batch_size + logits_size) * 4
    bwd = (logits_size + batch_size + logits_size) * 4
    bwd += logits_size * 4 + logits_size * elem_bytes
    return int(fwd), int(bwd)


def time_cuda(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    # Return output as attribute for callers that need it without timing a tuple
    # allocation outside the event span.
    time_cuda.last_out = out
    return float(start.elapsed_time(end))


def init_dist_once():
    if dist.is_initialized():
        return
    init_file = tempfile.NamedTemporaryFile(delete=True)
    dist.init_process_group(
        "nccl",
        rank=0,
        world_size=1,
        init_method="file://" + init_file.name,
    )
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    init_dist_once._init_file = init_file


def benchmark_ce_case(case: dict, impl: str, warmup: int, repeats: int) -> dict:
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    init_dist_once()
    torch.cuda.empty_cache()
    b, s, v = case["batch_size"], case["seq_len"], case["vocab_size"]
    base = torch.randn(s, b, v, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, v, (s, b), device="cuda", dtype=torch.long)

    def ce(logits):
        if impl == "nonfusion":
            return vocab_parallel_cross_entropy(logits, target)
        if impl == "te_fusion":
            return parallel_cross_entropy(logits, target, 0.0, False, dist.group.WORLD)
        raise ValueError(impl)

    fwd_times, bwd_times = [], []
    for idx in range(warmup + repeats):
        logits = base.detach().clone().requires_grad_(True)
        fwd_ms = time_cuda(lambda: ce(logits))
        loss = time_cuda.last_out
        grad = torch.ones_like(loss)
        bwd_ms = time_cuda(lambda: loss.backward(grad))
        if idx >= warmup:
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
        del logits, loss, grad
        torch.cuda.empty_cache()

    fwd_bytes, bwd_bytes = ce_theoretical_bytes(b, s, v, fusion=(impl == "te_fusion"))
    return {
        **case,
        "impl": impl,
        "fwd_ms_median": median(fwd_times),
        "bwd_ms_median": median(bwd_times),
        "total_ms_median": median([a + b for a, b in zip(fwd_times, bwd_times)]),
        "fwd_ms_min": pmin(fwd_times),
        "bwd_ms_min": pmin(bwd_times),
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "total_bytes": fwd_bytes + bwd_bytes,
        "repeats": repeats,
        "warmup": warmup,
    }


def make_routing_map(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool)
    token_ids = torch.arange(num_tokens, device="cuda")[:, None]
    topk_ids = torch.arange(topk, device="cuda")[None, :]
    experts = (token_ids * topk + topk_ids) % num_experts
    routing_map.scatter_(1, experts, True)
    return routing_map


def permute_theoretical_bytes(case: dict, elem_bytes: int = 2) -> tuple[int, int]:
    t, h, e, k = case["num_tokens"], case["hidden_size"], case["num_experts"], case["topk"]
    capacity = math.ceil((t * k) / e) * case.get("capacity_factor", 1)
    permuted_tokens = capacity * e
    input_act = t * h
    permuted_act = permuted_tokens * h
    permutation_bytes = (input_act + permuted_act + permuted_act + permuted_act) * elem_bytes
    unpermutation_bytes = (2 * permuted_act + permuted_act + input_act) * elem_bytes
    return int(permutation_bytes + unpermutation_bytes), int(permutation_bytes + unpermutation_bytes)


def benchmark_permute_case(case: dict, warmup: int, repeats: int) -> dict:
    from transformer_engine.pytorch.permutation import moe_permute_with_probs, moe_unpermute

    torch.cuda.empty_cache()
    t, h, e, k = case["num_tokens"], case["hidden_size"], case["num_experts"], case["topk"]
    capacity = math.ceil((t * k) / e) * case.get("capacity_factor", 1)
    num_out_tokens = capacity * e
    base = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    routing_map = make_routing_map(t, e, k)
    probs = torch.rand(t, e, device="cuda", dtype=torch.float32)

    fwd_times, bwd_times = [], []
    for idx in range(warmup + repeats):
        tokens = base.detach().clone().requires_grad_(True)

        def fwd():
            permuted, _permuted_probs, sorted_indices = moe_permute_with_probs(
                tokens,
                probs,
                routing_map,
                num_out_tokens=num_out_tokens,
            )
            restored = moe_unpermute(permuted, sorted_indices, restore_shape=tokens.shape)
            return restored

        fwd_ms = time_cuda(fwd)
        restored = time_cuda.last_out
        bwd_ms = time_cuda(lambda: restored.sum().backward())
        if idx >= warmup:
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
        del tokens, restored
        torch.cuda.empty_cache()

    fwd_bytes, bwd_bytes = permute_theoretical_bytes(case)
    return {
        **case,
        "capacity": capacity,
        "num_out_tokens": num_out_tokens,
        "fwd_ms_median": median(fwd_times),
        "bwd_ms_median": median(bwd_times),
        "fwd_ms_min": pmin(fwd_times),
        "bwd_ms_min": pmin(bwd_times),
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "repeats": repeats,
        "warmup": warmup,
    }


def weighted_efficiency(rows: list[dict], time_key: str, bytes_key: str, gbps: float, latency_us: float, launches: int) -> float:
    total_bytes = sum(int(row[bytes_key]) for row in rows)
    total_ms = sum(float(row[time_key]) for row in rows)
    return efficiency_from_bytes_time(total_bytes, total_ms, gbps, latency_us, launches * len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="configs/system/b200_bf16_ceperm.json")
    parser.add_argument("--output", default="simu_tools/efficency_test/one_click_outputs/b200_ce_permute_summary.json")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    cuda_root = apply_cuda_toolchain_env()
    require_no_unknown_gpu_apps("start")
    system_path = (REPO_ROOT / args.system).resolve()
    out_path = (REPO_ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ce_gbps, ce_latency = load_bandwidth(system_path, "ce")
    perm_fwd_gbps, perm_fwd_latency = load_bandwidth(system_path, "permute_fwd")
    perm_bwd_gbps, perm_bwd_latency = load_bandwidth(system_path, "permute_bwd")

    ce_cases = [
        {"name": "ds236_tp1", "batch_size": 1, "seq_len": 4096, "vocab_size": 102400},
        {"name": "ds671_tp1", "batch_size": 1, "seq_len": 4096, "vocab_size": 129280},
        {"name": "llama_tp1", "batch_size": 1, "seq_len": 4096, "vocab_size": 128256},
        {"name": "llama_tp2", "batch_size": 1, "seq_len": 4096, "vocab_size": 64128},
        {"name": "llama_tp4", "batch_size": 1, "seq_len": 4096, "vocab_size": 32128},
        {"name": "llama_tp8", "batch_size": 1, "seq_len": 4096, "vocab_size": 16128},
        {"name": "llama8b_tp1_padded", "batch_size": 1, "seq_len": 4096, "vocab_size": 128512},
    ]
    permute_cases = [
        {"name": "ds236", "num_tokens": 4096, "hidden_size": 5120, "num_experts": 160, "topk": 6, "capacity_factor": 1},
        {"name": "ds671", "num_tokens": 4096, "hidden_size": 7168, "num_experts": 256, "topk": 8, "capacity_factor": 1},
    ]

    ce_rows = []
    for impl in ("nonfusion", "te_fusion"):
        for case in ce_cases:
            print(f"CE {impl} {case['name']}", flush=True)
            ce_rows.append(benchmark_ce_case(case, impl, args.warmup, args.repeats))
            require_no_unknown_gpu_apps(f"after CE {impl} {case['name']}")

    permute_rows = []
    for case in permute_cases:
        print(f"permute {case['name']}", flush=True)
        permute_rows.append(benchmark_permute_case(case, args.warmup, args.repeats))
        require_no_unknown_gpu_apps(f"after permute {case['name']}")

    ce_nonfusion = [row for row in ce_rows if row["impl"] == "nonfusion"]
    ce_fusion = [row for row in ce_rows if row["impl"] == "te_fusion"]
    recommended = {
        "ce": weighted_efficiency(
            ce_nonfusion,
            "total_ms_median",
            "total_bytes",
            ce_gbps,
            ce_latency,
            launches=2,
        ),
        "ce_fusion": weighted_efficiency(
            ce_fusion,
            "total_ms_median",
            "total_bytes",
            ce_gbps,
            ce_latency,
            launches=2,
        ),
        "permute_fwd": weighted_efficiency(
            permute_rows,
            "fwd_ms_median",
            "fwd_bytes",
            perm_fwd_gbps,
            perm_fwd_latency,
            launches=4,
        ),
        "permute_bwd": weighted_efficiency(
            permute_rows,
            "bwd_ms_median",
            "bwd_bytes",
            perm_bwd_gbps,
            perm_bwd_latency,
            launches=4,
        ),
    }

    summary = {
        "system": str(system_path),
        "cuda_root": cuda_root,
        "bandwidth_reference": {
            "ce": {"gbps": ce_gbps, "latency_us": ce_latency},
            "permute_fwd": {"gbps": perm_fwd_gbps, "latency_us": perm_fwd_latency},
            "permute_bwd": {"gbps": perm_bwd_gbps, "latency_us": perm_bwd_latency},
        },
        "recommended_efficient_factor": recommended,
        "ce_rows": ce_rows,
        "permute_rows": permute_rows,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "recommended_efficient_factor": recommended}, indent=2), flush=True)
    require_no_unknown_gpu_apps("end")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
