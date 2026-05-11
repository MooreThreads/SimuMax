from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    try:
        importlib.import_module("torch_musa")
    except ImportError:
        pass

import torch
import torch.distributed as dist


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def get_device_module():
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa", torch.musa
    if torch.cuda.is_available():
        return "cuda", torch.cuda
    raise RuntimeError("No MUSA/CUDA device available")


def create_event(device_mod):
    return device_mod.Event(enable_timing=True)


def sync_device(device_mod):
    device_mod.synchronize()


def current_stream(device_mod):
    if hasattr(device_mod, "current_stream"):
        return device_mod.current_stream()
    return None


def set_device(device_kind, local_rank):
    if device_kind == "musa":
        torch.musa.set_device(local_rank)
        return torch.device("musa", local_rank)
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def get_backend(device_kind):
    if device_kind == "musa":
        return "mccl"
    if device_kind == "cuda":
        return "nccl"
    return "gloo"


def build_pattern(mode, block_count):
    if mode == "comm_only":
        return ["comm"] * block_count
    if mode == "gemm_only":
        return ["gemm"] * block_count
    if mode == "comm_gemm_alternating":
        return [item for _ in range(block_count) for item in ("comm", "gemm")]
    raise ValueError(f"Unknown mode: {mode}")


def _aligned_elems(total_bytes, bytes_per_elem, align=1):
    elems = max(total_bytes // bytes_per_elem, align)
    elems -= elems % align
    return elems


def make_buffers(device, dtype, world_size, gemm_m, gemm_n, gemm_k, comm_mb, comm_op):
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    logical_bytes = comm_mb * 1024 * 1024

    lhs = torch.randn((gemm_m, gemm_k), device=device, dtype=dtype)
    rhs = torch.randn((gemm_k, gemm_n), device=device, dtype=dtype)
    out = torch.empty((gemm_m, gemm_n), device=device, dtype=dtype)

    if comm_op == "all_reduce":
        comm_elems = _aligned_elems(logical_bytes, bytes_per_elem, align=world_size)
        comm_in = torch.randn((comm_elems,), device=device, dtype=dtype)
        comm_out = torch.empty_like(comm_in)
    elif comm_op == "all_gather":
        total_elems = _aligned_elems(logical_bytes, bytes_per_elem, align=world_size)
        input_elems = total_elems // world_size
        comm_in = torch.randn((input_elems,), device=device, dtype=dtype)
        comm_out = torch.empty((total_elems,), device=device, dtype=dtype)
    elif comm_op == "reduce_scatter":
        total_elems = _aligned_elems(logical_bytes, bytes_per_elem, align=world_size)
        output_elems = total_elems // world_size
        comm_in = torch.randn((total_elems,), device=device, dtype=dtype)
        comm_out = torch.empty((output_elems,), device=device, dtype=dtype)
    elif comm_op == "all_to_all":
        comm_elems = _aligned_elems(logical_bytes, bytes_per_elem, align=world_size)
        comm_in = torch.randn((comm_elems,), device=device, dtype=dtype)
        comm_out = torch.empty_like(comm_in)
    else:
        raise ValueError(f"Unsupported comm op: {comm_op}")
    return lhs, rhs, out, comm_in, comm_out


def run_gemm(lhs, rhs, out):
    torch.mm(lhs, rhs, out=out)


def run_comm(op_name, comm_in, comm_out):
    if op_name == "all_reduce":
        dist.all_reduce(comm_in)
        return
    if op_name == "all_gather":
        if hasattr(dist, "all_gather_into_tensor"):
            dist.all_gather_into_tensor(comm_out, comm_in)
        else:
            chunks = list(torch.chunk(comm_out, dist.get_world_size()))
            dist.all_gather(chunks, comm_in)
        return
    if op_name == "reduce_scatter":
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(comm_out, comm_in, op=dist.ReduceOp.SUM)
        else:
            inputs = list(torch.chunk(comm_in, dist.get_world_size()))
            dist.reduce_scatter(comm_out, inputs, op=dist.ReduceOp.SUM)
        return
    if op_name == "all_to_all":
        dist.all_to_all_single(comm_out, comm_in)
        return
    raise ValueError(f"Unsupported comm op: {op_name}")


def summarize_values(values):
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def run_iteration(
    device_mod,
    pattern,
    comm_op,
    lhs,
    rhs,
    out,
    comm_in,
    comm_out,
    window_start,
    window_len,
):
    starts = []
    ends = []
    names = []
    stream = current_stream(device_mod)

    for item in pattern:
        start = create_event(device_mod)
        end = create_event(device_mod)
        start.record(stream)
        if item == "gemm":
            run_gemm(lhs, rhs, out)
        else:
            run_comm(comm_op, comm_in, comm_out)
        end.record(stream)
        starts.append(start)
        ends.append(end)
        names.append(item)

    sync_device(device_mod)

    ops = []
    for idx, name in enumerate(names):
        ops.append(
            {
                "index": idx,
                "name": name,
                "duration_ms": starts[idx].elapsed_time(ends[idx]),
            }
        )

    gaps = []
    for idx in range(len(names) - 1):
        gaps.append(
            {
                "from_index": idx,
                "to_index": idx + 1,
                "from_name": names[idx],
                "to_name": names[idx + 1],
                "gap_to_next_ms": ends[idx].elapsed_time(starts[idx + 1]),
            }
        )

    window_end = window_start + window_len
    window_total_ms = starts[window_start].elapsed_time(ends[window_end - 1])
    window_ops = ops[window_start:window_end]
    window_gap_values = [gap["gap_to_next_ms"] for gap in gaps if window_start <= gap["from_index"] < window_end - 1]
    window_sum_op_ms = sum(item["duration_ms"] for item in window_ops)
    window_bubble_ms = window_total_ms - window_sum_op_ms

    return {
        "window_start": window_start,
        "window_len": window_len,
        "window_total_ms": window_total_ms,
        "window_sum_op_ms": window_sum_op_ms,
        "window_bubble_ms": window_bubble_ms,
        "window_mean_gap_ms": (sum(window_gap_values) / len(window_gap_values)) if window_gap_values else 0.0,
        "window_max_gap_ms": max(window_gap_values) if window_gap_values else 0.0,
        "ops": ops,
        "gaps": gaps,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure middle-window comm burst overhead using device events.")
    parser.add_argument("--comm-op", choices=["all_reduce", "all_gather", "reduce_scatter", "all_to_all"], required=True)
    parser.add_argument("--mode", choices=["comm_only", "gemm_only", "comm_gemm_alternating"], default="comm_only")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--block-count", type=int, default=20)
    parser.add_argument("--window-start", type=int, default=5)
    parser.add_argument("--window-len", type=int, default=10)
    parser.add_argument("--gemm-m", type=int, default=6144)
    parser.add_argument("--gemm-n", type=int, default=6144)
    parser.add_argument("--gemm-k", type=int, default=6144)
    parser.add_argument("--comm-mb", type=int, default=32)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="bf16")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.window_start < 0 or args.window_len <= 0 or args.window_start + args.window_len > len(build_pattern(args.mode, args.block_count)):
        raise ValueError("invalid window range")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    device_kind, device_mod = get_device_module()
    device = set_device(device_kind, local_rank)
    dist.init_process_group(get_backend(device_kind), rank=rank, world_size=world_size)

    dtype = DTYPE_MAP[args.dtype]
    lhs, rhs, out, comm_in, comm_out = make_buffers(
        device=device,
        dtype=dtype,
        world_size=world_size,
        gemm_m=args.gemm_m,
        gemm_n=args.gemm_n,
        gemm_k=args.gemm_k,
        comm_mb=args.comm_mb,
        comm_op=args.comm_op,
    )
    pattern = build_pattern(args.mode, args.block_count)

    for _ in range(args.warmup_iters):
        run_iteration(
            device_mod,
            pattern,
            args.comm_op,
            lhs,
            rhs,
            out,
            comm_in,
            comm_out,
            args.window_start,
            args.window_len,
        )
    dist.barrier()

    iteration_results = []
    for _ in range(args.measure_iters):
        iteration_results.append(
            run_iteration(
                device_mod,
                pattern,
                args.comm_op,
                lhs,
                rhs,
                out,
                comm_in,
                comm_out,
                args.window_start,
                args.window_len,
            )
        )
    dist.barrier()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    window_total_values = [item["window_total_ms"] for item in iteration_results]
    window_bubble_values = [item["window_bubble_ms"] for item in iteration_results]
    window_gap_values = [item["window_mean_gap_ms"] for item in iteration_results]
    rank_payload = {
        "rank": rank,
        "world_size": world_size,
        "device_kind": device_kind,
        "comm_op": args.comm_op,
        "mode": args.mode,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "block_count": args.block_count,
        "window_start": args.window_start,
        "window_len": args.window_len,
        "gemm_shape": [args.gemm_m, args.gemm_n, args.gemm_k],
        "comm_mb": args.comm_mb,
        "dtype": args.dtype,
        "iteration_results": iteration_results,
        "rank_summary": {
            "window_total_ms": summarize_values(window_total_values),
            "window_bubble_ms": summarize_values(window_bubble_values),
            "window_mean_gap_ms": summarize_values(window_gap_values),
        },
    }
    (output_dir / f"rank{rank}.json").write_text(json.dumps(rank_payload, indent=2), encoding="utf-8")
    dist.barrier()

    if rank == 0:
        rank_payloads = []
        for idx in range(world_size):
            rank_payloads.append(json.loads((output_dir / f"rank{idx}.json").read_text()))

        aggregate = {}
        for key in ["window_total_ms", "window_bubble_ms", "window_mean_gap_ms"]:
            rank_means = [item["rank_summary"][key]["mean"] for item in rank_payloads]
            aggregate[key] = {
                "rank_mean": sum(rank_means) / len(rank_means),
                "rank_min": min(rank_means),
                "rank_max": max(rank_means),
            }
        summary = {
            "world_size": world_size,
            "device_kind": device_kind,
            "comm_op": args.comm_op,
            "mode": args.mode,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "block_count": args.block_count,
            "window_start": args.window_start,
            "window_len": args.window_len,
            "comm_mb": args.comm_mb,
            "dtype": args.dtype,
            "aggregate": aggregate,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
