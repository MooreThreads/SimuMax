#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from simumax.core.config import SystemConfig
from simu_tools.efficency_test.one_click_common import (
    COLLECTIVE_SPECS,
    dump_json,
    fit_bw_latency,
    parse_nccl_rows,
    round_sig,
    run_compute_pipeline,
    run_nccl_collective_suite,
    update_system_network_from_fit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
WORKER = SCRIPT_DIR / "measure_comm_burst_window_worker.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click benchmark (compute efficiency + NCCL communication fit + fixed-latency calibration)."
    )
    parser.add_argument("--max-tflops", type=float, required=True)
    parser.add_argument("--sys-name", required=True)
    parser.add_argument("--param-file", default=str(SCRIPT_DIR / "run_params.json"))
    parser.add_argument("--pcie-intra-link", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fc8-mode", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--compute-cache-mode",
        default="supplement",
        choices=["supplement", "rebuild"],
        help="supplement: reuse the selected cache namespace and only fill missing entries; rebuild: use a fresh cache namespace and benchmark into a new config.",
    )
    parser.add_argument(
        "--compute-cache-tag",
        default="",
        help="Optional suffix for compute cache namespace. In rebuild mode, if omitted, a timestamped tag is generated automatically.",
    )
    parser.add_argument("--device-bw-gbps", type=float, default=None, help="Nominal intra-node link bandwidth for efficient_factor fit. If unset, infer from GPU model.")
    parser.add_argument("--skip-compute", action="store_true")
    parser.add_argument("--skip-comm", action="store_true")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-fixed-latency", action="store_true")
    parser.add_argument("--nccl-bin-dir", default="", help="Directory containing nccl-tests binaries; empty means use PATH.")
    parser.add_argument("--nccl-begin", default="1M")
    parser.add_argument("--nccl-end", default="8G")
    parser.add_argument("--nccl-fallback-end", default="256M")
    parser.add_argument("--nccl-world-size", type=int, default=8)
    parser.add_argument("--nccl-dtype", default="bfloat16")
    parser.add_argument("--nccl-iters", type=int, default=10)
    parser.add_argument("--nccl-warmup", type=int, default=2)
    parser.add_argument("--nccl-repeats", type=int, default=3, help="Repeat each nccl op and aggregate fit with median.")
    parser.add_argument("--fixed-latency-ws", default="2,4,8")
    parser.add_argument("--fixed-latency-comm-mb", type=int, default=32)
    parser.add_argument("--fixed-latency-block-count", type=int, default=20)
    parser.add_argument("--fixed-latency-window-start", type=int, default=5)
    parser.add_argument("--fixed-latency-window-len", type=int, default=10)
    parser.add_argument("--fixed-latency-measure-iters", type=int, default=20)
    parser.add_argument("--fixed-latency-warmup-iters", type=int, default=10)
    parser.add_argument("--fixed-latency-threshold-us", type=float, default=25.0)
    parser.add_argument("--fixed-latency-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "one_click_outputs"))
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args()


def infer_device_bw_gbps() -> tuple[float, str]:
    # References:
    # - NVIDIA HGX specs list B200 "NVLink GPU-to-GPU Bandwidth 1.8 TB/s".
    # - Here we use unidirectional baseline for fitting/config.
    #   https://www.nvidia.com/en-us/data-center/hgx/
    try:
        import torch
    except Exception:
        return 900.0, "fallback:no_torch"

    name = ""
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        name = str(torch.cuda.get_device_name(0)).upper()
    elif hasattr(torch, "musa") and torch.musa.is_available():
        name = str(torch.musa.get_device_name(0)).upper()

    if "B200" in name or "BLACKWELL" in name:
        return 900.0, f"auto:{name}"
    if "H100" in name or "H200" in name:
        return 450.0, f"auto:{name}"
    if "A100" in name:
        return 300.0, f"auto:{name}"
    return 900.0, f"fallback:{name or 'unknown'}"


def infer_mem_bw_gbps() -> tuple[float, str]:
    try:
        import torch
    except Exception:
        return 1600.0, "fallback:no_torch"
    name = ""
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        name = str(torch.cuda.get_device_name(0)).upper()
    elif hasattr(torch, "musa") and torch.musa.is_available():
        name = str(torch.musa.get_device_name(0)).upper()
    if "B200" in name or "BLACKWELL" in name:
        return 8000.0, f"auto:{name}"
    if "H100" in name or "H200" in name:
        return 3350.0, f"auto:{name}"
    if "A100" in name:
        return 2039.0, f"auto:{name}"
    return 1600.0, f"fallback:{name or 'unknown'}"


def apply_accelerator_mem_bw(system_json_path: Path, mem_bw_gbps: float) -> None:
    payload = json.loads(system_json_path.read_text(encoding="utf-8"))
    accel = payload.get("accelerator", {})
    bw = accel.get("bandwidth", {})
    if isinstance(bw, dict):
        for key, val in bw.items():
            if isinstance(val, dict):
                val["gbps"] = float(mem_bw_gbps)
    system_json_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def run_fixed_latency_case(
    *,
    op_name: str,
    ws: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    spec = COLLECTIVE_SPECS[op_name]
    cmd = [
        args.python_exe,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(ws),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(29000 + ws * 37 + int(datetime.now(timezone.utc).timestamp()) % 97),
        str(WORKER),
        "--comm-op",
        spec["torch_op"],
        "--warmup-iters",
        str(args.fixed_latency_warmup_iters),
        "--measure-iters",
        str(args.fixed_latency_measure_iters),
        "--block-count",
        str(args.fixed_latency_block_count),
        "--window-start",
        str(args.fixed_latency_window_start),
        "--window-len",
        str(args.fixed_latency_window_len),
        "--comm-mb",
        str(args.fixed_latency_comm_mb),
        "--dtype",
        args.fixed_latency_dtype,
        "--output-dir",
        str(out_dir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    (out_dir / "torchrun_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "torchrun_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"fixed-latency case failed: {op_name} ws{ws}, see {out_dir}")
    rank_payloads = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("rank*.json"))]
    totals, bubbles = [], []
    for payload in rank_payloads:
        for item in payload["iteration_results"]:
            totals.append(item["window_total_ms"] / args.fixed_latency_window_len)
            bubbles.append(item["window_bubble_ms"] / args.fixed_latency_window_len)
    return {
        "case_dir": str(out_dir),
        "per_comm_total_ms": {
            "mean": statistics.mean(totals),
            "std": statistics.pstdev(totals),
            "min": min(totals),
            "max": max(totals),
        },
        "per_comm_bubble_ms": {
            "mean": statistics.mean(bubbles),
            "std": statistics.pstdev(bubbles),
            "min": min(bubbles),
            "max": max(bubbles),
        },
    }


def main() -> None:
    args = parse_args()
    if args.device_bw_gbps is None:
        device_bw_gbps, bw_source = infer_device_bw_gbps()
    else:
        device_bw_gbps, bw_source = float(args.device_bw_gbps), "arg"
    mem_bw_gbps, mem_bw_source = infer_mem_bw_gbps()
    output_root = Path(args.output_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = output_root / f"{args.sys_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at_utc": stamp,
        "repo_root": str(REPO_ROOT),
        "sys_name": args.sys_name,
        "run_dir": str(run_dir),
        "device_bw_gbps": device_bw_gbps,
        "device_bw_source": bw_source,
        "mem_bw_gbps": mem_bw_gbps,
        "mem_bw_source": mem_bw_source,
        "steps": {},
    }

    param_file = Path(args.param_file)
    if not param_file.is_absolute():
        param_file = (REPO_ROOT / param_file).resolve()

    generated_system = SCRIPT_DIR / f"{args.sys_name}.json"

    if not args.skip_compute:
        compute_cache_tag = args.compute_cache_tag.strip()
        if args.compute_cache_mode == "rebuild" and not compute_cache_tag:
            compute_cache_tag = f"{args.sys_name}_{stamp}"
        generated_system = run_compute_pipeline(
            repo_root=REPO_ROOT,
            eff_dir=SCRIPT_DIR,
            max_tflops=args.max_tflops,
            sys_name=args.sys_name,
            pcie_intra_link=args.pcie_intra_link,
            fc8_mode=args.fc8_mode,
            param_file=param_file,
            scripts=[
                "test_gemm_efficiency.py",
                "test_grouped_gemm_efficiency.py",
                "test_fa_efficiency.py",
                "combine_efficiency.py",
            ],
            compute_cache_mode=args.compute_cache_mode,
            compute_cache_tag=(compute_cache_tag or None),
            python_exe=args.python_exe,
        )
        summary["steps"]["compute"] = {
            "ok": True,
            "system_json": str(generated_system),
            "compute_cache_mode": args.compute_cache_mode,
            "compute_cache_tag": compute_cache_tag,
        }
    elif not generated_system.exists():
        raise FileNotFoundError(f"--skip-compute was set but system json does not exist: {generated_system}")
    apply_accelerator_mem_bw(generated_system, mem_bw_gbps)

    if not args.skip_comm:
        nccl_bin_dir = Path(args.nccl_bin_dir).resolve() if args.nccl_bin_dir else None
        comm_results = run_nccl_collective_suite(
            work_dir=SCRIPT_DIR,
            output_dir=run_dir / "comm_raw",
            nccl_bin_dir=nccl_bin_dir,
            world_size=args.nccl_world_size,
            begin=args.nccl_begin,
            end=args.nccl_end,
            fallback_end=(args.nccl_fallback_end or None),
            dtype=args.nccl_dtype,
            num_iters=args.nccl_iters,
            warmup_iters=args.nccl_warmup,
            repeats=args.nccl_repeats,
        )
        summary["steps"]["comm"] = comm_results
    else:
        comm_results = summary.get("steps", {}).get("comm", {})

    fit_results = {}
    if not args.skip_fit:
        if not comm_results:
            raise RuntimeError("fit step requires comm logs; remove --skip-comm or pass precomputed logs flow.")
        failed = []
        for op_name, item in comm_results.items():
            if item["return_code"] != 0:
                failed.append(op_name)
                continue
            ws = item["world_size"]
            scale = item["scale"]
            logs = item.get("run_logs") or [item["log_path"]]
            per_run = []
            for log_path in logs:
                rows = parse_nccl_rows(Path(log_path))
                per_run.append(fit_bw_latency(rows, scale=scale, world_size=ws, fit_count=11))
            valid = [x for x in per_run if x["bw_gbps"] > 0 and x["slope_us_per_byte"] > 0]
            if not valid:
                failed.append(op_name)
                continue
            fit = {
                "fit_count": int(statistics.median([x["fit_count"] for x in valid])),
                "bw_gbps": float(statistics.median([x["bw_gbps"] for x in valid])),
                "base_latency_us": float(statistics.median([x["base_latency_us"] for x in valid])),
                "slope_us_per_byte": float(statistics.median([x["slope_us_per_byte"] for x in valid])),
                "intercept_us": float(statistics.median([x["intercept_us"] for x in valid])),
                "world_size": ws,
                "collective_scale": scale,
                "repeats": len(valid),
                "per_run": per_run,
            }
            fit_results[op_name] = fit
        summary["steps"]["fit"] = fit_results
        summary["steps"]["fit_failed_ops"] = failed

        update_system_network_from_fit(
            system_json_path=generated_system,
            fit_result=fit_results,
            device_bw_gbps=device_bw_gbps,
        )

    fixed_latency_summary = {}
    if not args.skip_fixed_latency:
        ws_list = [int(x.strip()) for x in args.fixed_latency_ws.split(",") if x.strip()]
        ws_list = sorted(set(ws_list))
        if not ws_list:
            raise ValueError("empty --fixed-latency-ws")

        payload_bytes = args.fixed_latency_comm_mb * 1024 * 1024
        base_system_dict = json.loads(Path(generated_system).read_text(encoding="utf-8"))
        temp_system_dict = json.loads(Path(generated_system).read_text(encoding="utf-8"))
        for net_name, net_dict in temp_system_dict.get("networks", {}).items():
            if not isinstance(net_dict, dict):
                continue
            op_dict = net_dict.get("op", {})
            if not isinstance(op_dict, dict):
                continue
            for spec in COLLECTIVE_SPECS.values():
                cfg = op_dict.get(spec["config_op"])
                if isinstance(cfg, dict):
                    cfg["fixed_latency_us"] = 0.0
                    cfg["fixed_latency_us_by_comm_num"] = {}
        temp_system = SystemConfig.init_from_dict(temp_system_dict)

        fl_root = run_dir / "fixed_latency_runs"
        fl_root.mkdir(parents=True, exist_ok=True)

        for op_name, spec in COLLECTIVE_SPECS.items():
            per_ws = {}
            fixed_values = []
            for ws in ws_list:
                case_dir = fl_root / f"{op_name}_ws{ws}"
                case_dir.mkdir(parents=True, exist_ok=True)
                event = run_fixed_latency_case(op_name=op_name, ws=ws, out_dir=case_dir, args=args)
                simumax_wo_fixed_ms = temp_system.compute_net_op_time(
                    spec["config_op"],
                    payload_bytes,
                    ws,
                    net="high_intra_node",
                    comm_stage="tp",
                )
                fixed_latency_us = max((event["per_comm_total_ms"]["mean"] - simumax_wo_fixed_ms) * 1e3, 0.0)
                fixed_values.append(fixed_latency_us)
                per_ws[str(ws)] = {
                    "event_per_comm_ms": event["per_comm_total_ms"]["mean"],
                    "event_per_comm_std_ms": event["per_comm_total_ms"]["std"],
                    "event_per_comm_bubble_ms": event["per_comm_bubble_ms"]["mean"],
                    "simumax_wo_fixed_ms": simumax_wo_fixed_ms,
                    "fixed_latency_us": fixed_latency_us,
                    "event_case_dir": event["case_dir"],
                }

            spread_us = max(fixed_values) - min(fixed_values)
            small_gap = spread_us <= args.fixed_latency_threshold_us
            if small_gap:
                scalar = sum(fixed_values) / len(fixed_values)
                decision = {"mode": "scalar", "fixed_latency_us": scalar, "spread_us": spread_us}
            else:
                decision = {
                    "mode": "by_comm_num",
                    "fixed_latency_us_by_comm_num": {k: v["fixed_latency_us"] for k, v in per_ws.items()},
                    "spread_us": spread_us,
                }

            for net_name, net_dict in base_system_dict.get("networks", {}).items():
                if not isinstance(net_dict, dict) or net_name == "inter_node":
                    continue
                op_cfg = net_dict.get("op", {}).get(spec["config_op"])
                if not isinstance(op_cfg, dict):
                    continue
                if decision["mode"] == "scalar":
                    op_cfg["fixed_latency_us"] = round(decision["fixed_latency_us"], 4)
                    op_cfg["fixed_latency_us_by_comm_num"] = {}
                else:
                    op_cfg["fixed_latency_us"] = 0.0
                    op_cfg["fixed_latency_us_by_comm_num"] = {
                        k: round(v, 4) for k, v in decision["fixed_latency_us_by_comm_num"].items()
                    }

            fixed_latency_summary[op_name] = {
                "per_ws": per_ws,
                "decision": decision,
            }

        Path(generated_system).write_text(json.dumps(base_system_dict, indent=4), encoding="utf-8")
        summary["steps"]["fixed_latency"] = {
            "ws_list": ws_list,
            "comm_mb": args.fixed_latency_comm_mb,
            "threshold_us": args.fixed_latency_threshold_us,
            "ops": fixed_latency_summary,
        }

    system_copy = run_dir / f"{args.sys_name}.json"
    shutil.copyfile(generated_system, system_copy)
    summary["final_system_json"] = str(system_copy)

    compact_fit = {}
    for op_name, fit in fit_results.items():
        config_op = COLLECTIVE_SPECS.get(op_name, {}).get("config_op", "p2p")
        compact_fit[op_name] = {
            "config_op": config_op,
            "bw_gbps": round_sig(fit["bw_gbps"]),
            "efficient_factor": round_sig(fit["bw_gbps"] / device_bw_gbps),
            "latency_us": round_sig(max(fit["base_latency_us"], 0.0)),
        }
    summary["fit_compact"] = compact_fit

    dump_json(run_dir / "summary.json", summary)
    print(system_copy)
    print(run_dir / "summary.json")


if __name__ == "__main__":
    main()
