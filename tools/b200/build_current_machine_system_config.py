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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from simu_tools.efficency_test.run_one_click_benchmark import (
    apply_accelerator_mem_bw,
    infer_device_bw_gbps,
    infer_mem_bw_gbps,
    run_fixed_latency_case,
)


EFF_DIR = REPO_ROOT / "simu_tools" / "efficency_test"
DEFAULT_SYSTEM = REPO_ROOT / "configs" / "system" / "b200_bf16_ceperm.json"
DEFAULT_PARAM_FILE = REPO_ROOT / "tools" / "b200" / "run_params" / "b200_bf16_ceperm_sweep.json"
DEFAULT_CEPERM_SUMMARY = EFF_DIR / "one_click_outputs" / "b200_ce_permute_summary.json"
DEFAULT_OUTPUT_ROOT = EFF_DIR / "one_click_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the public B200 system config from compute efficiency, "
            "NCCL communication fitting, fixed-latency calibration, and "
            "CE/permute supplementation. This path uses the shared calibration "
            "pipeline with a B200-specific shape sweep."
        )
    )
    parser.add_argument("--max-tflops", type=float, default=2250.0)
    parser.add_argument("--sys-name", default="b200_bf16_ceperm")
    parser.add_argument(
        "--param-file",
        default=str(DEFAULT_PARAM_FILE.relative_to(REPO_ROOT)),
        help="Repo-relative public B200 shape sweep definition.",
    )
    parser.add_argument(
        "--output-system",
        default=str(DEFAULT_SYSTEM.relative_to(REPO_ROOT)),
        help="Repo-relative path for the final public B200 system config.",
    )
    parser.add_argument(
        "--ceperm-output",
        default=str(DEFAULT_CEPERM_SUMMARY.relative_to(REPO_ROOT)),
        help="Repo-relative path for the CE/permute measurement summary artifact.",
    )
    parser.add_argument("--compute-cache-mode", default="supplement", choices=["supplement", "rebuild"])
    parser.add_argument("--compute-cache-tag", default="")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--measure-nonfusion-ce",
        action="store_true",
        help="Also measure Megatron nonfusion CE and update accelerator.bandwidth.ce. Disabled by default.",
    )
    parser.add_argument(
        "--megatron-root",
        default="",
        help=(
            "Optional Megatron-LM checkout for compute-efficiency FA/MLA coverage. "
            "Also used for Megatron nonfusion CE when --measure-nonfusion-ce is set."
        ),
    )
    parser.add_argument("--device-bw-gbps", type=float, default=None, help="Nominal intra-node link bandwidth for communication efficient_factor fit. If unset, infer from GPU model.")
    parser.add_argument("--nccl-bin-dir", default="", help="Directory containing nccl-tests binaries; empty means use PATH.")
    parser.add_argument("--nccl-begin", default="1M")
    parser.add_argument("--nccl-end", default="8G")
    parser.add_argument("--nccl-fallback-end", default="256M")
    parser.add_argument("--nccl-world-size", type=int, default=8)
    parser.add_argument("--nccl-dtype", default="bfloat16")
    parser.add_argument("--nccl-iters", type=int, default=10)
    parser.add_argument("--nccl-warmup", type=int, default=2)
    parser.add_argument("--nccl-repeats", type=int, default=3, help="Repeat each NCCL op and aggregate fit with median.")
    parser.add_argument("--fixed-latency-ws", default="2,4,8")
    parser.add_argument("--fixed-latency-comm-mb", type=int, default=32)
    parser.add_argument("--fixed-latency-block-count", type=int, default=20)
    parser.add_argument("--fixed-latency-window-start", type=int, default=5)
    parser.add_argument("--fixed-latency-window-len", type=int, default=10)
    parser.add_argument("--fixed-latency-measure-iters", type=int, default=20)
    parser.add_argument("--fixed-latency-warmup-iters", type=int, default=10)
    parser.add_argument("--fixed-latency-threshold-us", type=float, default=25.0)
    parser.add_argument("--fixed-latency-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT.relative_to(REPO_ROOT)))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--skip-compute", action="store_true")
    parser.add_argument("--skip-comm", action="store_true")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-fixed-latency", action="store_true")
    parser.add_argument("--skip-ceperm", action="store_true")
    parser.add_argument(
        "--allow-partial-comm-fit",
        action="store_true",
        help="Allow writing a system config when only a subset of NCCL communication fits succeed.",
    )
    return parser.parse_args()


def apply_ceperm_recommendations(system_path: Path, summary_path: Path) -> None:
    system = json.loads(system_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recommended = summary["recommended_efficient_factor"]
    bw = system.setdefault("accelerator", {}).setdefault("bandwidth", {})

    required = ("ce_fusion", "permute_fwd", "permute_bwd")
    for key in required:
        if key not in bw:
            raise KeyError(f"missing accelerator.bandwidth.{key} in {system_path}")
        if key not in recommended:
            raise KeyError(f"missing recommended_efficient_factor.{key} in {summary_path}")

    for key in ("ce", "ce_fusion", "permute_fwd", "permute_bwd"):
        if key not in recommended:
            continue
        if key not in bw:
            raise KeyError(f"missing accelerator.bandwidth.{key} in {system_path}")
        bw[key]["efficient_factor"] = float(recommended[key])

    system_path.write_text(json.dumps(system, indent=4), encoding="utf-8")


def refresh_run_dir_final_artifact(
    *,
    summary_path: Path,
    system_path: Path,
    ceperm_summary_path: Path | None = None,
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_system = Path(summary["final_system_json"])
    shutil.copyfile(system_path, final_system)
    if ceperm_summary_path is not None:
        summary.setdefault("steps", {})["ceperm"] = {
            "ok": True,
            "summary_json": str(ceperm_summary_path),
        }
    summary["final_system_json"] = str(final_system)
    dump_json(summary_path, summary)


def apply_comm_generation(system_path: Path, args: argparse.Namespace) -> Path:
    if args.device_bw_gbps is None:
        device_bw_gbps, bw_source = infer_device_bw_gbps()
    else:
        device_bw_gbps, bw_source = float(args.device_bw_gbps), "arg"
    mem_bw_gbps, mem_bw_source = infer_mem_bw_gbps()
    apply_accelerator_mem_bw(system_path, mem_bw_gbps)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = output_root / f"{args.sys_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at_utc": stamp,
        "repo_root": str(REPO_ROOT),
        "sys_name": args.sys_name,
        "run_dir": str(run_dir),
        "system_json": str(system_path),
        "device_bw_gbps": device_bw_gbps,
        "device_bw_source": bw_source,
        "mem_bw_gbps": mem_bw_gbps,
        "mem_bw_source": mem_bw_source,
        "steps": {},
    }

    comm_results = {}
    if not args.skip_comm:
        nccl_bin_dir = Path(args.nccl_bin_dir).resolve() if args.nccl_bin_dir else None
        comm_results = run_nccl_collective_suite(
            work_dir=EFF_DIR,
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

    fit_results = {}
    if not args.skip_fit:
        if not comm_results:
            raise RuntimeError("fit step requires comm logs; remove --skip-comm or pass --skip-fit too.")
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
        if failed and not args.allow_partial_comm_fit:
            summary_path = run_dir / "summary.json"
            dump_json(summary_path, summary)
            failed_ops = ", ".join(sorted(failed))
            raise RuntimeError(
                f"communication fit failed for op(s): {failed_ops}. "
                f"Summary written to {summary_path}. "
                "Use --allow-partial-comm-fit only if you intentionally want a mixed old/new network config."
            )
        update_system_network_from_fit(
            system_json_path=system_path,
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
        base_system_dict = json.loads(system_path.read_text(encoding="utf-8"))
        temp_system_dict = json.loads(system_path.read_text(encoding="utf-8"))
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

        system_path.write_text(json.dumps(base_system_dict, indent=4), encoding="utf-8")
        summary["steps"]["fixed_latency"] = {
            "ws_list": ws_list,
            "comm_mb": args.fixed_latency_comm_mb,
            "threshold_us": args.fixed_latency_threshold_us,
            "ops": fixed_latency_summary,
        }

    system_copy = run_dir / f"{args.sys_name}.json"
    shutil.copyfile(system_path, system_copy)
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

    summary_path = run_dir / "summary.json"
    dump_json(summary_path, summary)
    return summary_path


def main() -> None:
    args = parse_args()
    args.python_exe = args.python_exe or sys.executable
    output_system = (REPO_ROOT / args.output_system).resolve()
    param_file = (REPO_ROOT / args.param_file).resolve()
    ceperm_output = (REPO_ROOT / args.ceperm_output).resolve()

    if not args.skip_compute:
        generated = run_compute_pipeline(
            repo_root=REPO_ROOT,
            eff_dir=EFF_DIR,
            max_tflops=args.max_tflops,
            sys_name=args.sys_name,
            pcie_intra_link=0,
            fc8_mode=0,
            param_file=param_file,
            scripts=[
                "test_gemm_efficiency.py",
                "test_grouped_gemm_efficiency.py",
                "test_fa_efficiency.py",
                "combine_efficiency.py",
            ],
            compute_cache_mode=args.compute_cache_mode,
            compute_cache_tag=(args.compute_cache_tag or None),
            megatron_root=(args.megatron_root or None),
            python_exe=args.python_exe,
        )
    else:
        generated = EFF_DIR / f"{args.sys_name}.json"

    if not generated.exists():
        raise FileNotFoundError(f"generated B200 compute-efficiency system config not found: {generated}")

    output_system.parent.mkdir(parents=True, exist_ok=True)
    if generated.resolve() != output_system:
        shutil.copyfile(generated, output_system)

    comm_summary = None
    if not (args.skip_comm and args.skip_fit and args.skip_fixed_latency):
        comm_summary = apply_comm_generation(output_system, args)

    if not args.skip_ceperm:
        ceperm_output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_exe,
            str(EFF_DIR / "test_ce_permute_efficiency.py"),
            "--system",
            str(output_system.relative_to(REPO_ROOT)),
            "--output",
            str(ceperm_output.relative_to(REPO_ROOT)),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        if args.measure_nonfusion_ce:
            cmd.append("--measure-nonfusion-ce")
            if args.megatron_root:
                cmd.extend(["--megatron-root", args.megatron_root])
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        apply_ceperm_recommendations(output_system, ceperm_output)
        if comm_summary is not None:
            refresh_run_dir_final_artifact(
                summary_path=comm_summary,
                system_path=output_system,
                ceperm_summary_path=ceperm_output,
            )

    print(output_system)
    if comm_summary is not None:
        print(comm_summary)


if __name__ == "__main__":
    main()
