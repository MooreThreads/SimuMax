from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


COLLECTIVE_SPECS = {
    "all_reduce": {
        "bin": "all_reduce_perf",
        "scale": 2.0,
        "config_op": "all_reduce",
        "torch_op": "all_reduce",
    },
    "all_gather": {
        "bin": "all_gather_perf",
        "scale": 1.0,
        "config_op": "all_gather",
        "torch_op": "all_gather",
    },
    "reduce_scatter": {
        "bin": "reduce_scatter_perf",
        "scale": 1.0,
        "config_op": "reduce_scatter",
        "torch_op": "reduce_scatter",
    },
    "alltoall": {
        "bin": "alltoall_perf",
        "scale": 1.0,
        "config_op": "all2all",
        "torch_op": "all_to_all",
    },
}
P2P_SPEC = {"bin": "sendrecv_perf", "scale": 1.0, "config_op": "p2p"}


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)


def run_compute_pipeline(
    *,
    repo_root: Path,
    eff_dir: Path,
    max_tflops: float,
    sys_name: str,
    pcie_intra_link: int,
    fc8_mode: int,
    param_file: Path,
    scripts: list[str],
    compute_cache_mode: str = "supplement",
    compute_cache_tag: str | None = None,
    python_exe: str | None = None,
) -> Path:
    if python_exe is None:
        python_exe = sys.executable

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    env["MAX_TFLOPS"] = str(max_tflops)
    env["SYS_NAME"] = sys_name
    env["PICE_INTRA_LINK"] = str(pcie_intra_link)
    env["FC8_MODE"] = str(fc8_mode)
    env["PARAM_FILE"] = str(param_file.resolve())
    if compute_cache_mode not in {"supplement", "rebuild"}:
        raise ValueError(f"unsupported compute_cache_mode: {compute_cache_mode}")
    env["EFFICIENCY_CACHE_MODE"] = compute_cache_mode
    # One-click only uses two user-facing modes:
    # - supplement: reuse the selected cache namespace and fill missing entries
    # - rebuild: benchmark into a fresh cache namespace
    # In both cases, duplicates within the same run should be skipped and we do
    # not rely on in-place overwrite semantics for existing cache entries.
    env["EFFICIENCY_OVERWRITE"] = "0"
    if compute_cache_tag:
        env["EFFICIENCY_CACHE_TAG"] = str(compute_cache_tag)

    for script in scripts:
        proc = run_cmd([python_exe, script], cwd=eff_dir, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"compute step failed: {script}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

    generated = eff_dir / f"{sys_name}.json"
    if not generated.exists():
        raise FileNotFoundError(f"generated system json not found: {generated}")
    return generated


def _resolve_binary(bin_name: str, bin_dir: Path | None) -> str:
    if bin_dir is not None:
        candidate = bin_dir / bin_name
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"nccl binary not found: {candidate}")
    return bin_name


def run_nccl_collective_suite(
    *,
    work_dir: Path,
    output_dir: Path,
    nccl_bin_dir: Path | None,
    world_size: int,
    begin: str,
    end: str,
    fallback_end: str | None,
    dtype: str,
    num_iters: int,
    warmup_iters: int,
    repeats: int = 1,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    results = {}

    def _run_one(name: str, exe: str, args: list[str], out_name: str) -> tuple[int, str]:
        proc = run_cmd([exe] + args, cwd=work_dir, env=env)
        out_path = output_dir / out_name
        out_path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
        return proc.returncode, str(out_path)

    for op_name, spec in COLLECTIVE_SPECS.items():
        exe = _resolve_binary(spec["bin"], nccl_bin_dir)
        args = [
            "-n", str(num_iters), "-b", begin, "-e", end, "-f", "2",
            "-g", str(world_size), "-w", str(warmup_iters), "-d", dtype,
        ]
        run_logs: list[str] = []
        code = 0
        used_end = end
        fallback_used = False
        log_path = ""
        for ridx in range(repeats):
            code, log_path = _run_one(op_name, exe, args, f"{op_name}_r{ridx+1}.txt")
            run_logs.append(log_path)
            if code != 0:
                break
        if code != 0 and fallback_end:
            args[5] = fallback_end
            run_logs = []
            for ridx in range(repeats):
                code, log_path = _run_one(op_name, exe, args, f"{op_name}_fallback_r{ridx+1}.txt")
                run_logs.append(log_path)
                if code != 0:
                    break
            used_end = fallback_end
            fallback_used = True
        results[op_name] = {
            "return_code": code,
            "log_path": log_path,
            "run_logs": run_logs,
            "world_size": world_size,
            "scale": spec["scale"],
            "used_end": used_end,
            "fallback_used": fallback_used,
        }

    # P2P
    p2p_exe = _resolve_binary(P2P_SPEC["bin"], nccl_bin_dir)
    p2p_args = [
        "-n", "1",
        "-b", begin,
        "-e", end,
        "-f", "2",
        "-g", "1",
        "-t", "1",
        "-d", dtype,
    ]
    p2p_logs: list[str] = []
    p2p_code = 0
    p2p_log = ""
    for ridx in range(repeats):
        p2p_code, p2p_log = _run_one("sendrecv", p2p_exe, p2p_args, f"sendrecv_r{ridx+1}.txt")
        p2p_logs.append(p2p_log)
        if p2p_code != 0:
            break
    p2p_used_end = end
    p2p_fallback_used = False
    if p2p_code != 0 and fallback_end:
        p2p_args[5] = fallback_end
        p2p_logs = []
        for ridx in range(repeats):
            p2p_code, p2p_log = _run_one("sendrecv", p2p_exe, p2p_args, f"sendrecv_fallback_r{ridx+1}.txt")
            p2p_logs.append(p2p_log)
            if p2p_code != 0:
                break
        p2p_used_end = fallback_end
        p2p_fallback_used = True
    results["sendrecv"] = {
        "return_code": p2p_code,
        "log_path": p2p_log,
        "run_logs": p2p_logs,
        "world_size": 2,
        "scale": P2P_SPEC["scale"],
        "used_end": p2p_used_end,
        "fallback_used": p2p_fallback_used,
    }
    return results


_NUM = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")


def _is_num(x: str) -> bool:
    return bool(_NUM.match(x))


def parse_nccl_rows(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        # standard nccl-tests columns: size ... time algbw busbw ...
        if not (_is_num(parts[5]) and _is_num(parts[6]) and _is_num(parts[7])):
            continue
        rows.append([float(parts[0]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not rows:
        raise RuntimeError(f"no valid nccl rows parsed from {path}")
    return np.array(rows, dtype=float)


def fit_bw_latency(rows: np.ndarray, *, scale: float, world_size: int, fit_count: int = 11) -> dict:
    x = rows[:, 0] * (world_size - 1) / world_size * scale
    y = rows[:, 1]
    n = min(fit_count, len(x))
    a, b = np.polyfit(x[:n], y[:n], 1)
    bw_gbps = 1.0 / a / (1024 ** 3) * (1000 ** 2)
    base_latency_us = b / max((world_size - 1) * scale, 1.0)
    return {
        "fit_count": n,
        "bw_gbps": float(bw_gbps),
        "base_latency_us": float(base_latency_us),
        "slope_us_per_byte": float(a),
        "intercept_us": float(b),
    }


def update_system_network_from_fit(
    *,
    system_json_path: Path,
    fit_result: dict,
    device_bw_gbps: float,
) -> None:
    payload = json.loads(system_json_path.read_text(encoding="utf-8"))
    networks = payload.get("networks", {})

    def _apply_op(net_dict: dict, config_op: str, fit_item: dict) -> None:
        op_dict = net_dict.get("op", {}).get(config_op)
        if not isinstance(op_dict, dict):
            return
        op_dict["efficient_factor"] = round(fit_item["bw_gbps"] / device_bw_gbps, 4)
        latency_us = fit_item.get("base_latency_us")
        collective_config_ops = {spec["config_op"] for spec in COLLECTIVE_SPECS.values()}
        if config_op in collective_config_ops and "intercept_us" in fit_item and "world_size" in fit_item:
            op_scale = float(op_dict.get("scale", fit_item.get("collective_scale", 1.0)))
            op_offset = float(op_dict.get("offset", -1.0))
            world_size = int(fit_item["world_size"])
            denominator = max((world_size + op_offset) * op_scale, 1.0)
            latency_us = float(fit_item["intercept_us"]) / denominator
        op_dict["latency_us"] = round(max(latency_us, 0.0), 4)

    for net_name, net_dict in networks.items():
        if not isinstance(net_dict, dict) or net_name == "inter_node":
            continue
        bw_dict = net_dict.get("bandwidth")
        if isinstance(bw_dict, dict):
            # Keep one nominal intra-node bandwidth and model op differences via op-level efficiency.
            bw_dict["gbps"] = float(device_bw_gbps)
            bw_dict["efficient_factor"] = 1.0
        for op_key, spec in COLLECTIVE_SPECS.items():
            if op_key in fit_result:
                _apply_op(net_dict, spec["config_op"], fit_result[op_key])
        if "sendrecv" in fit_result:
            _apply_op(net_dict, P2P_SPEC["config_op"], fit_result["sendrecv"])

    system_json_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def round_sig(x: float, digits: int = 4) -> float:
    if x == 0:
        return 0.0
    return round(x, digits - int(math.floor(math.log10(abs(x)))) - 1)
