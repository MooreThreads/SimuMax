#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simu_tools.efficency_test.one_click_common import run_compute_pipeline


EFF_DIR = REPO_ROOT / "simu_tools" / "efficency_test"
DEFAULT_SYSTEM = REPO_ROOT / "configs" / "system" / "b200_bf16_ceperm.json"
DEFAULT_PARAM_FILE = REPO_ROOT / "tools" / "b200" / "run_params" / "b200_bf16_ceperm_sweep.json"
DEFAULT_CEPERM_SUMMARY = EFF_DIR / "one_click_outputs" / "b200_ce_permute_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the public B200 system config from compute efficiency and "
            "CE/permute supplementation. This path uses the shared compute "
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
    parser.add_argument("--skip-compute", action="store_true")
    parser.add_argument("--skip-ceperm", action="store_true")
    return parser.parse_args()


def apply_ceperm_recommendations(system_path: Path, summary_path: Path) -> None:
    system = json.loads(system_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recommended = summary["recommended_efficient_factor"]
    bw = system.setdefault("accelerator", {}).setdefault("bandwidth", {})

    for key in ("ce", "ce_fusion", "permute_fwd", "permute_bwd"):
        if key not in bw:
            raise KeyError(f"missing accelerator.bandwidth.{key} in {system_path}")
        if key not in recommended:
            raise KeyError(f"missing recommended_efficient_factor.{key} in {summary_path}")
        bw[key]["efficient_factor"] = float(recommended[key])

    system_path.write_text(json.dumps(system, indent=4), encoding="utf-8")


def main() -> None:
    args = parse_args()
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
            python_exe=sys.executable,
        )
    else:
        generated = EFF_DIR / f"{args.sys_name}.json"

    if not generated.exists():
        raise FileNotFoundError(f"generated B200 compute-efficiency system config not found: {generated}")

    output_system.parent.mkdir(parents=True, exist_ok=True)
    if generated.resolve() != output_system:
        shutil.copyfile(generated, output_system)

    if not args.skip_ceperm:
        ceperm_output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
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
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        apply_ceperm_recommendations(output_system, ceperm_output)

    print(output_system)


if __name__ == "__main__":
    main()
