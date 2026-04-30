"""Sweep MFU and PP utilization across (model × pp_schedule) cells.

For each (model, pp_schedule) pair this script can run three phases:

1. **nominal** — no disturbances, 1 deterministic episode. Uses the strategy
   config ``<model>_optimal_mfu_<schedule>``.
2. **baseline** — full base disturbance profile, ``--episodes`` runs varying
   only the disturbance seed (matches eval_agents.py / run_gantt_demo.py
   semantics). Uses the strategy config
   ``<model>_optimal_mfu_<schedule>_both`` (the optimum found under
   disturbance) and the disturbance config selected by ``--base-disturbance``
   (default ``both``).
3. **ablation** — one-axis-at-a-time sweep of four disturbance knobs
   (``seq_len_std``, ``op_duration_std``, ``stage_slowdown_prob``,
   ``op_slowdown_prob``). Each axis runs an inclusive linspace ``[0, max]``
   with the swept knob set to the linspace value and the other three primary
   knobs zeroed. Auxiliary fields (``seq_len_mean``, ``stage_slowdown_k``,
   ``op_slowdown_k``, …) keep their baseline values so the swept knob has
   the same interpretation as under baseline. Strategy is the
   disturbance-tuned ``_both`` config, same as ``baseline``.

   The all-zero anchor (``ablation_value == 0``) is computed once per
   ``(model, schedule)`` (deterministic, 1 episode) and replicated into
   every per-axis output file at concat time. Non-zero points run
   ``--ablation-episodes`` runs each (default 30).

   Per-axis ranges:

   * ``seq_len_std``: 0..2.0, **multiplier × seq_len_mean** (so the actual
     ``DisturbanceConfig.seq_len_std`` applied is
     ``ablation_value × seq_len_mean``).
   * ``op_duration_std``: 0..baseline ``op_duration_std`` (raw).
   * ``stage_slowdown_prob``: 0..baseline ``stage_slowdown_prob`` (raw).
   * ``op_slowdown_prob``: 0..baseline ``op_slowdown_prob`` (raw).

Outputs per cell are written as per-(cell, phase) parquet shards under
``<output_dir>/_shards/`` and concatenated into long-format parquet files at
the end:

* ``nominal.parquet``
* ``baseline_disturbed.parquet``
* ``ablation_<axis>.parquet`` (one per swept axis; includes the shared zero
  anchor row stamped with ``ablation_var=<axis>``)

Re-running with ``--resume`` (default) skips shards that already exist on
disk, so killed runs can be picked up where they left off.

Example
-------
    uv run python examples/run_sweep.py \\
        --output-dir results/h100_nvlink \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium.utils import seeding

from simumax.core.config import (
    DisturbanceConfig,
    ModelConfig,
    PipelineScheduleConfig,
    StrategyConfig,
    SystemConfig,
)
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    RELEASE_MODELS,
    RELEASE_PP_SCHEDULING,
    get_simu_disturbance_config,
    get_simu_model_config,
    get_simu_pp_scheduling_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


VALID_PHASES = ("nominal", "baseline", "ablation")

ABLATION_AXES_DEFAULT: Tuple[str, ...] = (
    "seq_len_std", "op_duration_std",
    "stage_slowdown_prob", "op_slowdown_prob",
)

# Cap for the ``seq_len_std`` ablation_value (units: × seq_len_mean).
# 2.0 → applied std reaches 2 × seq_len_mean at the upper end of the sweep.
# The other three axes derive their max from the baseline disturbance config.
SEQ_LEN_STD_MULTIPLIER_MAX = 2.0


# ----------------------------------------------------------------------------
# Job specification
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    """One unit of work = one parquet shard for one (model, schedule, phase).

    Phase semantics:
      * ``nominal``        — nominal-optimal strategy, no disturbance, 1 episode.
      * ``baseline``       — disturbance-optimal strategy, full disturbance,
                             ``episodes`` runs.
      * ``ablation_zero``  — disturbance-optimal strategy, zero disturbance,
                             1 deterministic episode. Shared anchor across all
                             four ablation axes; replicated into each
                             ``ablation_<axis>.parquet`` at concat time.
      * ``ablation``       — disturbance-optimal strategy, single-axis
                             disturbance with all other primary knobs zeroed.
                             ``ablation_var`` and ``ablation_value`` set;
                             ``episodes`` runs.
    """
    model: str
    schedule: str
    phase: str
    episodes: int
    base_seed: int
    system: str
    base_disturbance_path: Optional[str]
    shard_path: str
    ablation_var: Optional[str] = None
    ablation_value: Optional[float] = None

    def shard_exists(self) -> bool:
        return os.path.exists(self.shard_path)


def _strategy_name_for(model: str, schedule: str, phase: str) -> str:
    safe = model.replace("-", "_").replace(".", "_")
    base = f"{safe}_optimal_mfu_{schedule}"
    if phase in ("baseline", "ablation", "ablation_zero"):
        return f"{base}_both"
    return base


def _shard_filename(output_dir: str, job_kind: str, model: str,
                    schedule: str, *, axis: Optional[str] = None,
                    idx: Optional[int] = None) -> str:
    safe_model = model.replace("/", "_")
    parts = [job_kind]
    if axis is not None:
        parts.append(axis)
    if idx is not None:
        parts.append(f"{idx:02d}")
    parts.extend([safe_model, schedule])
    return os.path.join(output_dir, "_shards",
                        "__".join(parts) + ".parquet")


def _ablation_axis_maxes(base_dict: Dict[str, Any]) -> Dict[str, float]:
    """Per-axis upper bound for ``ablation_value``. Sweep covers ``[0, max]``."""
    return {
        "seq_len_std":         SEQ_LEN_STD_MULTIPLIER_MAX,
        "op_duration_std":     float(base_dict["op_duration_std"]),
        "stage_slowdown_prob": float(base_dict["stage_slowdown_prob"]),
        "op_slowdown_prob":    float(base_dict["op_slowdown_prob"]),
    }


def build_jobs(models: List[str], schedules: List[str], phases: List[str],
               episodes: int, base_seed: int, system: str,
               base_disturbance_path: Optional[str],
               base_dict: Dict[str, Any],
               output_dir: str,
               ablation_vars: List[str],
               ablation_points: int,
               ablation_episodes: int) -> List[Job]:
    jobs: List[Job] = []
    axis_maxes = (_ablation_axis_maxes(base_dict)
                  if "ablation" in phases else {})
    for model in models:
        for schedule in schedules:
            if "nominal" in phases:
                jobs.append(Job(
                    model=model, schedule=schedule, phase="nominal",
                    episodes=1, base_seed=base_seed, system=system,
                    base_disturbance_path=None,
                    shard_path=_shard_filename(output_dir, "nominal",
                                               model, schedule),
                ))
            if "baseline" in phases:
                jobs.append(Job(
                    model=model, schedule=schedule, phase="baseline",
                    episodes=episodes, base_seed=base_seed, system=system,
                    base_disturbance_path=base_disturbance_path,
                    shard_path=_shard_filename(output_dir, "baseline",
                                               model, schedule),
                ))
            if "ablation" in phases:
                # Shared zero anchor (deterministic, 1 episode).
                jobs.append(Job(
                    model=model, schedule=schedule, phase="ablation_zero",
                    episodes=1, base_seed=base_seed, system=system,
                    base_disturbance_path=base_disturbance_path,
                    shard_path=_shard_filename(output_dir, "ablation_zero",
                                               model, schedule),
                    ablation_value=0.0,
                ))
                for axis in ablation_vars:
                    grid = np.linspace(0.0, axis_maxes[axis],
                                       ablation_points)
                    # Drop the leading 0; that point is the shared
                    # ablation_zero shard above.
                    for idx, val in enumerate(grid[1:], start=1):
                        jobs.append(Job(
                            model=model, schedule=schedule,
                            phase="ablation",
                            episodes=ablation_episodes,
                            base_seed=base_seed, system=system,
                            base_disturbance_path=base_disturbance_path,
                            shard_path=_shard_filename(
                                output_dir, "ablation", model, schedule,
                                axis=axis, idx=idx,
                            ),
                            ablation_var=axis,
                            ablation_value=float(val),
                        ))
    return jobs


# ----------------------------------------------------------------------------
# Per-cell execution (runs inside worker process)
# ----------------------------------------------------------------------------

_TIME_UNIT_TO_SECONDS = {
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9,
    "min": 60.0, "h": 3600.0,
}


def _parse_time_seconds(value: Any) -> float:
    """Convert analysis_cost's human-readable time strings (e.g. '27.2477 s',
    '813.5 ms') to seconds. Tolerates already-numeric inputs."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"unexpected time type: {type(value)!r}")
    parts = value.strip().split()
    if len(parts) != 2:
        raise ValueError(f"cannot parse time string: {value!r}")
    num, unit = parts
    if unit not in _TIME_UNIT_TO_SECONDS:
        raise ValueError(f"unknown time unit {unit!r} in {value!r}")
    return float(num) * _TIME_UNIT_TO_SECONDS[unit]


def _row_template(job: Job, strategy_name: str, perf_model: PerfLLM,
                  base_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": job.model,
        "strategy": strategy_name,
        "pp_schedule": perf_model.pp_scheduling.pp_schedule,
        "interleaving_size": perf_model.pp_scheduling.interleaving_size,
        "system": job.system,
        "phase": job.phase,
        "pp_size": perf_model.strategy.pp_size,
        "tp_size": perf_model.strategy.tp_size,
        "world_size": perf_model.strategy.world_size,
        "micro_batch_num": perf_model.strategy.micro_batch_num,
        "seq_len_strategy": perf_model.strategy.seq_len,
        "ablation_var": job.ablation_var,
        "ablation_value": job.ablation_value,
        "base_seq_len_mean": base_dict.get("seq_len_mean"),
        "base_seq_len_min": base_dict.get("seq_len_min"),
        "base_seq_len_max": base_dict.get("seq_len_max"),
        "base_stage_slowdown_k": base_dict.get("stage_slowdown_k"),
        "base_op_slowdown_k": base_dict.get("op_slowdown_k"),
        "base_op_slowdown_max_count": base_dict.get("op_slowdown_max_count"),
    }


def _peak_mem_gb(perf_model: PerfLLM) -> float:
    """Worst-stage peak memory (GB) for the just-completed run_estimate()."""
    mem_result = perf_model.analysis_mem()
    peak_list = perf_model.get_pp_stage_peak_mem(mem_result, "peak_mem", toG=True)
    return float(max(peak_list.values()))


def _run_episodes(perf_model: PerfLLM, episodes: int, base_seed: int) -> List[Dict[str, float]]:
    """Run `episodes` episodes mutating only disturbance.seed each iteration.

    Mirrors run_gantt_demo.py: a single seeded Generator feeds one fresh int
    per episode into perf_model.disturbance.seed.
    """
    rng, _ = seeding.np_random(base_seed)
    out: List[Dict[str, float]] = []
    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        perf_model.disturbance.seed = ep_seed
        perf_model.run_estimate()
        cost = perf_model.analysis_cost().data
        out.append({
            "episode_idx": ep,
            "disturbance_seed": ep_seed,
            "mfu": float(cost["mfu"]),
            "pp_utilization": float(cost["pp_utilization"]),
            "iter_time_s": _parse_time_seconds(cost["duration_time_per_iter"]),
            "peak_mem_gb": _peak_mem_gb(perf_model),
        })
    return out


def _build_ablation_disturbance(
    base_dict: Dict[str, Any], ablation_var: str,
    ablation_value: float, seed: int,
) -> DisturbanceConfig:
    """OAT disturbance: zero the four primary knobs, set the swept axis.

    Auxiliary fields (``seq_len_mean``, ``seq_len_min``, ``seq_len_max``,
    ``stage_slowdown_k``, ``op_slowdown_k``, ``op_slowdown_max_count``,
    op_duration clip factors, …) are inherited from ``base_dict``. Without
    them, e.g. ``stage_slowdown_prob > 0`` would do nothing because
    ``stage_slowdown_k`` would default to 1.0.

    For ``seq_len_std`` only, ``ablation_value`` is a multiplier ×
    ``seq_len_mean`` — the actual std written into ``DisturbanceConfig`` is
    ``ablation_value * base_dict['seq_len_mean']``.
    """
    cfg: Dict[str, Any] = dict(base_dict)
    cfg["seed"] = seed
    cfg["seq_len_std"] = 0.0
    cfg["op_duration_std"] = 0.0
    cfg["stage_slowdown_prob"] = 0.0
    cfg["op_slowdown_prob"] = 0.0
    if ablation_var == "seq_len_std":
        seq_len_mean = float(base_dict["seq_len_mean"])
        cfg["seq_len_std"] = float(ablation_value) * seq_len_mean
    elif ablation_var == "op_duration_std":
        cfg["op_duration_std"] = float(ablation_value)
    elif ablation_var == "stage_slowdown_prob":
        cfg["stage_slowdown_prob"] = float(ablation_value)
    elif ablation_var == "op_slowdown_prob":
        cfg["op_slowdown_prob"] = float(ablation_value)
    else:
        raise ValueError(f"unknown ablation_var: {ablation_var}")
    return DisturbanceConfig.init_from_dict(cfg)


def execute_job(job: Job) -> Dict[str, Any]:
    """Run one shard's worth of episodes; write the parquet shard."""
    t0 = time.time()
    try:
        strategy_name = _strategy_name_for(job.model, job.schedule, job.phase)

        strategy_path = get_simu_strategy_config(strategy_name)
        schedule_path = get_simu_pp_scheduling_config(job.schedule)
        model_path = get_simu_model_config(job.model)
        system_path = get_simu_system_config(job.system)

        if job.phase == "nominal":
            base_disturbance = DisturbanceConfig()
            base_dict: Dict[str, Any] = {}
        elif job.phase == "baseline":
            base_dict = json.loads(Path(job.base_disturbance_path).read_text())
            base_disturbance = DisturbanceConfig.init_from_config_file(
                job.base_disturbance_path
            )
        elif job.phase == "ablation_zero":
            base_dict = json.loads(Path(job.base_disturbance_path).read_text())
            base_disturbance = DisturbanceConfig()
        elif job.phase == "ablation":
            base_dict = json.loads(Path(job.base_disturbance_path).read_text())
            assert job.ablation_var is not None and job.ablation_value is not None
            base_disturbance = _build_ablation_disturbance(
                base_dict, job.ablation_var, float(job.ablation_value),
                job.base_seed,
            )
        else:
            raise ValueError(f"unknown phase: {job.phase}")

        perf_model = PerfLLM()
        perf_model.configure(
            strategy_config=StrategyConfig.init_from_config_file(strategy_path),
            model_config=ModelConfig.init_from_config_file(model_path),
            system_config=SystemConfig.init_from_config_file(system_path),
            pp_scheduling_config=PipelineScheduleConfig.init_from_config_file(schedule_path),
            disturbance_config=base_disturbance,
        )

        rows: List[Dict[str, Any]] = []
        template = _row_template(job, strategy_name, perf_model, base_dict)

        if job.phase in ("nominal", "ablation_zero"):
            # All-zero primary knobs → no per-task randomness, so a single
            # deterministic episode is enough.
            perf_model.disturbance.seed = job.base_seed
            perf_model.run_estimate()
            cost = perf_model.analysis_cost().data
            row = dict(template)
            row.update({
                "episode_idx": 0,
                "disturbance_seed": job.base_seed,
                "mfu": float(cost["mfu"]),
                "pp_utilization": float(cost["pp_utilization"]),
                "iter_time_s": _parse_time_seconds(cost["duration_time_per_iter"]),
                "peak_mem_gb": _peak_mem_gb(perf_model),
            })
            rows.append(row)
        elif job.phase in ("baseline", "ablation"):
            for ep_data in _run_episodes(perf_model, job.episodes, job.base_seed):
                row = dict(template)
                row.update(ep_data)
                rows.append(row)
        else:
            raise ValueError(f"unknown phase: {job.phase}")

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(job.shard_path), exist_ok=True)
        # Atomic-ish write: temp file + rename, so a crash mid-write doesn't
        # leave a half-shard that --resume would pick up as "done".
        tmp_path = job.shard_path + ".tmp"
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, job.shard_path)

        return {
            "status": "ok",
            "model": job.model,
            "schedule": job.schedule,
            "phase": job.phase,
            "ablation_var": job.ablation_var,
            "ablation_value": job.ablation_value,
            "rows": len(rows),
            "elapsed_s": time.time() - t0,
            "shard_path": job.shard_path,
        }
    except (AssertionError, FileNotFoundError) as e:
        # configure() rejection or missing strategy/config file — log+skip.
        return {
            "status": "skipped",
            "model": job.model,
            "schedule": job.schedule,
            "phase": job.phase,
            "ablation_var": job.ablation_var,
            "ablation_value": job.ablation_value,
            "reason": f"{type(e).__name__}: {e}",
            "elapsed_s": time.time() - t0,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "model": job.model,
            "schedule": job.schedule,
            "phase": job.phase,
            "ablation_var": job.ablation_var,
            "ablation_value": job.ablation_value,
            "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_s": time.time() - t0,
        }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def _list_release_models() -> List[str]:
    return sorted(k for k in RELEASE_MODELS.keys() if k != "root")


def _list_release_schedules() -> List[str]:
    return sorted(k for k in RELEASE_PP_SCHEDULING.keys() if k != "root")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_combined(output_dir: str, name: str, paths: List[str]) -> None:
    dfs = [pd.read_parquet(p) for p in paths]
    if not dfs:
        return
    combined = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(output_dir, f"{name}.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"[concat] {out_path}  rows={len(combined):,}  shards={len(paths)}")


def _concat_shards(output_dir: str, jobs: List[Job],
                   skipped: List[Dict[str, Any]]) -> None:
    """Concatenate shards into per-phase / per-axis parquet files.

    The shared ``ablation_zero`` shard (one row per (model, schedule)) is
    replicated into every ``ablation_<axis>.parquet`` with
    ``ablation_var=<axis>``, so each per-axis file contains a complete curve
    over ``[0, max]`` without plot-time anchor synthesis.
    """
    nominal_paths: List[str] = []
    baseline_paths: List[str] = []
    zero_paths: List[str] = []
    axis_paths: Dict[str, List[str]] = {}

    for job in jobs:
        if not job.shard_exists():
            continue
        if job.phase == "nominal":
            nominal_paths.append(job.shard_path)
        elif job.phase == "baseline":
            baseline_paths.append(job.shard_path)
        elif job.phase == "ablation_zero":
            zero_paths.append(job.shard_path)
        elif job.phase == "ablation":
            assert job.ablation_var is not None
            axis_paths.setdefault(job.ablation_var, []).append(job.shard_path)

    if nominal_paths:
        _write_combined(output_dir, "nominal", sorted(nominal_paths))
    if baseline_paths:
        _write_combined(output_dir, "baseline_disturbed",
                        sorted(baseline_paths))

    if axis_paths:
        zero_df: Optional[pd.DataFrame] = None
        if zero_paths:
            zero_dfs = [pd.read_parquet(p) for p in sorted(zero_paths)]
            zero_df = pd.concat(zero_dfs, ignore_index=True)
        for axis, paths in sorted(axis_paths.items()):
            sweep_dfs = [pd.read_parquet(p) for p in sorted(paths)]
            sweep_df = pd.concat(sweep_dfs, ignore_index=True)
            if zero_df is not None:
                anchor = zero_df.copy()
                anchor["ablation_var"] = axis
                combined = pd.concat([anchor, sweep_df], ignore_index=True)
            else:
                combined = sweep_df
            combined = combined.sort_values(
                ["model", "pp_schedule", "ablation_value", "episode_idx"]
            ).reset_index(drop=True)
            out_path = os.path.join(output_dir, f"ablation_{axis}.parquet")
            combined.to_parquet(out_path, index=False)
            print(f"[concat] {out_path}  rows={len(combined):,}  "
                  f"sweep_shards={len(paths)}  "
                  f"zero_shards={len(zero_paths)}")

    if skipped:
        skipped_path = os.path.join(output_dir, "skipped.json")
        with open(skipped_path, "w") as f:
            json.dump(skipped, f, indent=2)
        print(f"[concat] {skipped_path}  entries={len(skipped)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--system", default="h100_nvlink",
                   help="System config name (without .json).")
    p.add_argument("--base-disturbance", default="both",
                   help="Disturbance config name used for the baseline phase "
                        "and as the auxiliary-field source for ablations.")
    p.add_argument("--models", default="all",
                   help="Comma-separated model names, or 'all'.")
    p.add_argument("--schedules", default="all",
                   help="Comma-separated pp_scheduling names, or 'all'.")
    p.add_argument("--phases", default="nominal,baseline,ablation",
                   help=f"Subset of {set(VALID_PHASES)}.")
    p.add_argument("--episodes", type=int, default=100,
                   help="Episodes per (model, schedule) baseline cell. "
                        "Nominal always runs 1 deterministic episode; "
                        "ablation episodes are controlled separately by "
                        "--ablation-episodes.")
    p.add_argument("--ablation-vars",
                   default=",".join(ABLATION_AXES_DEFAULT),
                   help="Comma-separated ablation axes (subset of "
                        f"{ABLATION_AXES_DEFAULT}).")
    p.add_argument("--ablation-points", type=int, default=10,
                   help="Number of points in the inclusive [0, max] linspace "
                        "per ablation axis. Point 0 is the shared "
                        "ablation_zero anchor; the remaining N-1 are real "
                        "per-axis sweep cells.")
    p.add_argument("--ablation-episodes", type=int, default=30,
                   help="Episodes per non-zero ablation cell. The shared "
                        "ablation_zero anchor always runs 1 deterministic "
                        "episode (zero disturbance has no randomness).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; per-episode disturbance seeds are derived "
                        "from a Generator seeded with this value.")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of worker processes.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write parquet shards + final outputs.")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run all jobs even if shards exist on disk.")
    p.add_argument("--keep-shards", action="store_true",
                   help="Keep per-cell shards after final concat (default: keep).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.models == "all":
        models = _list_release_models()
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.schedules == "all":
        schedules = _list_release_schedules()
    else:
        schedules = [s.strip() for s in args.schedules.split(",") if s.strip()]

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for ph in phases:
        if ph not in VALID_PHASES:
            sys.exit(f"unknown phase: {ph} (valid: {VALID_PHASES})")
    if not phases:
        sys.exit(f"--phases must contain at least one of {set(VALID_PHASES)}")

    ablation_vars: List[str] = []
    if "ablation" in phases:
        ablation_vars = [v.strip() for v in args.ablation_vars.split(",")
                         if v.strip()]
        for v in ablation_vars:
            if v not in ABLATION_AXES_DEFAULT:
                sys.exit(f"unknown ablation var: {v} "
                         f"(valid: {ABLATION_AXES_DEFAULT})")
        if args.ablation_points < 2:
            sys.exit("--ablation-points must be >= 2")
        if args.ablation_episodes < 1:
            sys.exit("--ablation-episodes must be >= 1")

    base_disturbance_path: Optional[str] = None
    base_dict: Dict[str, Any] = {}
    if "baseline" in phases or "ablation" in phases:
        base_disturbance_path = get_simu_disturbance_config(args.base_disturbance)
        # Sanity-check it parses now so we fail fast.
        DisturbanceConfig.init_from_config_file(base_disturbance_path)
        base_dict = json.loads(Path(base_disturbance_path).read_text())
        if "ablation" in phases:
            for required in ("seq_len_mean", "op_duration_std",
                             "stage_slowdown_prob", "op_slowdown_prob"):
                if required not in base_dict:
                    sys.exit(f"--base-disturbance {args.base_disturbance!r} "
                             f"is missing field {required!r} required for "
                             f"ablations.")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(output_dir, "_shards"), exist_ok=True)

    jobs = build_jobs(
        models=models, schedules=schedules, phases=phases,
        episodes=args.episodes, base_seed=args.seed, system=args.system,
        base_disturbance_path=base_disturbance_path,
        base_dict=base_dict, output_dir=output_dir,
        ablation_vars=ablation_vars,
        ablation_points=args.ablation_points,
        ablation_episodes=args.ablation_episodes,
    )

    if not args.no_resume:
        pending = [j for j in jobs if not j.shard_exists()]
        skipped_resume = len(jobs) - len(pending)
    else:
        pending = jobs
        skipped_resume = 0

    print(f"models={len(models)}  schedules={len(schedules)}  phases={phases}")
    print(f"jobs total={len(jobs)}  pending={len(pending)}  "
          f"resumed={skipped_resume}  workers={args.workers}")
    print(f"output_dir={output_dir}")
    if base_disturbance_path:
        print(f"base_disturbance={base_disturbance_path}")
    if "ablation" in phases:
        axis_maxes = _ablation_axis_maxes(base_dict)
        print(f"ablation_vars={ablation_vars}  "
              f"points={args.ablation_points}  "
              f"episodes_per_cell={args.ablation_episodes}")
        for axis in ablation_vars:
            print(f"  {axis}: 0..{axis_maxes[axis]}")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "system": args.system,
        "base_disturbance_path": base_disturbance_path,
        "models": models,
        "schedules": schedules,
        "phases": phases,
        "episodes": args.episodes,
        "seed": args.seed,
        "ablation_vars": ablation_vars,
        "ablation_points": (args.ablation_points
                            if "ablation" in phases else None),
        "ablation_episodes": (args.ablation_episodes
                              if "ablation" in phases else None),
        "ablation_axis_maxes": (_ablation_axis_maxes(base_dict)
                                if "ablation" in phases else None),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    completed = 0
    t_start = time.time()

    if pending:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=ctx) as pool:
            future_to_job = {pool.submit(execute_job, j): j for j in pending}
            for fut in as_completed(future_to_job):
                result = fut.result()
                completed += 1
                tag = (f"{result.get('model')}/{result.get('schedule')}"
                       f"/{result.get('phase')}")
                if result.get("ablation_var") is not None:
                    tag += (f"/{result['ablation_var']}="
                            f"{result['ablation_value']:.4g}")
                elapsed = result.get("elapsed_s", 0.0)
                if result["status"] == "ok":
                    print(f"[{completed}/{len(pending)}] OK    {tag}  "
                          f"rows={result.get('rows')}  {elapsed:.1f}s",
                          flush=True)
                elif result["status"] == "skipped":
                    skipped.append(result)
                    print(f"[{completed}/{len(pending)}] SKIP  {tag}  "
                          f"({result['reason']})", flush=True)
                else:
                    failed.append(result)
                    print(f"[{completed}/{len(pending)}] FAIL  {tag}  "
                          f"({result['reason']})", flush=True)

    print(f"\nelapsed: {(time.time() - t_start)/60:.1f} min  "
          f"ok={completed - len(skipped) - len(failed)}  "
          f"skipped={len(skipped)}  failed={len(failed)}")

    if failed:
        failed_path = os.path.join(output_dir, "failed.json")
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"failures written to {failed_path}")

    _concat_shards(output_dir, jobs, skipped)


if __name__ == "__main__":
    main()
