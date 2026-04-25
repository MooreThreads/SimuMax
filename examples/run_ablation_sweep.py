"""Sweep MFU and PP utilization across (model × pp_schedule) cells.

For each model in ``configs/models/`` paired with its ``<model>_optimal_mfu``
strategy, this script runs three phases under a single system config:

1. **nominal** — no disturbances, 1 deterministic episode.
2. **baseline_disturbed** — full base disturbance profile, ``--episodes`` runs
   varying only the disturbance seed (matches eval_agents.py / run_gantt_demo.py
   semantics).
3. **ablation** — one-axis-at-a-time sweep over four disturbance fields
   (``seq_len_std``, ``op_duration_std``, ``stage_slowdown_prob``,
   ``op_slowdown_prob``). All other disturbance fields are zeroed for the axis
   being swept; ``seq_len_mean``/``min``/``max`` and the ``*_k`` magnitudes are
   inherited from the base disturbance config.

Outputs per cell are written as per-(cell, phase[, axis]) parquet shards under
``<output_dir>/_shards/`` and concatenated into long-format parquet files at
the end. Re-running with ``--resume`` (default) skips shards that already
exist on disk, so killed runs can be picked up where they left off.

The ``seq_len_std`` ablation values are specified as multiples of
``seq_len_mean``; the absolute std passed to ``DisturbanceConfig`` is
``multiplier * seq_len_mean``.

Example
-------
    uv run python examples/run_ablation_sweep.py \\
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ----------------------------------------------------------------------------
# Ablation grid
# ----------------------------------------------------------------------------

# seq_len_std ablation values are *multipliers* of seq_len_mean; the absolute
# std passed to DisturbanceConfig is multiplier * seq_len_mean.
SEQ_LEN_STD_MULTIPLIERS: Tuple[float, ...] = tuple(
    round(0.2 * i, 4) for i in range(1, 11)
)
OP_DURATION_STD_VALUES: Tuple[float, ...] = tuple(
    round(0.01 * i, 4) for i in range(1, 11)
)
STAGE_SLOWDOWN_PROB_VALUES: Tuple[float, ...] = tuple(
    round(0.05 * i, 4) for i in range(1, 11)
)
OP_SLOWDOWN_PROB_VALUES: Tuple[float, ...] = tuple(
    round(0.01 * i, 4) for i in range(1, 11)
)

ABLATION_AXES: Tuple[str, ...] = (
    "seq_len_std",
    "op_duration_std",
    "stage_slowdown_prob",
    "op_slowdown_prob",
)


def _ablation_values(axis: str) -> Tuple[float, ...]:
    if axis == "seq_len_std":
        return SEQ_LEN_STD_MULTIPLIERS
    if axis == "op_duration_std":
        return OP_DURATION_STD_VALUES
    if axis == "stage_slowdown_prob":
        return STAGE_SLOWDOWN_PROB_VALUES
    if axis == "op_slowdown_prob":
        return OP_SLOWDOWN_PROB_VALUES
    raise ValueError(f"unknown ablation axis: {axis}")


# ----------------------------------------------------------------------------
# Job specification
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    """One unit of work = one parquet shard.

    A job runs `episodes` episodes for a fixed (model, schedule, phase[, axis]).
    For ablations the same job iterates all 10 grid points before being
    persisted, so a single configure() amortizes across them.
    """
    model: str
    schedule: str
    phase: str             # "nominal" | "baseline" | "ablation"
    axis: str              # "" or one of ABLATION_AXES
    episodes: int          # episodes per disturbance setting (1 for nominal)
    base_seed: int
    system: str
    base_disturbance_path: Optional[str]  # None for nominal
    shard_path: str

    def shard_exists(self) -> bool:
        return os.path.exists(self.shard_path)


def _strategy_name_for(model: str) -> str:
    return model.replace("-", "_").replace(".", "_") + "_optimal_mfu"


def _shard_filename(output_dir: str, job_kind: str, model: str, schedule: str,
                    axis: str = "") -> str:
    safe_model = model.replace("/", "_")
    if axis:
        return os.path.join(output_dir, "_shards",
                            f"{job_kind}__{axis}__{safe_model}__{schedule}.parquet")
    return os.path.join(output_dir, "_shards",
                        f"{job_kind}__{safe_model}__{schedule}.parquet")


def build_jobs(models: List[str], schedules: List[str], phases: List[str],
               episodes: int, base_seed: int, system: str,
               base_disturbance_path: str, output_dir: str) -> List[Job]:
    jobs: List[Job] = []
    for model in models:
        for schedule in schedules:
            if "nominal" in phases:
                jobs.append(Job(
                    model=model, schedule=schedule, phase="nominal", axis="",
                    episodes=1, base_seed=base_seed, system=system,
                    base_disturbance_path=None,
                    shard_path=_shard_filename(output_dir, "nominal",
                                               model, schedule),
                ))
            if "baseline" in phases:
                jobs.append(Job(
                    model=model, schedule=schedule, phase="baseline", axis="",
                    episodes=episodes, base_seed=base_seed, system=system,
                    base_disturbance_path=base_disturbance_path,
                    shard_path=_shard_filename(output_dir, "baseline",
                                               model, schedule),
                ))
            if "ablation" in phases:
                for axis in ABLATION_AXES:
                    jobs.append(Job(
                        model=model, schedule=schedule, phase="ablation",
                        axis=axis, episodes=episodes, base_seed=base_seed,
                        system=system,
                        base_disturbance_path=base_disturbance_path,
                        shard_path=_shard_filename(output_dir, "ablation",
                                                   model, schedule, axis),
                    ))
    return jobs


# ----------------------------------------------------------------------------
# Per-cell execution (runs inside worker process)
# ----------------------------------------------------------------------------

def _build_disturbance_for_axis(base_dict: Dict[str, Any], axis: str,
                                ablation_value: float) -> DisturbanceConfig:
    """Construct a DisturbanceConfig with all axes zeroed except ``axis``.

    Magnitude/auxiliary fields (``stage_slowdown_k``, ``op_slowdown_k``,
    ``op_slowdown_max_count``) are inherited from the base config so each axis
    is swept against the same magnitude as the baseline disturbed run.
    The seed is filled in per-episode by the caller.
    """
    fields = {
        "seed": 0,
        "seq_len_mean": base_dict.get("seq_len_mean"),
        "seq_len_std": 0.0,
        "seq_len_min": base_dict.get("seq_len_min", 1),
        "seq_len_max": base_dict.get("seq_len_max"),
        "op_duration_std": 0.0,
        "op_duration_min_factor": base_dict.get("op_duration_min_factor", 0.1),
        "op_duration_max_factor": base_dict.get("op_duration_max_factor", 10.0),
        "stage_slowdown_prob": 0.0,
        "stage_slowdown_k": base_dict.get("stage_slowdown_k", 1.0),
        "op_slowdown_prob": 0.0,
        "op_slowdown_k": base_dict.get("op_slowdown_k", 1.0),
        "op_slowdown_max_count": base_dict.get("op_slowdown_max_count"),
    }
    if axis == "seq_len_std":
        seq_len_mean = fields["seq_len_mean"]
        if seq_len_mean is None:
            raise ValueError(
                "seq_len_std ablation requires seq_len_mean in base disturbance"
            )
        fields["seq_len_std"] = float(ablation_value) * float(seq_len_mean)
    elif axis == "op_duration_std":
        fields["op_duration_std"] = float(ablation_value)
    elif axis == "stage_slowdown_prob":
        fields["stage_slowdown_prob"] = float(ablation_value)
    elif axis == "op_slowdown_prob":
        fields["op_slowdown_prob"] = float(ablation_value)
    else:
        raise ValueError(f"unknown ablation axis: {axis}")
    return DisturbanceConfig(**fields)


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
        "ablation_axis": job.axis,
        "pp_size": perf_model.strategy.pp_size,
        "tp_size": perf_model.strategy.tp_size,
        "world_size": perf_model.strategy.world_size,
        "micro_batch_num": perf_model.strategy.micro_batch_num,
        "seq_len_strategy": perf_model.strategy.seq_len,
        "base_seq_len_mean": base_dict.get("seq_len_mean"),
        "base_seq_len_min": base_dict.get("seq_len_min"),
        "base_seq_len_max": base_dict.get("seq_len_max"),
        "base_stage_slowdown_k": base_dict.get("stage_slowdown_k"),
        "base_op_slowdown_k": base_dict.get("op_slowdown_k"),
        "base_op_slowdown_max_count": base_dict.get("op_slowdown_max_count"),
    }


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
        })
    return out


def execute_job(job: Job) -> Dict[str, Any]:
    """Run one shard's worth of episodes; write the parquet shard."""
    t0 = time.time()
    try:
        strategy_name = _strategy_name_for(job.model)

        strategy_path = get_simu_strategy_config(strategy_name)
        schedule_path = get_simu_pp_scheduling_config(job.schedule)
        model_path = get_simu_model_config(job.model)
        system_path = get_simu_system_config(job.system)

        # Initial disturbance for configure(): nominal phase uses defaults; all
        # other phases load the base. We swap disturbance per ablation point
        # without re-calling configure().
        if job.phase == "nominal":
            base_disturbance = DisturbanceConfig()
            base_dict: Dict[str, Any] = {}
        else:
            base_dict = json.loads(Path(job.base_disturbance_path).read_text())
            base_disturbance = DisturbanceConfig.init_from_config_file(
                job.base_disturbance_path
            )

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

        if job.phase == "nominal":
            # Single deterministic episode with all-zero disturbance.
            perf_model.disturbance.seed = job.base_seed
            perf_model.run_estimate()
            cost = perf_model.analysis_cost().data
            row = dict(template)
            row.update({
                "ablation_value": float("nan"),
                "ablation_value_absolute": float("nan"),
                "episode_idx": 0,
                "disturbance_seed": job.base_seed,
                "mfu": float(cost["mfu"]),
                "pp_utilization": float(cost["pp_utilization"]),
                "iter_time_s": _parse_time_seconds(cost["duration_time_per_iter"]),
            })
            rows.append(row)
        elif job.phase == "baseline":
            for ep_data in _run_episodes(perf_model, job.episodes, job.base_seed):
                row = dict(template)
                row.update({
                    "ablation_value": float("nan"),
                    "ablation_value_absolute": float("nan"),
                    **ep_data,
                })
                rows.append(row)
        elif job.phase == "ablation":
            for value in _ablation_values(job.axis):
                ablation_dc = _build_disturbance_for_axis(
                    base_dict, job.axis, value
                )
                perf_model._set_disturbance_config(ablation_dc)
                # Absolute value passed to DisturbanceConfig (for seq_len_std,
                # this is value * seq_len_mean; for the other axes it equals
                # the multiplier-style value 1:1).
                if job.axis == "seq_len_std":
                    abs_value = ablation_dc.seq_len_std
                elif job.axis == "op_duration_std":
                    abs_value = ablation_dc.op_duration_std
                elif job.axis == "stage_slowdown_prob":
                    abs_value = ablation_dc.stage_slowdown_prob
                else:
                    abs_value = ablation_dc.op_slowdown_prob
                for ep_data in _run_episodes(perf_model, job.episodes,
                                             job.base_seed):
                    row = dict(template)
                    row.update({
                        "ablation_value": float(value),
                        "ablation_value_absolute": float(abs_value),
                        **ep_data,
                    })
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
            "axis": job.axis,
            "rows": len(rows),
            "elapsed_s": time.time() - t0,
            "shard_path": job.shard_path,
        }
    except AssertionError as e:
        # configure() / sanity-check rejection — incompatible cell, log+skip.
        return {
            "status": "skipped",
            "model": job.model,
            "schedule": job.schedule,
            "phase": job.phase,
            "axis": job.axis,
            "reason": f"sanity_check: {e}",
            "elapsed_s": time.time() - t0,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "model": job.model,
            "schedule": job.schedule,
            "phase": job.phase,
            "axis": job.axis,
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


def _concat_shards(output_dir: str, jobs: List[Job], skipped: List[Dict[str, Any]]) -> None:
    """Concatenate shards into per-phase / per-axis parquet files."""
    groups: Dict[str, List[str]] = {}
    for job in jobs:
        if not job.shard_exists():
            continue
        if job.phase == "nominal":
            key = "nominal"
        elif job.phase == "baseline":
            key = "baseline_disturbed"
        else:
            key = f"ablation_{job.axis}"
        groups.setdefault(key, []).append(job.shard_path)

    for key, paths in sorted(groups.items()):
        dfs = [pd.read_parquet(p) for p in sorted(paths)]
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(output_dir, f"{key}.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"[concat] {out_path}  rows={len(combined):,}  shards={len(paths)}")

    if skipped:
        skipped_path = os.path.join(output_dir, "skipped.json")
        with open(skipped_path, "w") as f:
            json.dump(skipped, f, indent=2)
        print(f"[concat] {skipped_path}  entries={len(skipped)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--system", default="h100_nvlink",
                   help="System config name (without .json).")
    p.add_argument("--base-disturbance",
                   default="both",
                   help="Disturbance config name used for the baseline phase "
                        "and as the source of seq_len_mean/min/max plus the "
                        "*_k magnitudes for ablations.")
    p.add_argument("--models", default="all",
                   help="Comma-separated model names, or 'all'.")
    p.add_argument("--schedules", default="all",
                   help="Comma-separated pp_scheduling names, or 'all'.")
    p.add_argument("--phases", default="nominal,baseline,ablation",
                   help="Subset of {nominal,baseline,ablation}.")
    p.add_argument("--episodes", type=int, default=100,
                   help="Episodes per disturbance setting "
                        "(baseline + each ablation point).")
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
        if ph not in {"nominal", "baseline", "ablation"}:
            sys.exit(f"unknown phase: {ph}")

    base_disturbance_path = get_simu_disturbance_config(args.base_disturbance)
    # Sanity-check it parses now so we fail fast.
    DisturbanceConfig.init_from_config_file(base_disturbance_path)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(output_dir, "_shards"), exist_ok=True)

    jobs = build_jobs(
        models=models, schedules=schedules, phases=phases,
        episodes=args.episodes, base_seed=args.seed, system=args.system,
        base_disturbance_path=base_disturbance_path, output_dir=output_dir,
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
    print(f"base_disturbance={base_disturbance_path}")

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
        "ablation_axes": list(ABLATION_AXES),
        "seq_len_std_multipliers": list(SEQ_LEN_STD_MULTIPLIERS),
        "op_duration_std_values": list(OP_DURATION_STD_VALUES),
        "stage_slowdown_prob_values": list(STAGE_SLOWDOWN_PROB_VALUES),
        "op_slowdown_prob_values": list(OP_SLOWDOWN_PROB_VALUES),
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
                       f"/{result.get('phase')}"
                       + (f"/{result.get('axis')}" if result.get('axis') else ""))
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
