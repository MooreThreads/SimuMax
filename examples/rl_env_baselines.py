"""Closed-form baseline makespans for the RL env regression suite.

Runs SimuMax's built-in physical-rank schedulers on the same config the
RL env trains against. Trained agent is expected to reach (or beat)
these makespans on the fixed-seq / no-disturbance setting, and to beat
them under Phase 2 stochasticity.

Under disturbance (``seq_len_std > 0`` or any ``*_slowdown_prob > 0``
in the strategy config), each call to ``_compute_pp_total_time``
samples a fresh disturbance draw. Passing ``--n-episodes N`` replays
each schedule against N independent draws and reports mean / std /
min / max — that's the distribution the trained policy should beat.
When all stochasticity is disabled, all N draws coincide and std == 0.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import replace

import numpy as np

from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)

_SCHEDULES = ("1f1b", "gpipe", "zb_h1", "zb_h2")


def _load_strategy(name: str) -> StrategyConfig:
    return StrategyConfig.init_from_config_file(get_simu_strategy_config(name))


def _resample_episode(perf: PerfLLM, seed: int) -> None:
    """Re-seed and re-sample all four stochastic inputs for one episode.

    Same offsets as ``simumax.rl_env.backend.SimuMaxBackend.sample_episode``
    so baseline draws line up exactly with RL-env draws under the same
    base seed.
    """
    strategy = perf.strategy
    strategy.seq_len_seed = seed
    strategy.op_duration_seed = seed + 1
    strategy.op_slowdown_seed = seed + 2
    strategy.stage_slowdown_seed = seed + 3
    perf.seq_lens = perf._sample_seq_lens()
    perf._sample_op_disturbance()
    perf._sample_stage_disturbance()


def run_baseline(
    strategy_name: str,
    model_name: str,
    system_name: str,
    n_episodes: int = 1,
    base_seed: int = 0,
) -> dict[str, list[float]]:
    """Return ``{schedule: [iter_time_seconds, ...]}`` with ``n_episodes`` entries each."""
    base_strategy = _load_strategy(strategy_name)
    model_cfg = ModelConfig.init_from_config_file(get_simu_model_config(model_name))
    system_cfg = SystemConfig.init_from_config_file(get_simu_system_config(system_name))

    # Draw ep_seeds from the same BitGenerator chain gymnasium's
    # Env.reset(seed=base_seed) uses (SeedSequence -> PCG64 -> Generator).
    # That way PipelineSchedulingEnv constructed with base_seed and this
    # script constructed with the same base_seed see byte-identical draws,
    # episode-by-episode.
    ep_rng = np.random.default_rng(base_seed)
    ep_seeds = [int(ep_rng.integers(0, 2**31 - 1)) for _ in range(n_episodes)]

    out: dict[str, list[float]] = {}
    for schedule in _SCHEDULES:
        strategy = replace(base_strategy, pp_schedule=schedule)
        perf = PerfLLM()
        perf.configure(
            strategy_config=strategy, model_config=model_cfg, system_config=system_cfg
        )
        perf.run_estimate()

        makespans: list[float] = []
        for ep_seed in ep_seeds:
            _resample_episode(perf, seed=ep_seed)
            makespans.append(float(perf._compute_pp_total_time(draw=False)))
        out[schedule] = makespans
    return out


def _summarize(makespans: list[float]) -> dict[str, float]:
    n = len(makespans)
    mean = statistics.fmean(makespans)
    std = statistics.stdev(makespans) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(makespans),
        "max": max(makespans),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="tp1_pp2_dp4_mbs1")
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--system", default="a100_pcie")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of disturbance draws per schedule (default 1 = single deterministic run)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; episode k uses seed+k (aligned with SimuMaxBackend)",
    )
    args = parser.parse_args()

    results = run_baseline(
        args.strategy, args.model, args.system, args.n_episodes, args.seed
    )

    summaries = {s: _summarize(v) for s, v in results.items()}
    ordered = sorted(summaries.items(), key=lambda kv: kv[1]["mean"])

    print(
        f"Baselines for {args.strategy} / {args.model} / {args.system} "
        f"over {args.n_episodes} episode(s) (base_seed={args.seed}):"
    )
    print(f"  {'schedule':<10} {'n':>4} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    for schedule, stats in ordered:
        print(
            f"  {schedule:<10} {stats['n']:>4d} "
            f"{stats['mean']:>12.3f} {stats['std']:>12.3f} "
            f"{stats['min']:>12.3f} {stats['max']:>12.3f}"
        )


if __name__ == "__main__":
    main()
