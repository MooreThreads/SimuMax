"""Static-vs-env memory consistency.

For a fixed (strategy, schedule, seed) the env's per-episode peak —
driven by the live ``ActivationTracker`` over the actual op timeline
— must agree with ``PerfLLM.analysis_mem`` to within float tolerance.

This is `Schedule_Aware_Memory_Plan.md` §5.6.4 / §10 acceptance #3:
catches drift between the two memory drivers. If it ever fails, one
of them has diverged and needs reconciling before either can be
trusted.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from simumax.core.config import (
    ModelConfig,
    PipelineScheduleConfig,
    StrategyConfig,
    SystemConfig,
)
from simumax.core.perf_llm import PerfLLM
from simumax.rl.agents import make_agent
from simumax.rl.env.backend import SimuMaxBackend
from simumax.rl.env.env import PipelineSchedulingEnv, RLEnvConfig
from simumax.utils import (
    get_simu_model_config,
    get_simu_pp_scheduling_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


_GB = 1024.0 ** 3


def _peak_mem_per_rank_gb_static(strategy_name: str, schedule_name: str) -> list[float]:
    """Run static analysis, return per-rank peak in GB.

    Reads the same per-stage values the env reports under
    ``info["peak_mem_per_stage"]`` (in bytes). The static analyzer
    rolls these up into ``first/middle/last_stage`` buckets — we
    bypass that here by calling ``_walk_schedule_for_peak`` directly
    so the comparison stays apples-to-apples.
    """
    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config(strategy_name)
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config("llama3-70b")
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config("h100_nvlink")
        ),
        pp_scheduling_config=PipelineScheduleConfig.init_from_config_file(
            get_simu_pp_scheduling_config(schedule_name)
        ),
    )
    perf.run_estimate()
    # Force the schedule walk (also runs internally on first
    # ``analysis_mem`` call but doing it here keeps the call sites
    # symmetric).
    perf._compute_pp_total_time(draw=False)
    peak = perf._walk_schedule_for_peak()
    assert peak is not None, "static walker returned None"
    return [p / _GB for p in peak]


def _peak_mem_per_rank_gb_env(
    agent_name: str,
    strategy_name: str,
    schedule_name: str,
    seed: int,
) -> list[float]:
    """Run a static-schedule agent in the env, return per-rank peak in GB.

    The agent name (e.g. ``gpipe`` / ``one_f_one_b``) selects which
    deterministic schedule is played in the env, mirroring the static
    analyzer's ``--schedule`` choice.
    """
    env_config = RLEnvConfig(
        strategy_config=get_simu_strategy_config(strategy_name),
        model_config=get_simu_model_config("llama3-70b"),
        system_config=get_simu_system_config("h100_nvlink"),
        pp_scheduling_config=get_simu_pp_scheduling_config(schedule_name),
        seed=seed,
    )
    backend = SimuMaxBackend(
        strategy_config=env_config.strategy_config,
        model_config=env_config.model_config,
        system_config=env_config.system_config,
        pp_scheduling_config=env_config.pp_scheduling_config,
        disturbance_config=env_config.disturbance_config,
    )
    env = PipelineSchedulingEnv(env_config, backend=backend)
    agent = make_agent(agent_name, backend.num_gpus, backend.num_microbatches)

    obs, info = env.reset()
    agent.reset()
    terminated = truncated = False
    steps = 0
    max_steps = (backend.num_microbatches * backend.num_gpus * 3 + 1) * 20
    last_info = info
    while not (terminated or truncated):
        action = agent.act(obs, info["action_mask"])
        obs, _r, terminated, truncated, info = env.step(action)
        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                f"Agent {agent_name} did not terminate within {max_steps} steps"
            )
        last_info = info
    peak = last_info["peak_mem_per_stage"]
    return [p / _GB for p in peak]


# ---------------------------------------------------------------------------
# §5.6.4 tests: 3 schedules × debug strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schedule, agent",
    [
        ("1f1b", "1f1b"),
        ("gpipe", "gpipe"),
        ("zb_h1", "zb_h1"),
    ],
)
def test_env_peak_matches_static(schedule: str, agent: str):
    strategy = "llama3_70b_debug"
    static = _peak_mem_per_rank_gb_static(strategy, schedule)
    env = _peak_mem_per_rank_gb_env(agent, strategy, schedule, seed=42)
    assert len(static) == len(env), (
        f"static vs env per-rank length mismatch: {len(static)} vs {len(env)}"
    )
    for r, (s, e) in enumerate(zip(static, env)):
        # Tolerance: 1% relative. The two paths use the same
        # per-stage constants but differ on which physical schedule
        # they walk: the static analyzer plays the analytical scheduler
        # in ``calculate_*_bubble``, the env plays the agent's queue.
        # For the canonical schedules (1F1B, GPipe) these should
        # produce equivalent op orderings.
        assert s == pytest.approx(e, rel=1e-2), (
            f"{schedule} rank {r}: static {s:.4f} GB vs env {e:.4f} GB"
        )
