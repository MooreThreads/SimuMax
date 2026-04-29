"""Variable-seq_len sanity for the schedule-aware peak memory.

`Schedule_Aware_Memory_Plan.md` §5.6.3 / §3.2: every non-trivial
activation term is linear in seq_len, so the per-mb memory tracker
should yield deterministic peaks at ``seq_len_std == 0`` and
seed-dependent peaks at ``seq_len_std > 0``.
"""

from __future__ import annotations

import numpy as np
import pytest

from simumax.core.config import (
    DisturbanceConfig,
    ModelConfig,
    PipelineScheduleConfig,
    StrategyConfig,
    SystemConfig,
)
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    get_simu_model_config,
    get_simu_pp_scheduling_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


_GB = 1024.0 ** 3


def _peak(seed: int, std: float, schedule: str = "1f1b") -> list[float]:
    """Per-rank peak in GB at given seed/std for llama3_70b_optimal_mfu."""
    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config("llama3_70b_optimal_mfu")
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config("llama3-70b")
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config("h100_nvlink")
        ),
        pp_scheduling_config=PipelineScheduleConfig.init_from_config_file(
            get_simu_pp_scheduling_config(schedule)
        ),
        disturbance_config=DisturbanceConfig(
            seed=seed,
            seq_len_mean=4096,
            seq_len_std=std,
            seq_len_min=256,
            seq_len_max=16384,
        ),
    )
    perf.run_estimate()
    perf._compute_pp_total_time(draw=False)
    peak = perf._walk_schedule_for_peak()
    return [p / _GB for p in peak]


def test_zero_std_is_deterministic_across_seeds():
    """std == 0 → peak does not depend on seed (constant seq_lens)."""
    p0 = _peak(seed=0, std=0.0)
    p1 = _peak(seed=42, std=0.0)
    p2 = _peak(seed=2025, std=0.0)
    for r in range(len(p0)):
        assert p0[r] == pytest.approx(p1[r], rel=1e-9)
        assert p0[r] == pytest.approx(p2[r], rel=1e-9)


def test_nonzero_std_varies_across_seeds():
    """std > 0 → different seeds produce different peaks."""
    p0 = _peak(seed=0, std=2048.0)
    p1 = _peak(seed=42, std=2048.0)
    p2 = _peak(seed=2025, std=2048.0)
    # At least one rank should differ across seeds (with high probability
    # — using fixed seeds avoids any chance of flakiness on this set).
    differs = any(
        abs(p0[r] - p1[r]) > 1e-6 or abs(p0[r] - p2[r]) > 1e-6
        for r in range(len(p0))
    )
    assert differs, (
        f"variable seq_lens did not change peak: p0={p0} p1={p1} p2={p2}"
    )


def test_variable_seq_below_constant_max_seq_bound():
    """The variable-seq peak (driven by per-mb seq_lens) should be at most
    the peak you would get if every microbatch had ``max(seq_lens)`` ——
    a sanity check on the linear scaling.

    With per-mb scaling: live = sum(act * s_mb / nominal). The bound
    holds because max(seq_lens) >= each s_mb so a constant-max walk
    upper-bounds any per-mb-varying walk that uses the *same* ordering
    of mbs.
    """
    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config("llama3_70b_optimal_mfu")
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config("llama3-70b")
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config("h100_nvlink")
        ),
        pp_scheduling_config=PipelineScheduleConfig.init_from_config_file(
            get_simu_pp_scheduling_config("1f1b")
        ),
        disturbance_config=DisturbanceConfig(
            seed=42,
            seq_len_mean=4096,
            seq_len_std=2048.0,
            seq_len_min=256,
            seq_len_max=16384,
        ),
    )
    perf.run_estimate()
    perf._compute_pp_total_time(draw=False)
    actual_peak = perf._walk_schedule_for_peak()

    # Substitute max(seq_lens) for every mb and rerun the walker.
    perf.seq_lens = np.full_like(perf.seq_lens, perf.seq_lens.max())
    bound_peak = perf._walk_schedule_for_peak()

    for r, (a, b) in enumerate(zip(actual_peak, bound_peak)):
        assert a <= b * (1.0 + 1e-9), (
            f"rank {r}: actual peak {a/_GB:.4f} GB exceeds the "
            f"max-seq_len bound {b/_GB:.4f} GB"
        )
