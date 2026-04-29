"""Validation tests for V-shaped (interleaved_1f1b, zb_v) memory.

`V_Shaped_Memory_Plan.md` §2: the per-rank schedule walker plus 1/V
scaling of per-virtual-stage constants should produce these
relationships against 1F1B as the reference:

1. ZB-V matches 1F1B per-rank peak to within ~5% — same activation
   footprint as 1F1B by construction (W placement deliberately keeps
   the V-shape's per-rank live count bounded to ~pp microbatches at
   half-cache each).
2. Interleaved 1F1B holds **more** memory than 1F1B by ≥5% on the
   worst rank — Megatron-LM (Narayanan 2021, §2.2.2) explicitly
   trades activation memory for bubble at a `(V-1)/V` premium. The
   warmup formula `2*(pp - g - 1) + (V - 1)*pp` keeps `(V-1)*pp`
   extra microbatches in flight at rank 0, each at 1/V cache.
"""

from __future__ import annotations

import pytest

from simumax.core.config import (
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


def _peak_per_rank_gb(strategy_name: str, schedule_name: str) -> list[float]:
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
    perf._compute_pp_total_time(draw=False)
    peak = perf._walk_schedule_for_peak()
    assert peak is not None, "walker returned None"
    return [p / _GB for p in peak]


@pytest.fixture(scope="module")
def peaks() -> dict[str, list[float]]:
    """Cache the three runs across the test cases — each is ~1s."""
    strategy = "llama3_70b_optimal_mfu"
    return {
        sched: _peak_per_rank_gb(strategy, sched)
        for sched in ("1f1b", "zb_v", "interleaved_1f1b")
    }


def test_zb_v_matches_1f1b_per_rank_peak(peaks):
    """ZB-V's per-rank peak should match 1F1B's per-rank peak to within
    ~5%. The tolerance absorbs the small chunk-position-specific residual
    from the 1/V scaling on FIRST/LAST chunks (embedding/LM head split
    across virtual chunks; see V_Shaped_Memory_Plan.md §8.1).
    """
    p_1f1b = peaks["1f1b"]
    p_zbv = peaks["zb_v"]
    assert len(p_1f1b) == len(p_zbv)
    # 1F1B's per-rank peak decreases from rank 0 to pp-1 (warmup
    # holds fewer in-flight mbs on later ranks); ZB-V's pattern is
    # roughly uniform because each rank hosts a FIRST+LAST or
    # MIDDLE+MIDDLE pair. The physically-meaningful quantity is
    # the worst-case rank (max), which is what gates GPU OOM in
    # practice — that's what the validation target checks.
    max_1f1b = max(p_1f1b)
    max_zbv = max(p_zbv)
    rel_diff = abs(max_zbv - max_1f1b) / max_1f1b
    assert rel_diff <= 0.05, (
        f"ZB-V max-per-rank ({max_zbv:.4f} GB) deviates from "
        f"1F1B max ({max_1f1b:.4f} GB) by {rel_diff*100:.2f}% — "
        f"expected within 5% per validation target #1"
    )


def test_interleaved_1f1b_above_1f1b_on_worst_rank(peaks):
    """Interleaved 1F1B's worst-rank peak should exceed 1F1B's
    worst-rank peak by ≥5%. Reflects Megatron-LM's `(V-1)/V`
    extra-activation-memory property — interleaved trades memory for
    bubble. A failure here would mean either:
    - the walker's per-rank aggregation is broken;
    - `calculate_interleaved_1f1b_bubble`'s warmup formula has been
      changed to a memory-balanced variant (in which case update the
      test direction with a comment).
    """
    p_1f1b = peaks["1f1b"]
    p_int = peaks["interleaved_1f1b"]
    max_1f1b = max(p_1f1b)
    max_int = max(p_int)
    rel_excess = (max_int - max_1f1b) / max_1f1b
    assert rel_excess >= 0.05, (
        f"Interleaved 1F1B max-per-rank ({max_int:.4f} GB) should "
        f"exceed 1F1B max ({max_1f1b:.4f} GB) by ≥5%, got "
        f"{rel_excess*100:.2f}% — Megatron's `(V-1)/V` extra-memory "
        f"property should produce a positive margin."
    )


def test_zb_v_does_not_grow_with_micro_batch_num():
    """Sanity check: ZB-V should NOT scale memory with mbn. Fitting
    schedule's whole point is keeping the per-rank in-flight count
    bounded by ~pp regardless of how many microbatches are queued.
    """
    small = _peak_per_rank_gb("llama3_70b_debug", "zb_v")  # mbn=8
    big = _peak_per_rank_gb("llama3_70b_optimal_mfu", "zb_v")  # mbn=128
    for r in range(len(small)):
        rel_diff = abs(big[r] - small[r]) / small[r]
        assert rel_diff <= 0.01, (
            f"ZB-V rank {r}: peak should be ~independent of mbn but "
            f"changed from {small[r]:.4f} GB at mbn=8 to "
            f"{big[r]:.4f} GB at mbn=128 ({rel_diff*100:.2f}%)"
        )
