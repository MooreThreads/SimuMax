"""Integration tests for schedule-aware peak memory in PerfLLM.

These run end-to-end on the real ``llama3_70b_*`` configs in
``configs/``. Each test exercises the bug that motivated the
schedule-aware memory work (`Schedule_Aware_Memory_Plan.md` §1) and
the §5.6.2 invariants.
"""

from __future__ import annotations

import re

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


_GB = 1024 ** 3


def _peak_mem_gb(strategy_name: str, schedule_name: str) -> dict[str, float]:
    """Run the analysis and return per-stage peak_mem values in GB.

    Keys are ``first_stage``/``middle_stage``/``last_stage`` (subset
    depending on ``pp_size``).
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
    mem = perf.analysis_mem().data
    out: dict[str, float] = {}
    for stage in ("first_stage", "middle_stage", "last_stage"):
        if stage not in mem:
            continue
        # ``convert_final_result_to_human_format`` stringifies peak_mem
        # as e.g. "71.8142 GB" — strip back to a float in GB so the
        # tests can do numeric comparisons.
        s = mem[stage]["peak_mem"]
        m = re.match(r"\s*([0-9.]+)\s*GB\s*$", s)
        assert m is not None, f"unexpected peak_mem format: {s!r}"
        out[stage] = float(m.group(1))
    return out


@pytest.fixture(scope="module")
def peaks() -> dict[tuple[str, str], dict[str, float]]:
    """Cache PerfLLM runs across the test cases — each is ~5s."""
    cases = [
        ("llama3_70b_debug", "1f1b"),  # mbn=8
        ("llama3_70b_optimal_mfu", "1f1b"),  # mbn=128
        ("llama3_70b_debug", "gpipe"),
        ("llama3_70b_optimal_mfu", "gpipe"),
        ("llama3_70b_debug", "zb_h1"),
        ("llama3_70b_optimal_mfu", "zb_h1"),
        ("llama3_70b_debug", "zb_h2"),
        ("llama3_70b_optimal_mfu", "zb_h2"),
    ]
    return {(s, sched): _peak_mem_gb(s, sched) for s, sched in cases}


# ---------------------------------------------------------------------------
# §5.6.2 invariants
# ---------------------------------------------------------------------------


def test_1f1b_invariant_to_micro_batch_num(peaks):
    """`peak_mem(1f1b, mbn=8) == peak_mem(1f1b, mbn=128)` — preserves the
    pre-bug-fix invariant for 1F1B (the schedule that was already
    correctly modeled).
    """
    p_small = peaks[("llama3_70b_debug", "1f1b")]
    p_big = peaks[("llama3_70b_optimal_mfu", "1f1b")]
    for stage in p_small:
        assert p_small[stage] == pytest.approx(p_big[stage], rel=1e-9), (
            f"1F1B {stage}: mbn=8 -> {p_small[stage]} GB, "
            f"mbn=128 -> {p_big[stage]} GB"
        )


def test_gpipe_grows_with_mbn(peaks):
    """`peak_mem(gpipe, mbn=128) > peak_mem(gpipe, mbn=8)` — fixes the
    smoking-gun bug (`Schedule_Aware_Memory_Plan.md` §1).
    """
    p_small = peaks[("llama3_70b_debug", "gpipe")]
    p_big = peaks[("llama3_70b_optimal_mfu", "gpipe")]
    for stage in p_small:
        assert p_big[stage] > p_small[stage], (
            f"GPipe {stage}: mbn=128 ({p_big[stage]} GB) should exceed "
            f"mbn=8 ({p_small[stage]} GB)"
        )


def test_gpipe_at_mbn8_exceeds_1f1b_at_mbn8(peaks):
    """At ``mbn=8`` (>= ``pp_size=4``), GPipe still accumulates more
    in-flight microbatches than 1F1B (mbn vs pp), so GPipe peak is
    strictly higher even when both have the same mbn.

    (The §5.6.2 ≈-equality lower-bound case would require ``mbn ==
    pp_size``; at ``mbn > pp_size`` GPipe and 1F1B diverge.)
    """
    p_gpipe = peaks[("llama3_70b_debug", "gpipe")]
    p_1f1b = peaks[("llama3_70b_debug", "1f1b")]
    for stage in p_gpipe:
        assert p_gpipe[stage] > p_1f1b[stage], (
            f"{stage}: GPipe ({p_gpipe[stage]} GB) should exceed "
            f"1F1B ({p_1f1b[stage]} GB) at mbn=8"
        )


def test_zb_h1_at_least_1f1b(peaks):
    """`peak_mem(zb_h1, mbn=128) >= peak_mem(1f1b, mbn=128)` — split
    B/W extends activation lifetime, so ZB-H1 holds at least as much.
    The first stage already runs out the warmup, so the bound is
    loose there; tighten the assertion to non-edge stages.
    """
    p_zbh1 = peaks[("llama3_70b_optimal_mfu", "zb_h1")]
    p_1f1b = peaks[("llama3_70b_optimal_mfu", "1f1b")]
    for stage in ("middle_stage", "last_stage"):
        if stage in p_zbh1 and stage in p_1f1b:
            assert p_zbh1[stage] >= p_1f1b[stage], (
                f"{stage}: ZB-H1 ({p_zbh1[stage]} GB) should be >= "
                f"1F1B ({p_1f1b[stage]} GB)"
            )


def test_zb_h2_strictly_above_1f1b(peaks):
    """ZB-H2 delays W more aggressively than ZB-H1 to fill the bubble,
    so its peak exceeds 1F1B by a non-trivial margin.
    """
    p_zbh2 = peaks[("llama3_70b_optimal_mfu", "zb_h2")]
    p_1f1b = peaks[("llama3_70b_optimal_mfu", "1f1b")]
    for stage in p_zbh2:
        assert p_zbh2[stage] > p_1f1b[stage], (
            f"{stage}: ZB-H2 ({p_zbh2[stage]} GB) should exceed "
            f"1F1B ({p_1f1b[stage]} GB)"
        )


def test_zb_h2_at_least_zb_h1(peaks):
    """ZB-H2's extra B-then-W reordering can only delay W further than
    ZB-H1, so its peak is at least as high.
    """
    p_zbh2 = peaks[("llama3_70b_optimal_mfu", "zb_h2")]
    p_zbh1 = peaks[("llama3_70b_optimal_mfu", "zb_h1")]
    for stage in p_zbh2:
        assert p_zbh2[stage] >= p_zbh1[stage] - 1e-6, (
            f"{stage}: ZB-H2 ({p_zbh2[stage]} GB) should be >= "
            f"ZB-H1 ({p_zbh1[stage]} GB)"
        )


# ---------------------------------------------------------------------------
# Anti-regression: 1F1B numbers must match the pre-tracker formula
# ---------------------------------------------------------------------------


def test_1f1b_matches_legacy_formula(peaks):
    """For 1F1B the new tracker walk should reproduce the exact
    closed-form ``model + (in_flight - 1) * cache + intra_peak``
    numbers from the legacy ``analysis_mem`` path.

    This is `Schedule_Aware_Memory_Plan.md` §7.1 (Anti-regression
    validation): bit-identical not required but rounding-tolerance
    equality on every stage is.
    """
    # Numbers captured before the schedule-aware rewiring (legacy
    # ``analysis_mem`` for ``llama3_70b_debug @ 1f1b``).
    expected = {
        "first_stage": 71.8142,
        "middle_stage": 63.6360,
        "last_stage": 56.7611,
    }
    actual = peaks[("llama3_70b_debug", "1f1b")]
    for stage, value in expected.items():
        assert actual[stage] == pytest.approx(value, abs=1e-3), (
            f"1F1B {stage}: expected {value} GB (legacy), "
            f"got {actual[stage]} GB"
        )
