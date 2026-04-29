"""Unit tests for the schedule-agnostic ActivationTracker."""

from __future__ import annotations

import pytest

from simumax.core.memory_tracker import ActivationTracker


GB = 1024 ** 3


def _make_tracker(
    num_stages: int = 1,
    act: float = 1.0 * GB,
    intra: float = 0.5 * GB,
    model_mem: float = 2.0 * GB,
    nominal_seq_len: int = 4096,
) -> ActivationTracker:
    return ActivationTracker(
        num_stages=num_stages,
        act_per_mb_nominal=[act] * num_stages,
        peak_intra_mb_per_stage=[intra] * num_stages,
        model_mem_per_stage=[model_mem] * num_stages,
        nominal_seq_len=nominal_seq_len,
    )


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_constructor_validates_array_lengths():
    with pytest.raises(ValueError, match="act_per_mb_nominal has length"):
        ActivationTracker(
            num_stages=2,
            act_per_mb_nominal=[1.0],
            peak_intra_mb_per_stage=[0.0, 0.0],
            model_mem_per_stage=[0.0, 0.0],
            nominal_seq_len=4096,
        )


def test_constructor_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="num_stages must be positive"):
        ActivationTracker(
            num_stages=0,
            act_per_mb_nominal=[],
            peak_intra_mb_per_stage=[],
            model_mem_per_stage=[],
            nominal_seq_len=4096,
        )
    with pytest.raises(ValueError, match="nominal_seq_len must be positive"):
        ActivationTracker(
            num_stages=1,
            act_per_mb_nominal=[1.0],
            peak_intra_mb_per_stage=[0.0],
            model_mem_per_stage=[0.0],
            nominal_seq_len=0,
        )


def test_initial_peak_is_model_mem():
    """No F events fired yet — peak should equal the resident model_mem."""
    tracker = _make_tracker(num_stages=3, model_mem=2.0 * GB)
    assert tracker.peak() == (2.0 * GB, 2.0 * GB, 2.0 * GB)


# ---------------------------------------------------------------------------
# Synthetic op stream → known peak (§5.6.1 case 1)
# ---------------------------------------------------------------------------


def test_single_F_then_W_returns_to_baseline():
    """on_F then on_W returns live to zero (§5.6.1 case 2).

    Single mb: prior_live=0 at F-time, peak = model + intra
    (intra already includes the mb's cache being built up).
    """
    tracker = _make_tracker(act=1.0 * GB, intra=0.5 * GB, model_mem=2.0 * GB)
    tracker.on_F(stage=0, mb=0, seq_len=4096)
    assert tracker.peak() == (2.5 * GB,)
    assert tracker.live(0) == 1.0 * GB
    tracker.on_W(stage=0, mb=0)
    assert tracker.live(0) == 0.0
    # Peak does not regress after release.
    assert tracker.peak() == (2.5 * GB,)


def test_known_peak_under_growing_pipeline():
    """4 mb's all forwarded before any are released — GPipe-like pattern.

    At F of mb k, prior_live = k * act, so the worst-case peak is
    observed at F of the last mb: model + 3*act + intra.
    """
    tracker = _make_tracker(act=1.0 * GB, intra=0.5 * GB, model_mem=2.0 * GB)
    for mb in range(4):
        tracker.on_F(0, mb, seq_len=4096)
    # At F of mb=3: prior_live = 3 GB, peak = 2 + 3 + 0.5 = 5.5 GB.
    assert tracker.peak() == (5.5 * GB,)
    # After all forwards, live = 4 GB but peak doesn't regress.
    assert tracker.live(0) == 4.0 * GB
    for mb in range(4):
        tracker.on_W(0, mb)
    assert tracker.live(0) == 0.0
    assert tracker.peak() == (5.5 * GB,)


def test_steady_state_pattern_caps_at_one_in_flight():
    """Strict alternation of F/W: prior_live is always 0 at F-time."""
    tracker = _make_tracker(act=1.0 * GB, intra=0.5 * GB, model_mem=2.0 * GB)
    for mb in range(8):
        tracker.on_F(0, mb, seq_len=4096)
        tracker.on_W(0, mb)
    # Every F sees prior_live=0: peak = model + intra = 2.5 GB.
    assert tracker.peak() == (2.5 * GB,)


# ---------------------------------------------------------------------------
# Multiple stages tracked independently (§5.6.1 case 3)
# ---------------------------------------------------------------------------


def test_stages_are_independent():
    tracker = ActivationTracker(
        num_stages=3,
        act_per_mb_nominal=[1.0 * GB, 0.5 * GB, 0.25 * GB],
        peak_intra_mb_per_stage=[0.5 * GB, 0.25 * GB, 0.125 * GB],
        model_mem_per_stage=[2.0 * GB, 1.5 * GB, 1.0 * GB],
        nominal_seq_len=4096,
    )
    tracker.on_F(0, 0, seq_len=4096)
    tracker.on_F(0, 1, seq_len=4096)  # stage 0 has 2 live mbs
    tracker.on_F(1, 0, seq_len=4096)  # stage 1 has 1 live mb
    # stage 2 untouched
    peak = tracker.peak()
    # stage 0 peaks at F of mb=1: prior_live=1 GB, intra=0.5 -> 2 + 1 + 0.5 = 3.5
    # stage 1 peaks at F of mb=0: prior_live=0,    intra=0.25 -> 1.5 + 0.25 = 1.75
    # stage 2: 1 GB (model only, no F)
    assert peak == (3.5 * GB, 1.75 * GB, 1.0 * GB)


def test_W_on_one_stage_does_not_affect_another():
    tracker = _make_tracker(num_stages=2)
    tracker.on_F(0, 0, seq_len=4096)
    tracker.on_F(1, 0, seq_len=4096)
    tracker.on_W(0, 0)
    assert tracker.live(0) == 0.0
    assert tracker.live(1) == 1.0 * GB


# ---------------------------------------------------------------------------
# Per-mb seq_len scaling (§5.6.1 case 4)
# ---------------------------------------------------------------------------


def test_seq_len_scaling_is_linear():
    """Doubling seq_len doubles activation; halving halves it."""
    tracker = _make_tracker(act=1.0 * GB, intra=0.5 * GB, model_mem=0.0)
    tracker.on_F(0, 0, seq_len=8192)  # 2x nominal
    assert tracker.live(0) == 2.0 * GB
    # peak at F (prior_live=0) = 0 + 0 + 1 (intra * 2) = 1 GB
    assert tracker.peak() == (1.0 * GB,)
    tracker.on_W(0, 0)
    tracker.on_F(0, 1, seq_len=2048)  # 0.5x nominal
    assert tracker.live(0) == 0.5 * GB


def test_release_uses_F_time_quantity():
    """on_W releases exactly the activation stamped at on_F, even if
    seq_len varies between events on the same stage."""
    tracker = _make_tracker(act=1.0 * GB, intra=0.0, model_mem=0.0)
    tracker.on_F(0, 0, seq_len=4096)  # adds 1 GB
    tracker.on_F(0, 1, seq_len=8192)  # adds 2 GB -> live = 3 GB
    assert tracker.live(0) == 3.0 * GB
    tracker.on_W(0, 1)
    assert tracker.live(0) == 1.0 * GB
    tracker.on_W(0, 0)
    assert tracker.live(0) == 0.0


def test_per_mb_peaks_grow_linearly_with_seq_len():
    """Sweep seq_len; peak (over a single F at that seq_len) scales linearly
    with seq_len at an exact factor of (s / nominal).

    Single-mb peak = model + 0 (prior_live) + intra * (s/nominal).
    """
    nominal = 4096
    act_nom = 1.0 * GB
    intra_nom = 0.5 * GB
    for s in (1024, 4096, 8192, 16384):
        tracker = _make_tracker(
            act=act_nom, intra=intra_nom, model_mem=0.0, nominal_seq_len=nominal
        )
        tracker.on_F(0, 0, seq_len=s)
        factor = s / nominal
        expected = factor * intra_nom
        assert tracker.peak()[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Misuse detection
# ---------------------------------------------------------------------------


def test_double_F_for_same_pair_raises():
    tracker = _make_tracker()
    tracker.on_F(0, 0, seq_len=4096)
    with pytest.raises(ValueError, match="on_F called twice"):
        tracker.on_F(0, 0, seq_len=4096)


def test_W_without_F_raises():
    tracker = _make_tracker()
    with pytest.raises(KeyError, match="without a matching on_F"):
        tracker.on_W(0, 0)


def test_invalid_stage_raises():
    tracker = _make_tracker(num_stages=2)
    with pytest.raises((IndexError, ValueError)):
        tracker.on_F(5, 0, seq_len=4096)


def test_zero_seq_len_rejected():
    tracker = _make_tracker()
    with pytest.raises(ValueError, match="seq_len must be positive"):
        tracker.on_F(0, 0, seq_len=0)


# ---------------------------------------------------------------------------
# Per-rank aggregation (V-shaped schedule support)
# ---------------------------------------------------------------------------


def test_identity_stage_to_rank_collapses_to_per_stage():
    """When stage_to_rank is omitted (or identity), peak_per_rank()
    must equal peak() bit-for-bit. Anti-regression for non-V callers.
    """
    tracker = _make_tracker(num_stages=3, model_mem=1.0 * GB)
    tracker.on_F(0, 0, seq_len=4096)
    tracker.on_F(1, 0, seq_len=4096)
    tracker.on_F(2, 0, seq_len=4096)
    assert tracker.peak() == tracker.peak_per_rank()


def test_explicit_identity_mapping_matches_default():
    """Passing ``stage_to_rank=range(num_stages)`` is equivalent to
    the default — a useful invariant for callers that always want
    to be explicit.
    """
    args = dict(
        num_stages=2,
        act_per_mb_nominal=[1.0 * GB, 1.0 * GB],
        peak_intra_mb_per_stage=[0.5 * GB, 0.5 * GB],
        model_mem_per_stage=[2.0 * GB, 2.0 * GB],
        nominal_seq_len=4096,
    )
    a = ActivationTracker(**args)
    b = ActivationTracker(**args, stage_to_rank=[0, 1])
    for t in (a, b):
        t.on_F(0, 0, seq_len=4096)
        t.on_F(1, 0, seq_len=4096)
        t.on_F(0, 1, seq_len=4096)
    assert a.peak() == b.peak()
    assert a.peak_per_rank() == b.peak_per_rank()


def test_two_stages_on_one_rank_sum_live():
    """Stages 0 and 1 both on rank 0 → per-rank live is the sum.

    F on stage 0 (no prior live anywhere): peak_rank[0] = model_rank
    + 0 + intra(stage 0).
    F on stage 1 (with stage 0's mb already cached): peak_rank[0] =
    model_rank + act_stage0 + intra(stage 1).
    """
    tracker = ActivationTracker(
        num_stages=2,
        act_per_mb_nominal=[1.0 * GB, 0.5 * GB],
        peak_intra_mb_per_stage=[0.5 * GB, 0.25 * GB],
        # Per-stage model_mem; per-rank baseline is the sum (3 GB).
        model_mem_per_stage=[2.0 * GB, 1.0 * GB],
        nominal_seq_len=4096,
        stage_to_rank=[0, 0],
    )
    # Per-rank baseline: 2 + 1 = 3 GB before any F.
    assert tracker.peak_per_rank() == (3.0 * GB,)
    tracker.on_F(0, 0, seq_len=4096)
    # peak_rank = 3 + 0 + 0.5 = 3.5 GB.
    assert tracker.peak_per_rank() == (3.5 * GB,)
    tracker.on_F(1, 0, seq_len=4096)
    # prior live on rank 0 = 1 GB (stage 0's mb). intra(stage 1) = 0.25.
    # peak_rank = 3 + 1 + 0.25 = 4.25 GB.
    assert tracker.peak_per_rank() == (4.25 * GB,)


def test_per_rank_live_releases_on_W():
    """on_W reduces both per-stage and per-rank live counters."""
    tracker = ActivationTracker(
        num_stages=2,
        act_per_mb_nominal=[1.0 * GB, 1.0 * GB],
        peak_intra_mb_per_stage=[0.0, 0.0],
        model_mem_per_stage=[0.0, 0.0],
        nominal_seq_len=4096,
        stage_to_rank=[0, 0],
    )
    tracker.on_F(0, 0, seq_len=4096)
    tracker.on_F(1, 0, seq_len=4096)
    assert tracker.live(0) == 1.0 * GB
    assert tracker.live(1) == 1.0 * GB
    tracker.on_W(0, 0)
    assert tracker.live(0) == 0.0
    assert tracker.live(1) == 1.0 * GB
    tracker.on_W(1, 0)
    assert tracker.live(1) == 0.0


def test_mixed_layout_across_ranks():
    """Stages 0,1 → rank 0; stage 2 → rank 1. Per-rank peaks track
    independently with proper aggregation per rank.
    """
    tracker = ActivationTracker(
        num_stages=3,
        act_per_mb_nominal=[1.0 * GB, 1.0 * GB, 1.0 * GB],
        peak_intra_mb_per_stage=[0.5 * GB, 0.5 * GB, 0.5 * GB],
        model_mem_per_stage=[1.0 * GB, 1.0 * GB, 2.0 * GB],
        nominal_seq_len=4096,
        stage_to_rank=[0, 0, 1],
    )
    # Baseline: rank 0 = 2 GB (sum of stages 0,1); rank 1 = 2 GB.
    assert tracker.peak_per_rank() == (2.0 * GB, 2.0 * GB)
    # Drive 2 mbs through both stages on rank 0, then one through rank 1.
    tracker.on_F(0, 0, seq_len=4096)  # rank 0 prior live=0; peak=2+0+0.5=2.5
    tracker.on_F(1, 0, seq_len=4096)  # rank 0 prior live=1; peak=2+1+0.5=3.5
    tracker.on_F(0, 1, seq_len=4096)  # rank 0 prior live=2; peak=2+2+0.5=4.5
    tracker.on_F(2, 0, seq_len=4096)  # rank 1 prior live=0; peak=2+0+0.5=2.5
    assert tracker.peak_per_rank() == (4.5 * GB, 2.5 * GB)


def test_stage_to_rank_validation():
    """stage_to_rank length must match num_stages; ranks non-negative."""
    with pytest.raises(ValueError, match="stage_to_rank has length"):
        ActivationTracker(
            num_stages=2,
            act_per_mb_nominal=[1.0, 1.0],
            peak_intra_mb_per_stage=[0.0, 0.0],
            model_mem_per_stage=[0.0, 0.0],
            nominal_seq_len=4096,
            stage_to_rank=[0, 0, 1],
        )
    with pytest.raises(ValueError, match="non-negative"):
        ActivationTracker(
            num_stages=2,
            act_per_mb_nominal=[1.0, 1.0],
            peak_intra_mb_per_stage=[0.0, 0.0],
            model_mem_per_stage=[0.0, 0.0],
            nominal_seq_len=4096,
            stage_to_rank=[0, -1],
        )
