"""Schedule-agnostic peak-activation memory tracker.

Drives a per-stage live-activation account from F (forward done) and W
(weight-grad / activation-release) events, so the same code can serve the
static analyzer (`PerfLLM.analysis_mem`) and the RL env (per-episode peak).

Universal release-on-W rule:

    An activation is released on W-done, not B-done.

For the linear `y = xW`, the chain rule gives `dL/dW = x^T . dL/dy`, so the
saved input `x` must stay live until W is computed. For schedules that fuse
B and W into a single op (1f1b/gpipe/interleaved_1f1b in the analytical
path, and any agent emitting `FUSED_BACKWARD` in the env), the W event
coincides with the B event — so the same `on_W` call covers both fused and
split-backward cases. For ZB schedules where W is deliberately delayed,
this rule correctly extends the activation lifetime, which is precisely
the memory-vs-bubble trade documented in the ZB paper.

Per-mb seq_len scaling is applied here: every non-trivial activation term
in SimuMax's leaf modules is linear in `seq_len` (verified for
`dense_module`, MoE, and Flash attention), so

    act(stage, mb) = act_per_mb_nominal[stage] * seq_lens[mb] / nominal_seq_len

is accurate to first order. The static memory walk runs once at the
nominal seq_len; per-mb scaling lives here, not in the leaf-module
memory model.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple


class ActivationTracker:
    """Per-stage live-activation tracker driven by F/W events.

    The tracker is a pure bookkeeper: callers supply the per-stage
    activation cache size, intra-mb peak, model memory baseline, and
    nominal seq_len once at construction, then drive the walk by calling
    ``on_F`` / ``on_W`` in event order.

    Parameters
    ----------
    num_stages
        Number of independent memory accounts. For physical-rank
        schedules (1f1b, gpipe, zb_h1, zb_h2) this is ``pp_size``. For
        virtual-stage schedules (interleaved_1f1b, zb_v) this is
        ``V * pp_size`` and ``stage_to_rank`` (below) maps each virtual
        stage onto the GPU it lives on.
    act_per_mb_nominal
        Per-stage activation cache size held by one in-flight microbatch
        at the nominal seq_len, in bytes.
    peak_intra_mb_per_stage
        Per-stage intra-microbatch peak (the worst-of fwd / bwd /
        recompute peaks for the chunk on this stage), in bytes. Added
        on top of the live cache at every F event; the peak GPU
        footprint at any moment is dominated by `model_mem + sum(live)
        + intra_peak`.
    model_mem_per_stage
        Per-stage model memory baseline (params + grads + sharded
        optim state), in bytes. Constant across the episode.
    nominal_seq_len
        The seq_len under which ``act_per_mb_nominal`` was computed.
        Per-mb seq_len scaling uses this as the reference.
    stage_to_rank
        Optional per-stage mapping to a physical-rank index. When
        supplied, the tracker also accumulates per-rank live activations
        (sum across stages on the same rank) and tracks ``peak_per_rank``
        — the relevant quantity for V-shaped schedules where multiple
        virtual chunks share a GPU. Defaults to identity (stage == rank);
        in that case ``peak_per_rank()`` returns the same tuple as
        ``peak()`` so existing callers behave unchanged.
    """

    def __init__(
        self,
        num_stages: int,
        act_per_mb_nominal: Sequence[float],
        peak_intra_mb_per_stage: Sequence[float],
        model_mem_per_stage: Sequence[float],
        nominal_seq_len: int,
        stage_to_rank: Optional[Sequence[int]] = None,
    ) -> None:
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        if nominal_seq_len <= 0:
            raise ValueError(
                f"nominal_seq_len must be positive, got {nominal_seq_len}"
            )
        for name, arr in (
            ("act_per_mb_nominal", act_per_mb_nominal),
            ("peak_intra_mb_per_stage", peak_intra_mb_per_stage),
            ("model_mem_per_stage", model_mem_per_stage),
        ):
            if len(arr) != num_stages:
                raise ValueError(
                    f"{name} has length {len(arr)}, expected {num_stages}"
                )

        self._num_stages = num_stages
        self._act_per_mb_nominal = tuple(float(x) for x in act_per_mb_nominal)
        self._peak_intra_mb_per_stage = tuple(
            float(x) for x in peak_intra_mb_per_stage
        )
        self._model_mem_per_stage = tuple(float(x) for x in model_mem_per_stage)
        self._nominal_seq_len = int(nominal_seq_len)

        if stage_to_rank is None:
            stage_to_rank = tuple(range(num_stages))
        else:
            if len(stage_to_rank) != num_stages:
                raise ValueError(
                    f"stage_to_rank has length {len(stage_to_rank)}, "
                    f"expected {num_stages}"
                )
            stage_to_rank = tuple(int(r) for r in stage_to_rank)
            if any(r < 0 for r in stage_to_rank):
                raise ValueError(
                    f"stage_to_rank must contain non-negative ranks, "
                    f"got {stage_to_rank}"
                )
        self._stage_to_rank = stage_to_rank
        self._num_ranks = max(stage_to_rank) + 1

        # Per-rank model memory is the sum of model_mem for all stages
        # assigned to the rank — multiple virtual chunks share one GPU.
        rank_model_mem = [0.0] * self._num_ranks
        for s, r in enumerate(stage_to_rank):
            rank_model_mem[r] += self._model_mem_per_stage[s]
        self._model_mem_per_rank = tuple(rank_model_mem)

        self._live = [0.0] * num_stages
        self._live_per_rank = [0.0] * self._num_ranks
        # The peak baseline at t=0 is `model_mem` alone (no in-flight
        # activations). on_F will only raise this; if the schedule is
        # a no-op the reported peak still accounts for the resident model.
        self._peak = list(self._model_mem_per_stage)
        self._peak_per_rank = list(self._model_mem_per_rank)
        # Stamp the activation size at F-time so on_W releases the same
        # quantity even if anything ever rescales mid-episode.
        self._mb_act: Dict[Tuple[int, int], float] = {}

    @property
    def num_stages(self) -> int:
        return self._num_stages

    @property
    def nominal_seq_len(self) -> int:
        return self._nominal_seq_len

    def live(self, stage: int) -> float:
        return self._live[stage]

    def _check_stage(self, stage: int) -> None:
        if not 0 <= stage < self._num_stages:
            raise IndexError(
                f"stage index {stage} out of range [0, {self._num_stages})"
            )

    def peak(self) -> Tuple[float, ...]:
        """Per-stage peak memory observed so far, in bytes."""
        return tuple(self._peak)

    def peak_per_rank(self) -> Tuple[float, ...]:
        """Per-rank peak GPU memory observed so far, in bytes.

        For V-shaped schedules (multiple virtual chunks per GPU) this
        is the physically-meaningful number: at every event we sum
        live activations across all stages on the rank and add the
        single active op's intra-mb peak. When ``stage_to_rank`` was
        not supplied at construction, the identity mapping makes the
        per-rank account collapse to the per-stage account, so this
        returns the same tuple as :meth:`peak`.
        """
        return tuple(self._peak_per_rank)

    def on_F(self, stage: int, mb: int, seq_len: int) -> None:
        """Record a forward op for ``(stage, mb)``.

        Memory model:

            peak_during_F = model_mem + prior_live + intra_mb_peak

        where ``prior_live`` is the activation cache from microbatches
        that have already completed F but not yet been released by W —
        i.e. the "other" in-flight microbatches, not counting the one
        currently being computed. ``intra_mb_peak`` (max of fwd / bwd /
        recomp peaks from the chunk's ``PeakPoint``) already includes
        the new microbatch's cache being built up during the forward op
        plus any transient per-op buffers, so adding it on top of
        ``prior_live`` is the correct envelope.

        After the peak update, the new microbatch's cache becomes part
        of ``_live`` for the benefit of subsequent F events. The matching
        ``on_W`` releases exactly the stamped quantity, even if seq_len
        scaling has changed in the meantime.

        ``mb`` is used as a unique key for the (stage, mb) pair. For
        virtual-stage schedules where the same rank hosts multiple
        chunks of the same microbatch, encode the virtual index into
        ``stage`` or pass a unique ``mb`` per (chunk, microbatch) pair.
        """
        self._check_stage(stage)
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        key = (stage, mb)
        if key in self._mb_act:
            raise ValueError(
                f"on_F called twice for (stage={stage}, mb={mb}); "
                f"missing on_W release between F events?"
            )
        s_factor = seq_len / self._nominal_seq_len
        act = self._act_per_mb_nominal[stage] * s_factor
        intra = self._peak_intra_mb_per_stage[stage] * s_factor
        # Peak observed during this F op uses the "prior" live cache:
        # intra already encompasses the new microbatch's cache plus
        # transients, so adding the new act here would double-count.
        candidate = (
            self._model_mem_per_stage[stage] + self._live[stage] + intra
        )
        if candidate > self._peak[stage]:
            self._peak[stage] = candidate
        # Per-rank candidate: only the currently-running op contributes
        # an intra spike on its rank — adjacent virtual chunks on the
        # same GPU are idle while this stage runs F.
        rank = self._stage_to_rank[stage]
        rank_candidate = (
            self._model_mem_per_rank[rank]
            + self._live_per_rank[rank]
            + intra
        )
        if rank_candidate > self._peak_per_rank[rank]:
            self._peak_per_rank[rank] = rank_candidate
        # After F finishes, the new mb's cache joins the live pool —
        # both per-stage and per-rank.
        self._live[stage] += act
        self._live_per_rank[rank] += act
        self._mb_act[key] = act

    def on_W(self, stage: int, mb: int) -> None:
        """Record completion of the W (or fused B+W) op for ``(stage, mb)``.

        Releases the activation stamped at the matching ``on_F``. Raises
        ``KeyError`` if no F was recorded for this pair, which would
        indicate a schedule walker bug.
        """
        self._check_stage(stage)
        key = (stage, mb)
        if key not in self._mb_act:
            raise KeyError(
                f"on_W called for (stage={stage}, mb={mb}) without a "
                f"matching on_F"
            )
        act = self._mb_act.pop(key)
        self._live[stage] -= act
        rank = self._stage_to_rank[stage]
        self._live_per_rank[rank] -= act
        # Numerical guard: with float arithmetic, a long episode can
        # accumulate a tiny negative residual after the last release.
        # Snap to zero when within rounding of the original add.
        if -1e-6 * abs(act) <= self._live[stage] < 0.0:
            self._live[stage] = 0.0
        if -1e-6 * abs(act) <= self._live_per_rank[rank] < 0.0:
            self._live_per_rank[rank] = 0.0
