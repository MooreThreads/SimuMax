"""SimuMax-backed per-episode duration sampling for the RL env.

The backend owns a long-lived ``PerfLLM`` instance (expensive
``build()`` + ``_run()``) and exposes :meth:`sample_episode` which
re-samples the three stochastic inputs (variable seq_len, Feature A op
noise, Feature B stage slowdown, Feature C op bernoulli slowdown) and
returns a frozen per-task duration table.

The RL env always models tasks in F/B/W-split form; fused-backward
semantics (e.g. classical GPipe / 1F1B) are expressed at the agent
level via ``FUSED_BACKWARD``, not by switching the backend's timing
representation. Accordingly this backend unconditionally pulls split
per-rank timings from :meth:`PerfLLM._per_rank_fwd_b_w_times`,
regardless of the configured ``pp_scheduling.pp_schedule``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from simumax.core.config import (
    DisturbanceConfig,
    ModelConfig,
    PipelineScheduleConfig,
    StrategyConfig,
    SystemConfig,
)
from simumax.core.perf_llm import PerfLLM


@dataclass(frozen=True)
class EpisodeData:
    """Per-episode duration table consumed by ``PipelineSchedulingEnv``.

    ``f_times[rank][mb]`` / ``b_times`` / ``w_times`` are the composed
    durations (base timing × A × B × C multipliers). For Phase 1 we
    only support schedules where stage index == physical rank, so
    ``rank`` can be read as ``stage`` directly.
    """

    p: int
    s: int
    m: int
    f_times: tuple[tuple[float, ...], ...]
    b_times: tuple[tuple[float, ...], ...]
    w_times: tuple[tuple[float, ...], ...]
    seq_lens: np.ndarray


ConfigLike = Union[
    StrategyConfig,
    ModelConfig,
    SystemConfig,
    DisturbanceConfig,
    PipelineScheduleConfig,
    str,
]


class SimuMaxBackend:
    """Wraps ``PerfLLM`` and exposes a cheap per-episode sampling API."""

    def __init__(
        self,
        strategy_config: ConfigLike,
        model_config: ConfigLike,
        system_config: ConfigLike,
        pp_scheduling_config: Optional[ConfigLike] = None,
        disturbance_config: Optional[ConfigLike] = None,
    ) -> None:
        perf = PerfLLM()
        perf.configure(
            strategy_config=strategy_config,
            model_config=model_config,
            system_config=system_config,
            pp_scheduling_config=pp_scheduling_config,
            disturbance_config=disturbance_config,
        )
        # run_estimate does build() + sample_seq_lens + _run() + initial
        # disturbance samples. We pay this once; sample_episode() then only
        # re-runs the cheap samplers.
        perf.run_estimate()

        self._perf = perf

    @property
    def perf(self) -> PerfLLM:
        return self._perf

    @property
    def num_gpus(self) -> int:
        return self._perf.strategy.pp_size

    @property
    def num_stages(self) -> int:
        # Phase 1: s == p (no virtual staging).
        return self._perf.strategy.pp_size

    @property
    def num_microbatches(self) -> int:
        return self._perf.strategy.micro_batch_num

    def sample_episode(self, seed: Optional[int] = None) -> EpisodeData:
        """Re-sample stochastic inputs and return a frozen duration table.

        When all stochasticity is disabled (``seq_len_std == 0`` and all
        disturbance probabilities / stds == 0), the returned arrays are
        deterministic and bit-identical across calls regardless of
        ``seed``.
        """
        perf = self._perf
        disturbance = perf.disturbance
        if seed is not None:
            # Single base seed — independent substreams are derived internally
            # via np.random.SeedSequence(seed).spawn(4).
            disturbance.seed = int(seed)

        perf.seq_lens = perf._sample_seq_lens()
        perf._sample_op_disturbance()
        perf._sample_stage_disturbance()
        f_times, b_times, w_times = perf._per_rank_fwd_b_w_times(apply_disturbance=True)

        strategy = perf.strategy
        return EpisodeData(
            p=strategy.pp_size,
            s=strategy.pp_size,
            m=strategy.micro_batch_num,
            f_times=tuple(tuple(float(x) for x in row) for row in f_times),
            b_times=tuple(tuple(float(x) for x in row) for row in b_times),
            w_times=tuple(tuple(float(x) for x in row) for row in w_times),
            seq_lens=perf.seq_lens.copy(),
        )
