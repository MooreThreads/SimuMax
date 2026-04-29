"""Gymnasium environment for SimuMax-driven pipeline-parallel scheduling.

Direct port of ``rlpp/environment/core.py`` with the duration backend
swapped for :class:`~simumax.rl.env.backend.SimuMaxBackend`. Step
semantics, action masking, and the MaskablePPO contract are identical
to rlpp so we can reuse the existing training harness unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn, Optional, Union

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from simumax.core.config import DisturbanceConfig, ModelConfig, StrategyConfig, SystemConfig
from simumax.core.gantt import GanttBar, plot_gantt
from simumax.core.memory_tracker import ActivationTracker
from simumax.rl.env.backend import ConfigLike, EpisodeData, SimuMaxBackend
from simumax.rl.env.event_queue import CompletionEvent, EventQueue
from simumax.rl.env.stage_mapping import StageMapping
from simumax.rl.env.task_graph import TaskGraph
from simumax.rl.env.types import OpType, RewardMode, TaskState

_INVALID_ACTION_PENALTY = -1e6


@dataclass
class RLEnvConfig:
    """Environment-side knobs (orthogonal to SimuMax's JSON configs).

    All scheduling / stochasticity / hardware parameters flow through
    the three SimuMax configs; this dataclass only carries RL-layer
    settings.
    """

    strategy_config: ConfigLike
    model_config: ConfigLike
    system_config: ConfigLike
    pp_scheduling_config: Optional[ConfigLike] = None
    disturbance_config: Optional[ConfigLike] = None
    reward_mode: RewardMode = RewardMode.UTILIZATION
    max_time_limit: Optional[float] = None
    seed: Optional[int] = None
    normalise_seq_lens: bool = True


def _require_reset(name: str) -> NoReturn:
    msg = f"Environment must be reset before accessing {name}."
    raise RuntimeError(msg)


class PipelineSchedulingEnv(gymnasium.Env):
    """Gym env where the agent dispatches PP tasks on SimuMax durations."""

    metadata: dict[str, Any] = {
        "render_modes": ["committed", "projected"],
        "render_fps": 0,
    }

    def __init__(
        self,
        env_config: Optional[RLEnvConfig] = None,
        *,
        backend: Optional[SimuMaxBackend] = None,
    ) -> None:
        super().__init__()

        if env_config is None:
            raise ValueError("env_config is required")
        self._env_config = env_config
        self._reward_mode = RewardMode(env_config.reward_mode)

        self._backend = backend or SimuMaxBackend(
            strategy_config=env_config.strategy_config,
            model_config=env_config.model_config,
            system_config=env_config.system_config,
            pp_scheduling_config=env_config.pp_scheduling_config,
            disturbance_config=env_config.disturbance_config,
        )

        self._p = self._backend.num_gpus
        self._s = self._backend.num_stages
        self._m = self._backend.num_microbatches
        self._n_tasks = self._m * self._s * 3

        self._stage_mapping = StageMapping(self._p, self._s)

        seq_len_nominal = float(self._backend.perf.strategy.seq_len)
        seq_len_mean = self._backend.perf.disturbance.seq_len_mean
        self._seq_len_scale = (
            float(seq_len_mean) if seq_len_mean else seq_len_nominal
        ) or 1.0

        self.observation_space = spaces.Dict(
            {
                "task_state": spaces.MultiDiscrete(
                    np.full(self._n_tasks, 4, dtype=np.int64)
                ),
                "gpu_status": spaces.MultiBinary(self._p),
                "seq_lens": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self._m,),
                    dtype=np.float64,
                ),
            }
        )
        self.action_space = spaces.Discrete(self._n_tasks + 1)

        self._task_graph: Optional[TaskGraph] = None
        self._event_queue: Optional[EventQueue] = None
        self._gpu_status: Optional[NDArray[np.int8]] = None
        self._gpu_task_start: Optional[NDArray[np.float32]] = None
        self._current_time: float = 0.0
        self._action_mask: Optional[NDArray[np.int8]] = None
        self._execution_log: list[dict[str, Any]] = []
        self._episode: Optional[EpisodeData] = None
        # Per-episode peak-memory tracker. Built fresh on every reset;
        # ``on_F`` / ``on_W`` fired from ``_dispatch_task`` /
        # ``_advance_time``. Memory release follows the universal
        # release-on-W rule (see ``ActivationTracker`` docstring) — for
        # FUSED_BACKWARD agents the env still emits W as a separate
        # task, so the same hook covers split and fused-backward agents.
        self._mem_tracker: Optional[ActivationTracker] = None

    # ------------------------------------------------------------------

    @property
    def _tg(self) -> TaskGraph:
        if self._task_graph is None:
            _require_reset("_task_graph")
        return self._task_graph

    @property
    def _eq(self) -> EventQueue:
        if self._event_queue is None:
            _require_reset("_event_queue")
        return self._event_queue

    @property
    def _gs(self) -> NDArray[np.int8]:
        if self._gpu_status is None:
            _require_reset("_gpu_status")
        return self._gpu_status

    @property
    def _gts(self) -> NDArray[np.float32]:
        if self._gpu_task_start is None:
            _require_reset("_gpu_task_start")
        return self._gpu_task_start

    @property
    def _mask(self) -> NDArray[np.int8]:
        if self._action_mask is None:
            _require_reset("_action_mask")
        return self._action_mask

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        # Seed self.np_random deterministically on the first reset (from the
        # caller's seed, or the RLEnvConfig fallback). On subsequent resets
        # with seed=None, super() preserves self.np_random's state so it
        # advances between episodes — each rollout draws a fresh scalar seed
        # for the backend from the single RNG stream, giving us both natural
        # per-episode variability and reproducibility from the initial seed.
        if seed is None and self._np_random is None:
            seed = self._env_config.seed
        super().reset(seed=seed)

        self._task_graph = TaskGraph(self._m, self._s)
        self._event_queue = EventQueue()
        self._gpu_status = np.zeros(self._p, dtype=np.int8)
        self._gpu_task_start = np.zeros(self._p, dtype=np.float32)
        self._current_time = 0.0
        self._execution_log = []

        ep_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._episode = self._backend.sample_episode(seed=ep_seed)

        mem_const = self._backend.stage_mem
        self._mem_tracker = ActivationTracker(
            num_stages=self._s,
            act_per_mb_nominal=mem_const.act_per_mb_nominal,
            peak_intra_mb_per_stage=mem_const.peak_intra_mb_per_stage,
            model_mem_per_stage=mem_const.model_mem_per_stage,
            nominal_seq_len=mem_const.nominal_seq_len,
        )

        self._action_mask = self._compute_action_mask()
        obs = self._build_observation()
        info: dict[str, Any] = {"action_mask": self._action_mask.copy()}
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        task_graph = self._tg
        event_queue = self._eq
        mask = self._mask

        if not mask[action]:
            self._action_mask = self._compute_action_mask()
            obs = self._build_observation()
            info: dict[str, Any] = {"action_mask": self._mask.copy()}
            return obs, _INVALID_ACTION_PENALTY, False, False, info

        if action < self._n_tasks:
            self._dispatch_task(action)

        should_advance = action == self._n_tasks
        if not should_advance:
            should_advance = not self._any_schedulable()

        if should_advance and not event_queue.is_empty():
            self._advance_time()

        self._action_mask = self._compute_action_mask()

        terminated = task_graph.all_done()
        truncated = (
            self._env_config.max_time_limit is not None
            and self._current_time > self._env_config.max_time_limit
        )

        reward = self._compute_reward(terminated)
        obs = self._build_observation()
        info = {"action_mask": self._mask.copy()}
        if terminated or truncated:
            info["iter_time"] = float(self._current_time)
            info["pp_utilization"] = self._compute_pp_utilization()
            info["mfu"] = self._compute_mfu()
            self._populate_memory_info(info)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> NDArray[np.int8]:
        return self._mask.copy()

    # ------------------------------------------------------------------
    # Observation / mask helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict[str, NDArray[Any]]:
        task_graph = self._tg
        gpu_status = self._gs
        seq_lens = self._episode.seq_lens.astype(np.float64, copy=True)
        if self._env_config.normalise_seq_lens:
            seq_lens = seq_lens / self._seq_len_scale
        return {
            "task_state": task_graph.get_state_matrix().ravel(),
            "gpu_status": gpu_status.copy(),
            "seq_lens": seq_lens,
        }

    def _compute_action_mask(self) -> NDArray[np.int8]:
        task_graph = self._tg
        gpu_status = self._gs
        result = np.zeros(self._n_tasks + 1, dtype=np.int8)

        for t in range(self._n_tasks):
            if task_graph.get_state(t) != TaskState.READY:
                continue
            _, stage, _ = task_graph.index_to_task(t)
            gpu = self._stage_mapping.stage_to_gpu(stage)
            if gpu_status[gpu] == 0:
                result[t] = 1

        any_schedulable = bool(np.any(result[: self._n_tasks]))
        all_gpus_idle = bool(np.all(gpu_status == 0))
        # No-op disallowed iff all GPUs idle AND a task is schedulable.
        result[self._n_tasks] = int(not (any_schedulable and all_gpus_idle))
        return result

    def _any_schedulable(self) -> bool:
        task_graph = self._tg
        gpu_status = self._gs
        for t in range(self._n_tasks):
            if task_graph.get_state(t) != TaskState.READY:
                continue
            _, stage, _ = task_graph.index_to_task(t)
            gpu = self._stage_mapping.stage_to_gpu(stage)
            if gpu_status[gpu] == 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Dispatch / advance
    # ------------------------------------------------------------------

    def _task_duration(self, mb: int, stage: int, op: OpType) -> float:
        ep = self._episode
        if ep is None:
            _require_reset("_episode")
        # Phase 1: s == p, so stage index == physical rank.
        if op == OpType.F:
            return ep.f_times[stage][mb]
        if op == OpType.B:
            return ep.b_times[stage][mb]
        return ep.w_times[stage][mb]

    def _dispatch_task(self, task_index: int) -> None:
        task_graph = self._tg
        event_queue = self._eq
        gpu_status = self._gs
        gpu_task_start = self._gts

        microbatch, stage, op = task_graph.index_to_task(task_index)
        gpu = self._stage_mapping.stage_to_gpu(stage)
        duration = self._task_duration(microbatch, stage, op)

        task_graph.set_state(task_index, TaskState.RUNNING)
        gpu_status[gpu] = 1
        gpu_task_start[gpu] = self._current_time

        event_queue.push(
            CompletionEvent(
                time=self._current_time + duration,
                task_index=task_index,
                gpu=gpu,
            )
        )

        self._execution_log.append(
            {
                "microbatch": microbatch,
                "stage": stage,
                "op": op,
                "gpu": gpu,
                "start_time": self._current_time,
                "end_time": None,
                "task_index": task_index,
                "duration": duration,
            }
        )

    def _advance_time(self) -> None:
        task_graph = self._tg
        event_queue = self._eq
        gpu_status = self._gs
        gpu_task_start = self._gts

        events = event_queue.pop_earliest()
        if not events:
            return
        new_time = events[0].time

        for event in events:
            task_graph.set_state(event.task_index, TaskState.DONE)
            duration = event.time - gpu_task_start[event.gpu]
            task_graph.set_duration(event.task_index, float(duration))
            gpu_status[event.gpu] = 0
            for record in reversed(self._execution_log):
                if record["task_index"] == event.task_index:
                    record["end_time"] = event.time
                    break
            task_graph.update_blocked_to_ready(event.task_index)
            # Memory accounting: F-completion adds the activation to
            # the live cache; W-completion releases it. B-completion is
            # a no-op for memory because the saved input ``x`` is still
            # needed by the matching W (chain rule for ``y = xW``).
            mb, stage, op = task_graph.index_to_task(event.task_index)
            if op == OpType.F:
                seq_len = int(self._episode.seq_lens[mb])
                self._mem_tracker.on_F(stage, mb, seq_len)
            elif op == OpType.W:
                self._mem_tracker.on_W(stage, mb)

        self._current_time = new_time

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_gpu_makespans(self) -> NDArray[np.float64]:
        gpu_start = np.full(self._p, np.inf)
        gpu_end = np.full(self._p, -np.inf)
        for record in self._execution_log:
            g = record["gpu"]
            gpu_start[g] = min(gpu_start[g], record["start_time"])
            if record["end_time"] is not None:
                gpu_end[g] = max(gpu_end[g], record["end_time"])
        return np.where(gpu_end > -np.inf, gpu_end - gpu_start, 0.0)

    def _populate_memory_info(self, info: dict[str, Any]) -> None:
        """Surface the per-episode peak GPU memory in the env's ``info``.

        Mirrors how ``mfu`` / ``pp_utilization`` are exposed: scalar
        keys (``peak_mem_max_gb``, ``peak_mem_first_gb``, …) are picked
        up by the SB3 ``ep_info_buffer`` and logged to tensorboard via
        ``simumax/rl/train.py``'s ``_EPISODE_INFO_KEYS`` mechanism. The
        full per-stage tuple is stashed under a non-scalar key for
        programmatic consumers (eval scripts).

        Per-stage memory is reported in the same units as the static
        analyzer (bytes), then converted to GB for the scalar keys so
        the tensorboard plots are directly comparable to device
        capacity.
        """
        if self._mem_tracker is None:
            _require_reset("_mem_tracker")
        gb = 1024.0 ** 3
        peak_per_stage = self._mem_tracker.peak()
        info["peak_mem_per_stage"] = peak_per_stage
        info["peak_mem_max_gb"] = max(peak_per_stage) / gb
        if self._p >= 2:
            info["peak_mem_first_gb"] = peak_per_stage[0] / gb
            info["peak_mem_last_gb"] = peak_per_stage[self._p - 1] / gb
            if self._p > 2:
                # Pick rank 1 as the representative middle stage to match
                # ``PerfLLM.analysis_mem``'s ``middle_stage`` reporting.
                info["peak_mem_middle_gb"] = peak_per_stage[1] / gb

    def _compute_mfu(self) -> float:
        # Episode-level MFU: PP iter time from the agent's schedule combined
        # with the constant DP+optim overhead and the episode's sampled
        # seq_lens, mirroring PerfLLM.analysis_cost (perf_llm.py:2606-2620).
        if self._episode is None:
            _require_reset("_episode")
        return self._backend.compute_mfu(self._current_time, self._episode.seq_lens)

    def _compute_pp_utilization(self) -> float:
        # Matches PerfLLM.analysis_cost's pp_utilization (perf_llm.py): both
        # divide Σ_r Σ_m disturbance-applied (f + b + w) by pp × makespan, so
        # the metric is the true scheduling bubble fraction (idle gaps), not a
        # mix of bubble and disturbance slowdown.
        T = self._current_time
        if T <= 0.0:
            return 0.0
        useful_work = float(np.sum(self._tg.get_duration_matrix()))
        return useful_work / (self._p * T)

    def _compute_reward(self, done: bool) -> float:
        if not done:
            return 0.0

        pp = self._p
        T = self._current_time
        useful_work = float(np.sum(self._tg.get_duration_matrix()))

        if self._reward_mode == RewardMode.MAKESPAN:
            return -T

        if self._reward_mode == RewardMode.BUBBLE:
            return -(pp * T - useful_work)

        if self._reward_mode == RewardMode.MFU:
            return self._compute_mfu()

        return useful_work / (pp * T) if T > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        output_path: Optional[str] = None,
        *,
        mode: str = "projected",
        title: Optional[str] = None,
        display: bool = False,
    ) -> None:
        """Render a Gantt chart of the episode so far.

        At least one of ``output_path`` (save PNG) or ``display`` (open a
        blocking matplotlib window) must be set; both may be combined.

        ``mode="committed"`` draws only completed tasks. ``mode="projected"``
        (the default) also draws in-flight tasks at their scheduled
        completion time (hatched, dimmed) and a vertical ``now`` cursor at
        ``self._current_time``. Under Phase 1, stage index equals rank, so
        rows map directly to pipeline stages.
        """
        if mode not in ("committed", "projected"):
            raise ValueError(
                f"mode must be 'committed' or 'projected'; got {mode!r}"
            )
        if output_path is None and not display:
            raise ValueError("render() needs output_path, display=True, or both.")
        if self._task_graph is None:
            _require_reset("render()")

        schedules: list[list[GanttBar]] = [[] for _ in range(self._p)]
        for rec in self._execution_log:
            end_time = rec["end_time"]
            in_flight = end_time is None
            if in_flight:
                if mode == "committed":
                    continue
                # Scheduled completion = dispatch time + sampled duration.
                end_time = rec["start_time"] + rec["duration"]
            schedules[rec["gpu"]].append(GanttBar(
                op=OpType(rec["op"]).name,
                mb=rec["microbatch"],
                start=rec["start_time"],
                duration=end_time - rec["start_time"],
                end=end_time,
                in_flight=in_flight,
            ))

        pp = self._p
        mbc = self._m
        default_title = (
            f"RL episode Gantt ({mode}, pp={pp}, mbc={mbc}, "
            f"t={self._current_time:.3f})"
        )
        plot_gantt(
            schedules, pp,
            title=title if title is not None else default_title,
            output_path=output_path,
            display=display,
            now=self._current_time if mode == "projected" else None,
        )

    @property
    def current_time(self) -> float:
        return self._current_time
