"""Common agent interface and base class for static-schedule policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

from numpy.typing import NDArray

from simumax.rl.env.types import OpType, TaskState


@runtime_checkable
class Agent(Protocol):
    """Structural interface every agent — static or learned — must satisfy.

    Implementations do not need to inherit from this; duck-typing is
    sufficient. The eval harness (:mod:`simumax.rl.eval`) relies only
    on these two methods.
    """

    def reset(self) -> None: ...
    def act(
        self,
        obs: dict[str, NDArray[Any]],
        action_mask: NDArray[Any],
    ) -> int: ...


class StaticScheduleAgent(ABC):
    """Deterministic agent that plays a classical PP schedule.

    Each concrete subclass overrides :meth:`_build_queues` to emit a
    per-rank ordered list of ``(op, microbatch)`` events derived from
    ``(pp, mbc)`` alone. At each step :meth:`act` fires the first rank
    whose head-of-queue passes :meth:`_can_fire`; if none can fire, it
    returns the no-op action so the env can advance time.

    Setting :attr:`FUSED_BACKWARD` makes :meth:`_can_fire` also block
    ``B(mb, g)`` until ``W(mb, g+1)`` has completed — emulating the
    classical fused-backward semantics in the env's B/W-split task
    graph. Leave it ``False`` for schedules that intentionally let W
    overlap (ZB-family, or the ``_overlap`` variants of 1F1B/GPipe).
    """

    FUSED_BACKWARD: ClassVar[bool] = False

    def __init__(self, pp: int, mbc: int) -> None:
        self.pp = pp
        self.mbc = mbc
        # Phase 1 scope: stage index equals physical rank.
        self._s = pp
        self._queues: list[list[tuple[OpType, int]]] = self._build_queues()
        if len(self._queues) != pp:
            msg = (
                f"_build_queues must return {pp} per-rank queues, "
                f"got {len(self._queues)}"
            )
            raise ValueError(msg)
        self._cursors: list[int] = [0] * pp
        self._done_state = int(TaskState.DONE)

    @abstractmethod
    def _build_queues(self) -> list[list[tuple[OpType, int]]]:
        """Return per-rank dispatch sequences of ``(op, mb)`` pairs."""

    def reset(self) -> None:
        self._cursors = [0] * self.pp

    def _n_tasks(self) -> int:
        return self.mbc * self._s * 3

    def _task_to_index(self, mb: int, stage: int, op: OpType) -> int:
        # Matches TaskGraph.task_to_index (simumax/rl/env/task_graph.py).
        return mb * (self._s * 3) + stage * 3 + int(op)

    def _can_fire(
        self,
        rank: int,
        op: OpType,
        mb: int,
        obs: dict[str, NDArray[Any]],
        action_mask: NDArray[Any],
        action: int,
    ) -> bool:
        if not action_mask[action]:
            return False
        if not self.FUSED_BACKWARD:
            return True
        if op != OpType.B or rank == self.pp - 1:
            return True
        w_neighbor = self._task_to_index(mb, rank + 1, OpType.W)
        return int(obs["task_state"][w_neighbor]) == self._done_state

    def act(
        self,
        obs: dict[str, NDArray[Any]],
        action_mask: NDArray[Any],
    ) -> int:
        for rank in range(self.pp):
            cur = self._cursors[rank]
            if cur >= len(self._queues[rank]):
                continue
            op, mb = self._queues[rank][cur]
            action = self._task_to_index(mb, rank, op)
            if self._can_fire(rank, op, mb, obs, action_mask, action):
                self._cursors[rank] += 1
                return action
        return self._n_tasks()
