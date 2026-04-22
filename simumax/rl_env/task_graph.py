"""Task dependency graph for pipeline-parallel scheduling.

Ported from ``rlpp.environment.task_graph`` unchanged apart from the
local ``OpType`` / ``TaskState`` import.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from simumax.rl_env.types import OpType, TaskState


class TaskGraph:
    """DAG of (microbatch, stage, op) tasks with per-task state.

    Tasks are indexed linearly as ``i * (s * 3) + j * 3 + op``, where
    *i* is the microbatch, *j* is the stage, and *op* is F=0 / B=1 / W=2.

    Dependency edges:
    - ``B(i, j)`` requires ``F(i, j)``
    - ``W(i, j)`` requires ``B(i, j)``
    - ``F(i, j)`` requires ``F(i, j-1)`` for ``j > 0``
    - ``B(i, j)`` requires ``B(i, j+1)`` for ``j < s-1``
    """

    def __init__(self, m: int, s: int) -> None:
        self._m = m
        self._s = s
        self._n_tasks = m * s * 3

        self._states = np.full(self._n_tasks, TaskState.BLOCKED, dtype=np.int32)
        self._durations: dict[int, float] = {}

        self._predecessors: list[list[int]] = [[] for _ in range(self._n_tasks)]
        self._successors: list[list[int]] = [[] for _ in range(self._n_tasks)]

        for i in range(m):
            for j in range(s):
                f_idx = self.task_to_index(i, j, OpType.F)
                b_idx = self.task_to_index(i, j, OpType.B)
                w_idx = self.task_to_index(i, j, OpType.W)

                self._add_edge(f_idx, b_idx)
                self._add_edge(b_idx, w_idx)

                if j > 0:
                    f_prev = self.task_to_index(i, j - 1, OpType.F)
                    self._add_edge(f_prev, f_idx)

                if j < s - 1:
                    b_next = self.task_to_index(i, j + 1, OpType.B)
                    self._add_edge(b_next, b_idx)

        for idx in range(self._n_tasks):
            if len(self._predecessors[idx]) == 0:
                self._states[idx] = TaskState.READY

    def _add_edge(self, from_idx: int, to_idx: int) -> None:
        self._predecessors[to_idx].append(from_idx)
        self._successors[from_idx].append(to_idx)

    def task_to_index(self, microbatch: int, stage: int, op: OpType) -> int:
        return microbatch * (self._s * 3) + stage * 3 + int(op)

    def index_to_task(self, index: int) -> tuple[int, int, OpType]:
        op_val = index % 3
        remainder = index // 3
        stage = remainder % self._s
        microbatch = remainder // self._s
        return microbatch, stage, OpType(op_val)

    def get_predecessors(self, index: int) -> list[int]:
        return self._predecessors[index]

    def get_successors(self, index: int) -> list[int]:
        return self._successors[index]

    def get_state(self, index: int) -> TaskState:
        return TaskState(self._states[index])

    def set_state(self, index: int, state: TaskState) -> None:
        self._states[index] = state

    def set_duration(self, index: int, duration: float) -> None:
        self._durations[index] = duration

    def update_blocked_to_ready(self, completed_index: int) -> list[int]:
        """Promote BLOCKED successors to READY after a task completes."""
        newly_ready: list[int] = []
        for succ in self._successors[completed_index]:
            if self._states[succ] != TaskState.BLOCKED:
                continue
            all_done = all(
                self._states[p] == TaskState.DONE for p in self._predecessors[succ]
            )
            if all_done:
                self._states[succ] = TaskState.READY
                newly_ready.append(succ)
        return newly_ready

    def all_done(self) -> bool:
        return bool(np.all(self._states == TaskState.DONE))

    def get_state_matrix(self) -> npt.NDArray[np.int32]:
        return self._states.reshape(self._m, self._s, 3).copy()

    def get_duration_matrix(self) -> npt.NDArray[np.float32]:
        durations = np.full(self._n_tasks, -1.0, dtype=np.float32)
        for idx, dur in self._durations.items():
            durations[idx] = dur
        return durations.reshape(self._m, self._s, 3)

    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @property
    def m(self) -> int:
        return self._m

    @property
    def s(self) -> int:
        return self._s
