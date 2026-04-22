"""GPipe agent variants.

Two flavors, both built on the same per-rank dispatch sequence
(all Fs, then all B/W pairs in reverse):

- :class:`GPipeAgent` — classical semantic: ``B(mb, g)`` waits not only
  for ``B(mb, g+1)`` but also for ``W(mb, g+1)``, so W never overlaps
  with a downstream rank's B. Matches
  :func:`PerfLLM.calculate_gpipe_bubble` iter-time exactly.
- :class:`GPipeOverlapAgent` — same queue, but no fused wait; W on
  rank ``g+1`` is free to run in parallel with ``B(mb, g)`` on rank
  ``g``. Strictly faster than the classical version in a B/W-split env.
"""

from __future__ import annotations

from simumax.rl.agents.base import StaticScheduleAgent
from simumax.rl.env.types import OpType


class _GPipeBase(StaticScheduleAgent):

    def _build_queues(self) -> list[list[tuple[OpType, int]]]:
        queues: list[list[tuple[OpType, int]]] = []
        for _ in range(self.pp):
            queue: list[tuple[OpType, int]] = []
            for mb in range(self.mbc):
                queue.append((OpType.F, mb))
            for mb in range(self.mbc - 1, -1, -1):
                queue.append((OpType.B, mb))
                queue.append((OpType.W, mb))
            queues.append(queue)
        return queues


class GPipeAgent(_GPipeBase):
    FUSED_BACKWARD = True


class GPipeOverlapAgent(_GPipeBase):
    FUSED_BACKWARD = False
