"""1F1B agent variants.

Two flavors, both sharing the same per-rank dispatch sequence
(warmup Fs → F/B+W steady state → B+W drain):

- :class:`OneFOneBAgent` — classical semantic: ``B(mb, g)`` waits for
  both ``B(mb, g+1)`` and ``W(mb, g+1)``, so W never overlaps with a
  downstream rank's B. Matches
  :func:`PerfLLM.calculate_1f1b_bubble` iter-time exactly.
- :class:`OneFOneBOverlapAgent` — same queue, but no fused wait, so W
  on rank ``g+1`` can run in parallel with ``B(mb, g)`` on rank ``g``.
"""

from __future__ import annotations

from simumax.rl.agents.base import StaticScheduleAgent
from simumax.rl.env.types import OpType


class _OneFOneBBase(StaticScheduleAgent):

    def _build_queues(self) -> list[list[tuple[OpType, int]]]:
        pp, mbc = self.pp, self.mbc
        queues: list[list[tuple[OpType, int]]] = []
        for g in range(pp):
            # Clipped so mbc < pp degenerate cases still produce a valid queue.
            warmup = min(pp - 1 - g, mbc)
            queue: list[tuple[OpType, int]] = []
            for mb in range(warmup):
                queue.append((OpType.F, mb))
            for mb in range(warmup, mbc):
                queue.append((OpType.F, mb))
                queue.append((OpType.B, mb - warmup))
                queue.append((OpType.W, mb - warmup))
            for mb in range(mbc - warmup, mbc):
                queue.append((OpType.B, mb))
                queue.append((OpType.W, mb))
            queues.append(queue)
        return queues


class OneFOneBAgent(_OneFOneBBase):
    FUSED_BACKWARD = True


class OneFOneBOverlapAgent(_OneFOneBBase):
    FUSED_BACKWARD = False
