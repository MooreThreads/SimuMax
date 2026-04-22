"""Zero-bubble agents ZB-H1 and ZB-H2 (Qi et al., ICLR 2024).

Shared queue builder parameterised by ``n_warmup_fn`` and
``mem_limit`` — identical to ``PerfLLM._calculate_zb_bubble``'s
``primary_tracks`` construction (the ``fill_tracks`` branch there is
dead code; all Ws land in the primary sequence).
"""

from __future__ import annotations

from collections.abc import Callable

from simumax.rl.agents.base import StaticScheduleAgent
from simumax.rl.env.types import OpType


def _build_zb_queue(
    g: int,
    pp: int,
    mbc: int,
    n_warmup_fn: Callable[[int], int],
    mem_limit: int,
) -> list[tuple[OpType, int]]:
    n_warmup = max(1, min(n_warmup_fn(g), mbc))
    order: list[tuple[OpType, int]] = []
    n_f = 0
    n_b = 0
    n_w = 0
    live = 0

    for _ in range(n_warmup):
        order.append((OpType.F, n_f)); n_f += 1; live += 1
    while live < mem_limit and n_f < mbc and n_b < n_f:
        order.append((OpType.B, n_b)); n_b += 1
        order.append((OpType.F, n_f)); n_f += 1; live += 1
    while n_f < mbc:
        order.append((OpType.B, n_b)); n_b += 1
        order.append((OpType.W, n_w)); n_w += 1; live -= 1
        order.append((OpType.F, n_f)); n_f += 1; live += 1
    while n_b < mbc:
        order.append((OpType.B, n_b)); n_b += 1
        order.append((OpType.W, n_w)); n_w += 1; live -= 1
    while n_w < mbc:
        order.append((OpType.W, n_w)); n_w += 1; live -= 1
    return order


class ZBH1Agent(StaticScheduleAgent):

    def _build_queues(self) -> list[list[tuple[OpType, int]]]:
        pp = self.pp
        return [
            _build_zb_queue(g, pp, self.mbc,
                            n_warmup_fn=lambda g: pp - g,
                            mem_limit=pp)
            for g in range(pp)
        ]


class ZBH2Agent(StaticScheduleAgent):

    def _build_queues(self) -> list[list[tuple[OpType, int]]]:
        pp = self.pp
        return [
            _build_zb_queue(g, pp, self.mbc,
                            n_warmup_fn=lambda g: 2 * (pp - g) - 1,
                            mem_limit=2 * pp - 1)
            for g in range(pp)
        ]
