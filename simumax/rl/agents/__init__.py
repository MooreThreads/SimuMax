"""Agents that drive :class:`simumax.rl.env.env.PipelineSchedulingEnv`.

Two families:

- **Static** (:class:`StaticScheduleAgent` subclasses): classical PP
  schedules ported from ``PerfLLM``. ``gpipe`` / ``1f1b`` are the
  fused-backward-equivalent "classic" versions; ``gpipe_overlap`` /
  ``1f1b_overlap`` are the aggressive B/W-split variants that let W
  on rank ``g+1`` run in parallel with B on rank ``g``. ZB-H1 and
  ZB-H2 are already B/W-aware by construction.
- **Learned** (:class:`PPOAgent`): adapter around a trained
  ``MaskablePPO`` checkpoint; constructed from a path, not a registry
  key (see :mod:`simumax.rl.eval` for CLI wiring).

Interleaved 1F1B and ZB-V are deferred until the env supports
virtual-stage timing.
"""

from __future__ import annotations

from simumax.rl.agents.base import Agent, StaticScheduleAgent
from simumax.rl.agents.gpipe import GPipeAgent, GPipeOverlapAgent
from simumax.rl.agents.one_f_one_b import OneFOneBAgent, OneFOneBOverlapAgent
from simumax.rl.agents.ppo import PPOAgent
from simumax.rl.agents.zb import ZBH1Agent, ZBH2Agent

AGENT_REGISTRY: dict[str, type[StaticScheduleAgent]] = {
    "gpipe": GPipeAgent,
    "gpipe_overlap": GPipeOverlapAgent,
    "1f1b": OneFOneBAgent,
    "1f1b_overlap": OneFOneBOverlapAgent,
    "zb_h1": ZBH1Agent,
    "zb_h2": ZBH2Agent,
}


def make_agent(name: str, pp: int, mbc: int) -> StaticScheduleAgent:
    try:
        cls = AGENT_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(AGENT_REGISTRY))
        msg = f"Unknown static agent {name!r}; available: {available}"
        raise ValueError(msg) from exc
    return cls(pp=pp, mbc=mbc)


__all__ = [
    "AGENT_REGISTRY",
    "Agent",
    "GPipeAgent",
    "GPipeOverlapAgent",
    "OneFOneBAgent",
    "OneFOneBOverlapAgent",
    "PPOAgent",
    "StaticScheduleAgent",
    "ZBH1Agent",
    "ZBH2Agent",
    "make_agent",
]
