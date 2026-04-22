"""Gymnasium environment driven by SimuMax's analytical performance model.

Ports the event-driven scheduler from ``rlpp`` (the reference
implementation at ``/Users/a.palmas/WorkFolder/Repos/pp/rlpp``) and
swaps its scalar toy-duration backend for SimuMax's per-(rank, mb, op)
duration table. See ``RL_ENV_ASSESSMENT.md`` at the repo root for the
full design.
"""

from simumax.rl_env.types import OpType, RewardMode, TaskState

__all__ = ["OpType", "RewardMode", "TaskState"]
