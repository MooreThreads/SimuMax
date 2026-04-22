"""Enumeration types for the RL scheduling environment."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class OpType(IntEnum):
    F = 0
    B = 1
    W = 2


class TaskState(IntEnum):
    BLOCKED = 0
    READY = 1
    RUNNING = 2
    DONE = 3


class RewardMode(StrEnum):
    MAKESPAN = "makespan"
    BUBBLE = "bubble"
    UTILIZATION = "utilization"
