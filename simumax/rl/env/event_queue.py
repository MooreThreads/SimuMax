"""Min-heap priority queue for task completion events.

Ported verbatim from ``rlpp.environment.event_queue``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class CompletionEvent:
    time: float
    task_index: int = field(compare=False)
    gpu: int = field(compare=False)


class EventQueue:
    """Min-heap that pops all events at the earliest timestamp together."""

    _EPSILON = 1e-9

    def __init__(self) -> None:
        self._heap: list[CompletionEvent] = []

    def push(self, event: CompletionEvent) -> None:
        heapq.heappush(self._heap, event)

    def pop_earliest(self) -> list[CompletionEvent]:
        if not self._heap:
            return []
        first = heapq.heappop(self._heap)
        result = [first]
        while self._heap and self._heap[0].time - first.time < self._EPSILON:
            result.append(heapq.heappop(self._heap))
        return result

    def peek_time(self) -> float | None:
        if not self._heap:
            return None
        return self._heap[0].time

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def clear(self) -> None:
        self._heap.clear()
