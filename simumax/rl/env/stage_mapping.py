"""Stage-to-GPU mapping.

Phase 1 covers only the base case ``s == p`` (physical-rank schedules:
1F1B, GPipe, ZB-H1, ZB-H2). The ZB-V ``s == 2p`` variant is kept for
Phase 3; interleaved zigzag (``s == p * V`` with V > 1) is not yet
implemented here.
"""

from __future__ import annotations


class StageMapping:

    def __init__(self, num_gpus: int, num_stages: int) -> None:
        if num_stages not in (num_gpus, 2 * num_gpus):
            raise ValueError(
                f"num_stages must be {num_gpus} or {2 * num_gpus}, got {num_stages}"
            )
        self._num_gpus = num_gpus
        self._num_stages = num_stages
        self._is_zbv = num_stages == 2 * num_gpus

        if self._is_zbv:
            s2g = [0] * num_stages
            g2s: list[list[int]] = [[] for _ in range(num_gpus)]
            for g in range(num_gpus):
                first = g
                second = 2 * num_gpus - 1 - g
                s2g[first] = g
                s2g[second] = g
                g2s[g] = [first, second]
            self._stage_to_gpu = s2g
            self._gpu_to_stages = [tuple(stages) for stages in g2s]
        else:
            self._stage_to_gpu = list(range(num_gpus))
            self._gpu_to_stages = [(g,) for g in range(num_gpus)]

    def stage_to_gpu(self, stage: int) -> int:
        return self._stage_to_gpu[stage]

    def gpu_to_stages(self, gpu: int) -> tuple[int, ...]:
        return self._gpu_to_stages[gpu]

    @property
    def is_zbv(self) -> bool:
        return self._is_zbv

    @property
    def num_gpus(self) -> int:
        return self._num_gpus

    @property
    def num_stages(self) -> int:
        return self._num_stages
