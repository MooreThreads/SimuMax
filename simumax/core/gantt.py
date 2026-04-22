"""Shared Gantt plotting for pipeline-parallel schedules.

Consumed by both ``PerfLLM``'s closed-form bubble calculators and
``PipelineSchedulingEnv.render``: both paths produce per-rank task
timelines and want the same visual conventions (bar heights, colors,
Stage labels top-to-bottom, dashed x-grid).

In-flight bars (``in_flight=True``) get hatched fill + lower alpha, and
an optional ``now`` cursor draws a vertical line — both only exercised
by the RL env when rendering a partial episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt


DEFAULT_COLORS = {"F": "#6b8ec9", "B": "#6db5b5", "W": "#5fa75f"}


@dataclass(frozen=True)
class GanttBar:
    """One task occurrence on one rank's timeline.

    ``vs_local`` is only set for virtual-stage schedules (interleaved
    1F1B, ZB-V) where ``vs_local > 0`` gets white text for contrast
    against the darker second band. ``in_flight`` is used by the RL
    env to distinguish tasks that are mid-execution at render time.
    """

    op: str
    mb: int
    start: float
    duration: float
    end: float
    vs_local: Optional[int] = None
    in_flight: bool = False
    # ZB-V overrides the default ``f"{op}{mb + 1}"`` label to include the
    # global virtual stage (e.g. ``"F3.2"``); leave None to use the default.
    label: Optional[str] = None


def plot_gantt(
    schedules: list[list[GanttBar]],
    pp: int,
    *,
    title: str,
    output_path: Optional[str] = None,
    display: bool = False,
    y_label_prefix: str = "Stage",
    figsize: tuple[float, float] = (14, 5),
    colors: Optional[dict[str, str]] = None,
    now: Optional[float] = None,
    label_fontsize: int = 8,
) -> None:
    """Render ``schedules`` as a Gantt chart.

    ``schedules[rank]`` is the list of bars to draw on rank's row.
    Rows are laid out top-to-bottom (rank 0 at top) via ``y = pp - 1 - rank``
    to match the existing SimuMax convention.

    At least one of ``output_path`` (save to disk) or ``display`` (open a
    GUI window via ``plt.show()``) must be truthy. Both may be combined —
    e.g. live-view while also dumping a PNG.
    """
    if output_path is None and not display:
        msg = "plot_gantt needs output_path, display=True, or both."
        raise ValueError(msg)
    palette = colors if colors is not None else DEFAULT_COLORS

    fig, ax = plt.subplots(figsize=figsize)
    for rank, tasks in enumerate(schedules):
        for bar in tasks:
            y = pp - 1 - rank
            face = palette[bar.op]
            if bar.in_flight:
                ax.barh(
                    y=y, width=bar.duration, left=bar.start, height=0.6,
                    color=face, edgecolor="black", alpha=0.45, hatch="//",
                )
            else:
                ax.barh(
                    y=y, width=bar.duration, left=bar.start, height=0.6,
                    color=face, edgecolor="black",
                )
            text_color = "white" if (bar.vs_local or 0) > 0 else "black"
            label = bar.label if bar.label is not None else f"{bar.op}{bar.mb + 1}"
            ax.text(
                bar.start + bar.duration / 2, y, label,
                va="center", ha="center",
                fontsize=label_fontsize, color=text_color,
            )

    if now is not None:
        ax.axvline(x=now, color="red", linestyle=":", linewidth=1.2, alpha=0.8)

    ax.set_yticks(range(pp))
    ax.set_yticklabels([f"{y_label_prefix} {i}" for i in reversed(range(pp))])
    ax.set_xlabel("Time")
    ax.set_title(title)
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    if display:
        # Blocks until the user closes the window. Use a non-Agg backend
        # (the default when importing matplotlib.pyplot without prior
        # ``matplotlib.use("Agg")``) for this to actually pop up.
        plt.show()
    plt.close(fig)
