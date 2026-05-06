#!/usr/bin/env python3
"""Generate public B200 release charts from the release summary JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _seq_value(case_name: str) -> int:
    match = re.search(r"seq(\d+)", case_name)
    if not match:
        raise ValueError(f"case has no seq field: {case_name}")
    return int(match.group(1))


def _seq_label(case_name: str) -> str:
    return f"{_seq_value(case_name) // 1024}K"


def _model_order(row: dict) -> int:
    return {"llama3-70b": 0, "llama3-405b": 1}.get(row.get("model"), 99)


def _parallel_order(row: dict) -> int:
    parallel = row.get("parallel", "")
    if "tp2cp4" in parallel:
        return 0
    if "tp1cp8" in parallel:
        return 1
    return 99


def _short_label(row: dict) -> str:
    model = (
        row["model"]
        .replace("llama3-", "")
        .replace("70b", "70B")
        .replace("405b", "405B")
    )
    parallel = row["parallel"].replace("pp1", "")
    label = f"{model} l{row['layer_num']} {parallel} {_seq_label(row['case'])}"
    return " ".join(label.split())


def _load_cp_rows(summary_path: Path) -> list[dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [row for row in summary["rows"] if "seq" in row["case"]]
    return sorted(
        rows,
        key=lambda row: (
            _seq_value(row["case"]),
            _model_order(row),
            _parallel_order(row),
        ),
    )


def plot_b200_cp_a2a(summary_path: Path, output_path: Path) -> None:
    rows = _load_cp_rows(summary_path)
    labels = [_short_label(row) for row in rows]
    x_pos = np.arange(len(rows))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(12.18, 8.64), dpi=150)
    fig.suptitle(
        "B200 CP A2A Perf vs Real (v1.2 release results)",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )

    timing_ax = axes[0]
    real_ms = [row["real_ms"] for row in rows]
    perf_ms = [row["perf_ms"] for row in rows]
    timing_ax.bar(x_pos - width / 2, real_ms, width, label="Real ms", color="#4c78a8")
    timing_ax.bar(x_pos + width / 2, perf_ms, width, label="Perf ms", color="#f58518")
    timing_ax.set_ylabel("Iteration time (ms)", fontsize=13)
    timing_ax.set_ylim(0, max(real_ms + perf_ms) * 1.18)
    _style_axis(timing_ax, x_pos, labels)
    timing_ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=13)
    for index, row in enumerate(rows):
        y_pos = max(real_ms[index], perf_ms[index])
        timing_ax.text(
            index,
            y_pos * 1.02,
            f"{row['rel_err_pct']:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    memory_ax = axes[1]
    real_alloc = [row["real_alloc_gib"] for row in rows]
    perf_alloc = [row["perf_alloc_gib"] for row in rows]
    memory_ax.bar(
        x_pos - width / 2,
        real_alloc,
        width,
        label="Real alloc GiB",
        color="#54a24b",
    )
    memory_ax.bar(
        x_pos + width / 2,
        perf_alloc,
        width,
        label="Perf alloc GiB",
        color="#b279a2",
    )
    memory_ax.set_ylabel("Peak alloc (GiB)", fontsize=13)
    memory_ax.set_ylim(0, max(real_alloc + perf_alloc) * 1.18)
    _style_axis(memory_ax, x_pos, labels)
    memory_ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=13)
    for index, row in enumerate(rows):
        y_pos = max(real_alloc[index], perf_alloc[index])
        memory_ax.text(
            index,
            y_pos * 1.025,
            f"{row['alloc_err_pct']:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=1.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _style_axis(ax, x_pos, labels) -> None:
    ax.set_xticks(x_pos, labels, rotation=27, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("docs/b200/b200_release_v1.2_summary.json"),
        help="B200 release summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/b200_cp_a2a_release_v1.2.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()
    plot_b200_cp_a2a(args.summary_json, args.output)


if __name__ == "__main__":
    main()
