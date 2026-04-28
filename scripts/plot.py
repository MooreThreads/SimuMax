"""Generate publication-quality figures from ablation sweep results.

Reads parquet outputs produced by ``examples/run_ablation_sweep.py`` and
writes vector PDFs (for LaTeX inclusion) plus PNG previews.

Subcommand-style CLI: each figure is its own subcommand and shares a common
set of flags (``--results-dir``, ``--figs-dir``, …). Adding a new figure:

    1. Write a ``plot_<name>(data, args)`` function below.
    2. Register it in ``FIGURES`` with the data it requires.
    3. Add a subparser in ``parse_args``.

Example
-------
    uv run python scripts/plot.py nominal \\
        --results-dir results/h100_nvlink \\
        --figs-dir paper/figs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Schedule presentation: canonical order, display labels, and a colorblind-safe
# palette (Wong 2011) paired with hatches so prints/photocopies stay legible.
# ----------------------------------------------------------------------------

SCHEDULE_ORDER: List[str] = [
    "gpipe", "1f1b", "interleaved_1f1b", "zb_h1", "zb_h2", "zb_v",
]

SCHEDULE_LABELS: Dict[str, str] = {
    "gpipe":            "GPipe",
    "1f1b":             "1F1B",
    "interleaved_1f1b": "1F1B-I",
    "zb_h1":            "ZB-H1",
    "zb_h2":            "ZB-H2",
    "zb_v":             "ZB-V",
}

# Tol "Vibrant" qualitative palette: colorblind-safe, saturated, widely used
# in publication-quality data viz. Mapped so the ZB family (zb_h1, zb_h2)
# uses related cool tones and the older schedules (gpipe, 1f1b) get the
# strongest warm/blue contrasts.
SCHEDULE_COLORS: Dict[str, str] = {
    "gpipe":            "#0077BB",  # blue
    "1f1b":             "#EE7733",  # orange
    "interleaved_1f1b": "#EE3377",  # magenta
    "zb_h1":            "#009988",  # teal
    "zb_h2":            "#33BBEE",  # cyan
    "zb_v":             "#CC3311",  # red
}


# 16 visually distinct markers — one per model. The 16th is a 6-pointed star
# tuple-spec since matplotlib only ships 15 well-differentiated single-char
# filled markers. Cycled by index when more models appear.
MODEL_MARKERS: List = [
    "o", "s", "^", "v", "<", ">", "D", "d",
    "p", "h", "H", "*", "X", "P", "8", (6, 1, 0),
]


# Approximate total parameter count (B), used for the default x-axis ordering.
# Source: optimal_strategies_h100_gbs256.md.
MODEL_PARAMS_B: Dict[str, float] = {
    "gemma4-31b":             29,
    "gemma3-27b":             30,
    "olmo2-32b":              34,
    "llama3-70b":             79,
    "glm-4.5-air":           106,
    "gpt-oss-120b":          116,
    "mixtral-8x22b":         144,
    "mistral-large":         147,
    "minimax-m2":            228,
    "qwen3-235b-a22b":       234,
    "glm-4.5":               358,
    "llama3-405b":           475,
    "qwen3-coder-480b-a35b": 478,
    "deepseek-r1":           701,
    "kimi-k2":              1045,
    "ling-1t":              1054,
}


# ----------------------------------------------------------------------------
# Style: serif fonts + tight defaults so figures drop straight into LaTeX.
# ----------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif",
                       "Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "grid.linestyle": "-",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType (editable in Illustrator/etc.)
        "ps.fonttype": 42,
    })


# ----------------------------------------------------------------------------
# Data loading (shared across all figures).
# ----------------------------------------------------------------------------

ABLATION_AXES: List[str] = [
    "seq_len_std", "op_duration_std",
    "stage_slowdown_prob", "op_slowdown_prob",
]


@dataclass
class SweepData:
    """Bundle of all parquet outputs from one run of run_ablation_sweep.py."""
    nominal: pd.DataFrame
    baseline: Optional[pd.DataFrame]
    ablations: Dict[str, pd.DataFrame]
    manifest: Dict
    results_dir: Path

    @property
    def models(self) -> List[str]:
        """Models present in the nominal data, in manifest order."""
        present = set(self.nominal["model"])
        manifest_models = self.manifest.get("models", [])
        ordered = [m for m in manifest_models if m in present]
        # Fall back to alphabetical for anything missing from the manifest.
        for m in sorted(present):
            if m not in ordered:
                ordered.append(m)
        return ordered

    @property
    def schedules(self) -> List[str]:
        """Schedules present in the data, in canonical display order."""
        present = set(self.nominal["pp_schedule"])
        ordered = [s for s in SCHEDULE_ORDER if s in present]
        for s in sorted(present):
            if s not in ordered:
                ordered.append(s)
        return ordered


def load_data(results_dir: Path) -> SweepData:
    nominal_path = results_dir / "nominal.parquet"
    if not nominal_path.exists():
        raise FileNotFoundError(f"missing {nominal_path}")
    nominal = pd.read_parquet(nominal_path)

    baseline_path = results_dir / "baseline_disturbed.parquet"
    baseline = (pd.read_parquet(baseline_path)
                if baseline_path.exists() else None)

    ablations: Dict[str, pd.DataFrame] = {}
    for axis in ABLATION_AXES:
        path = results_dir / f"ablation_{axis}.parquet"
        if path.exists():
            ablations[axis] = pd.read_parquet(path)

    manifest_path = results_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})

    return SweepData(
        nominal=nominal, baseline=baseline, ablations=ablations,
        manifest=manifest, results_dir=results_dir,
    )


def _order_models(data: SweepData, args: argparse.Namespace) -> List[str]:
    """X-axis ordering. Default sorts by parameter count (ascending);
    ``manifest`` keeps the manifest order; ``alpha`` sorts alphabetically.

    Models without a known parameter count are appended alphabetically with a
    one-line warning.
    """
    models = data.models
    order = getattr(args, "model_order", "params")
    if order == "alpha":
        return sorted(models)
    if order == "manifest":
        return models
    if order == "params":
        known = [m for m in models if m in MODEL_PARAMS_B]
        unknown = sorted(m for m in models if m not in MODEL_PARAMS_B)
        known.sort(key=lambda m: MODEL_PARAMS_B[m])
        if unknown:
            print(f"warning: no MODEL_PARAMS_B entry for {unknown}; "
                  f"appending alphabetically.")
        return known + unknown
    raise ValueError(f"unknown model_order: {order}")


def _save_figure(fig: plt.Figure, name: str, args: argparse.Namespace) -> None:
    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figs_dir / f"{name}.pdf"
    png_path = figs_dir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


# ----------------------------------------------------------------------------
# Figure: nominal MFU + PP utilization, clustered by model.
# ----------------------------------------------------------------------------

def plot_nominal_by_model(data: SweepData,
                          args: argparse.Namespace) -> None:
    """Two-panel grouped bar chart at peak (no disturbance).

    One cluster per model, one bar per PP schedule; metrics MFU and PP
    utilization stacked vertically with a shared x-axis. No error bars: the
    nominal phase is a single deterministic episode.
    """
    df = data.nominal
    models = _order_models(data, args)
    schedules = data.schedules

    n_models = len(models)
    n_sched = len(schedules)
    cluster_span = 0.82
    bar_width = cluster_span / n_sched
    x = np.arange(n_models)

    fig, axes = plt.subplots(
        2, 1, figsize=(args.fig_width, args.fig_height),
        sharex=True, gridspec_kw={"hspace": 0.18},
    )

    panel_specs = [
        (axes[0], "mfu",            "MFU (\\%)",            "(a)"),
        (axes[1], "pp_utilization", "PP utilization (\\%)", "(b)"),
    ]
    for ax, col, ylabel, panel in panel_specs:
        for i, sched in enumerate(schedules):
            offsets = (i - (n_sched - 1) / 2) * bar_width
            heights = []
            for model in models:
                row = df[(df["model"] == model)
                        & (df["pp_schedule"] == sched)]
                heights.append(
                    float(row[col].iloc[0]) * 100.0 if len(row) else np.nan
                )
            ax.bar(
                x + offsets, heights, width=bar_width,
                color=SCHEDULE_COLORS[sched],
                edgecolor="white", linewidth=0.6,
                label=SCHEDULE_LABELS[sched],
            )
        ax.set_ylabel(ylabel.replace("\\%", "%"))
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", which="major")
        ax.tick_params(axis="x", length=0)
        ax.text(
            -0.045, 1.02, panel, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left",
        )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, rotation=40, ha="right")
    axes[-1].set_xlabel("Model")

    axes[0].legend(
        ncol=n_sched, loc="lower center",
        bbox_to_anchor=(0.5, 1.08), frameon=False,
        columnspacing=1.6, handlelength=1.6, handletextpad=0.5,
    )

    _save_figure(fig, "nominal_by_model", args)


# ----------------------------------------------------------------------------
# Figure: per-axis ablation curves for one (model, schedule) cell.
# ----------------------------------------------------------------------------

# Human-readable axis labels and a hint about the ablation_value units.
# seq_len_std stores the multiplier (× seq_len_mean) in `ablation_value`;
# the other axes store the raw value passed to DisturbanceConfig.
_AXIS_XLABELS: Dict[str, str] = {
    "seq_len_std":         r"seq_len std (× seq_len_mean)",
    "op_duration_std":     r"op duration std",
    "stage_slowdown_prob": r"stage slowdown probability",
    "op_slowdown_prob":    r"op slowdown probability",
}


def _compact_schedule(name: str) -> str:
    """Filename-friendly schedule tag: drop underscores so e.g. zb_h2 → zbh2."""
    return name.replace("_", "")


def plot_ablation_for_cell(data: SweepData,
                           args: argparse.Namespace) -> None:
    """One figure per ablation axis, in three modes:

    * Both ``--model`` and ``--schedule`` set: a single curve with the
      detailed style (±1σ band, horizontal nominal reference line, star at
      x=0).
    * Only ``--model`` set: one curve per schedule (color from
      ``SCHEDULE_COLORS``), legend identifies the schedule.
    * Only ``--schedule`` set: one curve per model (color from ``tab20``,
      marker from ``MODEL_MARKERS``), legend identifies the model.

    Each figure has two stacked panels (MFU, PP utilization). Output:
    ``<figs-dir>/<model_or_all>_<sched_or_all>_<axis>.{pdf,png}``.
    """
    model = getattr(args, "model", None)
    schedule = getattr(args, "schedule", None)
    if model is None and schedule is None:
        raise SystemExit(
            "ablation requires at least one of --model or --schedule."
        )

    # curve_specs: (label, color, marker, model, schedule)
    if model is not None and schedule is not None:
        mode = "single"
        curve_specs = [(
            SCHEDULE_LABELS.get(schedule, schedule),
            SCHEDULE_COLORS.get(schedule, "#444444"), "o",
            model, schedule,
        )]
        out_tag = f"{model}_{_compact_schedule(schedule)}"
    elif model is not None:
        mode = "vary_schedule"
        curve_specs = [
            (SCHEDULE_LABELS.get(s, s),
             SCHEDULE_COLORS.get(s, "#444444"), "o",
             model, s)
            for s in data.schedules
        ]
        out_tag = f"{model}_all"
    else:
        mode = "vary_model"
        models = _order_models(data, args)
        cmap = plt.get_cmap("tab20")
        curve_specs = [
            (m, cmap(i % 20),
             MODEL_MARKERS[i % len(MODEL_MARKERS)],
             m, schedule)
            for i, m in enumerate(models)
        ]
        out_tag = f"all_{_compact_schedule(schedule)}"

    written_any = False
    for axis in ABLATION_AXES:
        if axis not in data.ablations:
            print(f"warning: missing ablation_{axis}.parquet, skipping")
            continue
        df = data.ablations[axis]

        curves = []
        for label, color, marker, m, s in curve_specs:
            nom = data.nominal[(data.nominal["model"] == m)
                               & (data.nominal["pp_schedule"] == s)]
            sub = df[(df["model"] == m) & (df["pp_schedule"] == s)]
            if nom.empty or sub.empty:
                print(f"warning: skipping curve ({m}, {s}) for axis={axis}")
                continue
            agg = (sub.groupby("ablation_value")[["mfu", "pp_utilization"]]
                       .agg(["mean", "std"]).sort_index())
            x_sweep = np.asarray(agg.index, dtype=float)
            mfu_mean = agg[("mfu", "mean")].to_numpy() * 100.0
            mfu_std = np.nan_to_num(agg[("mfu", "std")].to_numpy(),
                                    nan=0.0) * 100.0
            util_mean = agg[("pp_utilization", "mean")].to_numpy() * 100.0
            util_std = np.nan_to_num(
                agg[("pp_utilization", "std")].to_numpy(), nan=0.0,
            ) * 100.0
            nom_mfu = float(nom["mfu"].iloc[0]) * 100.0
            nom_util = float(nom["pp_utilization"].iloc[0]) * 100.0
            x = np.concatenate(([0.0], x_sweep))
            mfu = np.concatenate(([nom_mfu], mfu_mean))
            mfu_s = np.concatenate(([0.0], mfu_std))
            util = np.concatenate(([nom_util], util_mean))
            util_s = np.concatenate(([0.0], util_std))
            curves.append({
                "label": label, "color": color, "marker": marker,
                "x": x, "mfu": mfu, "mfu_s": mfu_s,
                "util": util, "util_s": util_s,
                "nom_mfu": nom_mfu, "nom_util": nom_util,
            })

        if not curves:
            print(f"warning: no curves to render for axis={axis}; skipping")
            continue

        fig, axes = plt.subplots(
            2, 1, figsize=(args.fig_width, args.fig_height),
            sharex=True, gridspec_kw={"hspace": 0.18},
        )
        panel_specs = [
            ("mfu", "mfu_s", "nom_mfu", "MFU (%)", "(a)"),
            ("util", "util_s", "nom_util", "PP utilization (%)", "(b)"),
        ]
        for panel_idx, (mean_key, std_key, nom_key, ylabel,
                        panel) in enumerate(panel_specs):
            ax = axes[panel_idx]
            for c in curves:
                color = c["color"]
                marker = c["marker"]
                x = c["x"]
                ymean = c[mean_key]
                ystd = c[std_key]
                nom_val = c[nom_key]
                if mode == "single":
                    ax.axhline(nom_val, color=color, linestyle=":",
                               linewidth=0.8, alpha=0.55, zorder=1)
                    ax.fill_between(x, ymean - ystd, ymean + ystd,
                                    color=color, alpha=0.18, linewidth=0,
                                    zorder=2)
                line_label = c["label"] if panel_idx == 0 else None
                ax.plot(x[1:], ymean[1:], "-", color=color, linewidth=1.4,
                        marker=marker, markersize=3.6,
                        markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3, label=line_label)
                ax.plot(x[:2], ymean[:2], "--", color=color, linewidth=0.9,
                        alpha=0.55, zorder=2)
                if mode == "single":
                    ax.plot(x[0], ymean[0], marker="*", markersize=9,
                            color=color, markerfacecolor=color,
                            markeredgecolor="white", markeredgewidth=0.6,
                            zorder=4)
                else:
                    ax.plot(x[0], ymean[0], marker=marker, markersize=4.5,
                            color=color, markerfacecolor=color,
                            markeredgecolor="white", markeredgewidth=0.5,
                            zorder=4)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 100)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.grid(axis="y", which="major")
            ax.text(-0.06, 1.02, panel, transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="bottom", ha="left")

        all_x = sorted({float(v) for c in curves for v in c["x"]})
        axes[-1].set_xticks(all_x)
        axes[-1].set_xticklabels([f"{v:g}" for v in all_x])
        axes[-1].set_xlabel(_AXIS_XLABELS.get(axis, axis))

        if mode == "single":
            color = curves[0]["color"]
            legend_handles = [
                plt.Line2D([0], [0], marker="*", color=color, linestyle="",
                           markersize=9, markerfacecolor=color,
                           markeredgecolor="white", markeredgewidth=0.6,
                           label="nominal"),
                plt.Line2D([0], [0], marker="o", color=color, linestyle="-",
                           linewidth=1.4, markersize=3.6,
                           markerfacecolor=color, markeredgecolor="white",
                           markeredgewidth=0.5,
                           label="mean over episodes"),
                mpl.patches.Patch(facecolor=color, alpha=0.18,
                                  label=r"$\pm 1\sigma$ band"),
            ]
            axes[0].legend(
                handles=legend_handles, ncol=3, loc="lower center",
                bbox_to_anchor=(0.5, 1.06), frameon=False,
                columnspacing=1.6, handlelength=1.6, handletextpad=0.5,
            )
        else:
            ncol = min(len(curves), 6 if mode == "vary_schedule" else 4)
            axes[0].legend(
                loc="lower center", bbox_to_anchor=(0.5, 1.06),
                frameon=False, ncol=ncol,
                columnspacing=1.2, handlelength=1.6, handletextpad=0.5,
                fontsize=7 if mode == "vary_model" else 8,
            )

        out_name = f"{out_tag}_{axis}"
        _save_figure(fig, out_name, args)
        written_any = True

    if not written_any:
        raise SystemExit(
            "no ablation parquet files contained data for the selection."
        )


# ----------------------------------------------------------------------------
# Figure: full-disturbance baseline MFU + PP utilization, clustered by model.
# ----------------------------------------------------------------------------

def plot_baseline_by_model(data: SweepData,
                           args: argparse.Namespace) -> None:
    """Two-panel grouped bar chart under the full-disturbance baseline.

    Mirrors ``plot_nominal_by_model`` (same model order, same schedule
    legend, same colors) but heights are the mean over episodes from
    ``baseline_disturbed.parquet`` with ±1σ error bars; the underlying
    disturbance profile is ``configs/disturbance/both.json``.
    """
    if data.baseline is None:
        raise SystemExit(
            "baseline figure requires baseline_disturbed.parquet."
        )
    df = data.baseline
    models = _order_models(data, args)
    schedules = data.schedules

    # Aggregate per (model, schedule); std is NaN for groups of size 1.
    agg = (df.groupby(["model", "pp_schedule"])[["mfu", "pp_utilization"]]
              .agg(["mean", "std"]))

    n_models = len(models)
    n_sched = len(schedules)
    cluster_span = 0.82
    bar_width = cluster_span / n_sched
    x = np.arange(n_models)

    fig, axes = plt.subplots(
        2, 1, figsize=(args.fig_width, args.fig_height),
        sharex=True, gridspec_kw={"hspace": 0.18},
    )

    panel_specs = [
        (axes[0], "mfu",            "MFU (%)",            "(a)"),
        (axes[1], "pp_utilization", "PP utilization (%)", "(b)"),
    ]
    for ax, col, ylabel, panel in panel_specs:
        for i, sched in enumerate(schedules):
            offsets = (i - (n_sched - 1) / 2) * bar_width
            heights, errors = [], []
            for model in models:
                key = (model, sched)
                if key in agg.index:
                    mean_v = float(agg.loc[key, (col, "mean")]) * 100.0
                    std_raw = agg.loc[key, (col, "std")]
                    std_v = (0.0 if pd.isna(std_raw)
                             else float(std_raw) * 100.0)
                else:
                    mean_v, std_v = np.nan, 0.0
                heights.append(mean_v)
                errors.append(std_v)
            ax.bar(
                x + offsets, heights, width=bar_width,
                color=SCHEDULE_COLORS[sched],
                edgecolor="white", linewidth=0.6,
                yerr=errors,
                error_kw={"elinewidth": 0.6, "capsize": 1.5,
                          "ecolor": "#222222", "alpha": 0.75},
                label=SCHEDULE_LABELS[sched],
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", which="major")
        ax.tick_params(axis="x", length=0)
        ax.text(
            -0.045, 1.02, panel, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left",
        )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, rotation=40, ha="right")
    axes[-1].set_xlabel("Model")

    axes[0].legend(
        ncol=n_sched, loc="lower center",
        bbox_to_anchor=(0.5, 1.08), frameon=False,
        columnspacing=1.6, handlelength=1.6, handletextpad=0.5,
    )

    _save_figure(fig, "baseline_by_model", args)


# ----------------------------------------------------------------------------
# Figure: nominal-vs-disturbed scatter (PP utilization × MFU), per model.
# ----------------------------------------------------------------------------

# Two-tone palette for the (nominal, disturbed) state pair, picked from the
# Tol Vibrant palette so it stays consistent with SCHEDULE_COLORS without
# clashing when the schedule color also appears elsewhere.
STATE_COLORS: Dict[str, str] = {
    "nominal":   "#0077BB",  # blue
    "disturbed": "#EE7733",  # orange
}


def plot_nominal_vs_disturbed_scatter(data: SweepData,
                                      args: argparse.Namespace) -> None:
    """Scatter of (PP utilization, MFU) per model, paired across states.

    For a fixed schedule, each model contributes two points: nominal (single
    deterministic episode) and disturbed (mean ± 1σ over episodes from the
    full-disturbance ``baseline_disturbed.parquet``). Marker shape encodes
    the model, color encodes the state. A faint grey segment connects each
    pair so the reader can read each model's "fragility vector" at a glance.

    Output: ``<figs-dir>/nominal_vs_disturbed_<schedule_compact>.{pdf,png}``.
    """
    if data.baseline is None:
        raise SystemExit(
            "scatter figure requires baseline_disturbed.parquet."
        )

    schedule = args.schedule
    sched_tag = _compact_schedule(schedule)

    nom_df = data.nominal[data.nominal["pp_schedule"] == schedule]
    dis_df = data.baseline[data.baseline["pp_schedule"] == schedule]
    if nom_df.empty or dis_df.empty:
        raise SystemExit(
            f"no rows for schedule={schedule} in nominal or baseline data."
        )

    models = _order_models(data, args)
    color_n = STATE_COLORS["nominal"]
    color_d = STATE_COLORS["disturbed"]

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    pairs = []  # (model, marker, x_n, y_n, x_d, y_d, x_d_std, y_d_std)
    for i, model in enumerate(models):
        marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        nom_row = nom_df[nom_df["model"] == model]
        dis_rows = dis_df[dis_df["model"] == model]
        if nom_row.empty or dis_rows.empty:
            print(f"warning: missing nominal or baseline rows for {model} "
                  f"({schedule}); skipping.")
            continue
        x_n = float(nom_row["pp_utilization"].iloc[0]) * 100.0
        y_n = float(nom_row["mfu"].iloc[0]) * 100.0
        x_d = float(dis_rows["pp_utilization"].mean()) * 100.0
        y_d = float(dis_rows["mfu"].mean()) * 100.0
        x_d_std = float(dis_rows["pp_utilization"].std(ddof=1) or 0.0) * 100.0
        y_d_std = float(dis_rows["mfu"].std(ddof=1) or 0.0) * 100.0
        pairs.append((model, marker, x_n, y_n, x_d, y_d, x_d_std, y_d_std))

    # Connector segments first so markers sit on top.
    for _, _, x_n, y_n, x_d, y_d, *_ in pairs:
        ax.plot([x_n, x_d], [y_n, y_d], "-", color="#888888",
                linewidth=0.7, alpha=0.5, zorder=1)

    for _, marker, x_n, y_n, x_d, y_d, x_d_std, y_d_std in pairs:
        # Error bars only (fmt="none"): errorbar's `fmt` expects a format
        # string, which doesn't accept tuple-spec markers like (6, 1, 0).
        ax.errorbar(
            x_d, y_d, xerr=x_d_std, yerr=y_d_std,
            fmt="none", elinewidth=0.7, capsize=2,
            ecolor=color_d, alpha=0.9, zorder=2,
        )
        ax.plot(
            x_d, y_d, marker=marker, color=color_d, linestyle="",
            markerfacecolor=color_d, markeredgecolor="white",
            markeredgewidth=0.6, markersize=7, zorder=3,
        )
        ax.plot(
            x_n, y_n, marker=marker, color=color_n, linestyle="",
            markerfacecolor=color_n, markeredgecolor="white",
            markeredgewidth=0.6, markersize=7, zorder=4,
        )

    ax.set_xlabel("PP utilization (%)")
    ax.set_ylabel("MFU (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, which="major")
    ax.set_aspect("equal", adjustable="box")

    state_handles = [
        plt.Line2D([0], [0], marker="o", color=color_n, linestyle="",
                   markerfacecolor=color_n, markeredgecolor="white",
                   markeredgewidth=0.6, markersize=7, label="nominal"),
        plt.Line2D([0], [0], marker="o", color=color_d, linestyle="",
                   markerfacecolor=color_d, markeredgecolor="white",
                   markeredgewidth=0.6, markersize=7,
                   label=r"disturbed (mean $\pm 1\sigma$)"),
    ]
    state_legend = ax.legend(
        handles=state_handles, loc="upper left", frameon=False,
        handletextpad=0.5,
    )
    ax.add_artist(state_legend)

    model_handles = [
        plt.Line2D([0], [0], marker=marker, color="#444444", linestyle="",
                   markerfacecolor="#444444", markeredgecolor="white",
                   markeredgewidth=0.5, markersize=6, label=model)
        for model, marker, *_ in pairs
    ]
    ax.legend(
        handles=model_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
        columnspacing=1.0, handletextpad=0.4, fontsize=7,
    )

    out_name = f"nominal_vs_disturbed_{sched_tag}"
    _save_figure(fig, out_name, args)


# ----------------------------------------------------------------------------
# Figure registry + dispatch.
# ----------------------------------------------------------------------------

@dataclass
class FigureSpec:
    func: Callable[[SweepData, argparse.Namespace], None]
    needs_baseline: bool = False
    needs_ablations: List[str] = field(default_factory=list)


FIGURES: Dict[str, FigureSpec] = {
    "nominal":  FigureSpec(plot_nominal_by_model),
    "baseline": FigureSpec(plot_baseline_by_model, needs_baseline=True),
    "ablation": FigureSpec(plot_ablation_for_cell,
                           needs_ablations=ABLATION_AXES),
    "scatter":  FigureSpec(plot_nominal_vs_disturbed_scatter,
                           needs_baseline=True),
}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory with parquet outputs from run_ablation_sweep.py.",
    )
    parser.add_argument(
        "--figs-dir", default="paper/figs",
        help="Output directory for PDF + PNG figures.",
    )
    parser.add_argument(
        "--fig-width", type=float, default=7.0,
        help="Figure width in inches. Default 7.0 (≈ \\textwidth).",
    )
    parser.add_argument(
        "--fig-height", type=float, default=4.4,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--model-order", choices=["params", "manifest", "alpha"],
        default="params",
        help="X-axis model ordering. Default sorts ascending by parameter "
             "count (MODEL_PARAMS_B).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="figure", required=True,
                                metavar="FIGURE")

    p_nom = sub.add_parser(
        "nominal",
        help="Two-panel grouped bar chart of nominal MFU and PP "
             "utilization (clustered by model, one bar per schedule).",
    )
    _add_common_args(p_nom)

    p_base = sub.add_parser(
        "baseline",
        help="Two-panel grouped bar chart under the full-disturbance "
             "baseline (configs/disturbance/both.json), clustered by "
             "model, one bar per schedule. Mirrors `nominal` but uses "
             "baseline_disturbed.parquet (mean ± 1σ over episodes).",
    )
    _add_common_args(p_base)

    p_abl = sub.add_parser(
        "ablation",
        help="One figure per ablation axis. Modes: pass --model and "
             "--schedule for a single cell; only --model for one curve "
             "per schedule; only --schedule for one curve per model. At "
             "least one of the two is required.",
    )
    _add_common_args(p_abl)
    # Column-fit defaults; tighter than the wide nominal-by-model figure.
    p_abl.set_defaults(fig_width=5.4, fig_height=4.0)
    p_abl.add_argument(
        "--model", default=None,
        help="Model name (canonical, as in configs/models/). Omit to "
             "produce one curve per schedule for the given --schedule.",
    )
    p_abl.add_argument(
        "--schedule", default=None,
        help="PP schedule name (canonical, as in configs/pp_scheduling/). "
             "Omit to produce one curve per model for the given --model.",
    )

    p_sca = sub.add_parser(
        "scatter",
        help="Single scatter (PP utilization × MFU) for a fixed schedule. "
             "One marker per model, two colors (nominal vs full-disturbance "
             "baseline) with paired connector segments.",
    )
    _add_common_args(p_sca)
    # Roughly square panel; smaller than the wide grouped-bar default.
    p_sca.set_defaults(fig_width=5.5, fig_height=5.0)
    p_sca.add_argument(
        "--schedule", default="zb_h2",
        help="PP schedule name (canonical, as in configs/pp_scheduling/).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    spec = FIGURES[args.figure]
    data = load_data(Path(args.results_dir))

    if spec.needs_baseline and data.baseline is None:
        raise SystemExit(
            f"figure '{args.figure}' requires baseline_disturbed.parquet "
            f"in {args.results_dir}"
        )
    for axis in spec.needs_ablations:
        if axis not in data.ablations:
            raise SystemExit(
                f"figure '{args.figure}' requires ablation_{axis}.parquet "
                f"in {args.results_dir}"
            )

    spec.func(data, args)


if __name__ == "__main__":
    main()
