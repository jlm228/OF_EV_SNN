"""Plots results contained in a specific folder ``results/`` into figures.

Examples:
--------
One run, plots all figures::

    python plots.py --run results/sweep_fgsm

Two runs, dark theme, PDF as well as PNG::

    python plots.py --run results/sweep_fgsm results/sweep_pgd \
        --theme dark --formats png pdf

Plotting specific figures::

    python plots.py --run results/sweep_fgsm --only epe_vs_epsilon angular_vs_epsilon
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator


# =============================================================================
# Theme
# =============================================================================
# Colour roles come from the validated reference palette. The categorical slots
# are assigned to *entities* (attack objectives) and never cycled or reassigned
# by rank, so an objective keeps its hue across every figure.

@dataclass(frozen=True)
class Theme:
    name: str
    surface: str            # chart surface
    page: str               # figure background
    ink: str                # primary text
    ink2: str               # secondary text -- also the "clean" reference line
    muted: str              # axis/tick labels -- also the "random" control line
    grid: str               # hairline gridlines
    axis: str               # baseline / spines
    series: tuple           # categorical slots 1..3
    seq_ramp: tuple         # sequential ramp, near-zero end first


_BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b")

LIGHT = Theme(
    name="light", surface="#fcfcfb", page="#fcfcfb", ink="#0b0b0b",
    ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"), seq_ramp=_BLUE_RAMP,
)

DARK = Theme(
    name="dark", surface="#1a1a19", page="#1a1a19", ink="#ffffff",
    ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"), seq_ramp=tuple(reversed(_BLUE_RAMP)),
)

THEMES = {"light": LIGHT, "dark": DARK}

# Fixed objective -> categorical slot. Entities, not ranks.
OBJECTIVE_SLOT = {"epe": 0, "angular": 1, "cosine": 2}
OBJECTIVE_LABEL = {"epe": "EPE objective", "angular": "angular objective",
                   "cosine": "cosine objective"}

# Metric key -> (column stem, axis label, short name)
METRICS = {
    "EPE": ("EPE", "End-point error (px)", "EPE"),
    "angular_deg": ("angular_deg", "Angular error (deg)", "angular error"),
    "one_minus_cos": ("one_minus_cos", "1 - cos(pred, gt)", "cosine error"),
}

POLARITY_LABEL = {0: "ON", 1: "OFF"}     # event2frame.py: ch 0 = ON, ch 1 = OFF


def objective_color(theme: Theme, objective: str) -> str:
    """Colour for an attack objective -- fixed per entity, never by rank."""
    return theme.series[OBJECTIVE_SLOT.get(objective, 0) % len(theme.series)]


def sequential_cmap(theme: Theme) -> LinearSegmentedColormap:
    """One-hue sequential ramp for magnitude, near-zero end at the surface."""
    return LinearSegmentedColormap.from_list("seq", list(theme.seq_ramp), N=256)


def apply_theme(theme: Theme) -> None:
    """Thin marks, hairline recessive chrome, system sans."""
    plt.rcParams.update({
        "figure.facecolor": theme.page,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.page,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial",
                            "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": theme.ink2,
        "axes.edgecolor": theme.axis,
        "axes.linewidth": 0.8,
        "axes.titlecolor": theme.ink,
        "text.color": theme.ink,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.muted,
        "ytick.labelcolor": theme.muted,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.color": theme.grid,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",          # solid hairlines only
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": theme.ink2,
        "figure.dpi": 110,
    })


def style_axes(ax, *, xgrid: bool = False) -> None:
    """Recessive chrome: y-hairlines, no top/right spines."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(bool(xgrid))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# =============================================================================
# Data model
# =============================================================================

RAW_GLOB = "raw_{attack}_*.csv"
GRADMAP_RE = re.compile(r"gradmap_(?P<attack>[^_]+)_(?P<objective>[^_]+)_(?P<sequence>.+)\.npy$")

POOL_COLS = ["sequence", "condition", "attack_loss", "epsilon", "n_points",
             "EPE_mean", "EPE_std", "angular_deg_mean", "angular_deg_std",
             "one_minus_cos_mean", "one_minus_cos_std",
             "count_drift_mean", "count_drift_max"]


@dataclass
class RunData:
    """Everything one sweep run wrote, plus the axes it was swept over."""
    attack: str
    root: str
    pooled: pd.DataFrame                       # per (sequence, condition, loss, eps)
    sweep: pd.DataFrame | None = None          # the aggregate curve as written
    raw: pd.DataFrame | None = None            # per-sample rows, when available
    attribution: pd.DataFrame | None = None
    gradmaps: dict = field(default_factory=dict)   # (objective, sequence) -> HxW
    calibration: dict | None = None

    @property
    def sequences(self) -> list:
        return sorted(self.pooled["sequence"].dropna().unique().tolist())

    @property
    def epsilons(self) -> list:
        eps = self.pooled.loc[self.pooled["condition"] == self.attack, "epsilon"]
        return sorted(eps.dropna().unique().tolist())

    @property
    def objectives(self) -> list:
        obj = self.pooled.loc[self.pooled["condition"] == self.attack, "attack_loss"]
        found = [o for o in obj.dropna().unique().tolist() if o]
        # keep the canonical order rather than discovery order
        return sorted(found, key=lambda o: OBJECTIVE_SLOT.get(o, 99))

    @property
    def has_random(self) -> bool:
        return bool((self.pooled["condition"] == "random").any())


def discover_runs(results_dir: str) -> list[str]:
    """Directories under ``results_dir`` that hold a sweep (recursively)."""
    hits = set()
    for path in glob.glob(os.path.join(results_dir, "**", "per_sequence_*.csv"),
                          recursive=True):
        hits.add(os.path.dirname(path))
    for path in glob.glob(os.path.join(results_dir, "**", "sweep_*.csv"),
                          recursive=True):
        hits.add(os.path.dirname(path))
    return sorted(hits)


def _read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # Blank attack_loss / epsilon cells (clean, random) read back as NaN.
    if "attack_loss" in df:
        df["attack_loss"] = df["attack_loss"].astype("object").where(
            df["attack_loss"].notna(), None)
    return df


def _attack_name(run_dir: str) -> str | None:
    for pattern, group in (("per_sequence_*.csv", "per_sequence_"),
                           ("sweep_*.csv", "sweep_")):
        for path in sorted(glob.glob(os.path.join(run_dir, pattern))):
            return os.path.basename(path)[len(group):-len(".csv")]
    return None


def pooled_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-sample rows into the ``per_sequence_*.csv`` schema.

    Used when a run kept its raw files but not the pooled CSV; the two are
    interchangeable downstream because both carry (n, mean, sd) per group.
    """
    keys = ["sequence", "condition", "attack_loss", "epsilon"]
    metrics = list(METRICS)
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, lambda s: float(s.std(ddof=0)))
    agg["n_points"] = (metrics[0], "size")
    agg["count_drift_mean"] = ("count_drift", "mean")
    agg["count_drift_max"] = ("count_drift", "max")
    out = raw.groupby(keys, dropna=False).agg(**agg).reset_index()
    return out[POOL_COLS]


def load_run(run_dir: str, calibration_path: str | None = None) -> RunData:
    """Load one sweep directory into a :class:`RunData`."""
    attack = _attack_name(run_dir)
    if attack is None:
        raise FileNotFoundError(f"No sweep_*/per_sequence_* CSV in {run_dir}")

    pooled = _read_csv(os.path.join(run_dir, f"per_sequence_{attack}.csv"))

    raw_files = sorted(glob.glob(os.path.join(run_dir, RAW_GLOB.format(attack=attack))))
    raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True) if raw_files else None
    if raw is not None and "attack_loss" in raw:
        raw["attack_loss"] = raw["attack_loss"].astype("object").where(
            raw["attack_loss"].notna(), None)

    if pooled is None:
        if raw is None:
            raise FileNotFoundError(
                f"{run_dir}: neither per_sequence_{attack}.csv nor raw_{attack}_*.csv")
        pooled = pooled_from_raw(raw)

    gradmaps = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "gradmap_*.npy"))):
        m = GRADMAP_RE.search(os.path.basename(path))
        if m and m.group("attack") == attack:
            gradmaps[(m.group("objective"), m.group("sequence"))] = np.load(path)

    calibration = None
    if calibration_path and os.path.isfile(calibration_path):
        with open(calibration_path) as fh:
            calibration = json.load(fh)

    return RunData(
        attack=attack,
        root=run_dir,
        pooled=pooled,
        sweep=_read_csv(os.path.join(run_dir, f"sweep_{attack}.csv")),
        raw=raw,
        attribution=_read_csv(os.path.join(run_dir, f"attribution_{attack}.csv")),
        gradmaps=gradmaps,
        calibration=calibration,
    )


# =============================================================================
# Aggregation
# =============================================================================

def _pool_rows(rows: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    """Micro-average (n, mean, sd) of a metric over per-sequence pooled rows.

    Weighted by ``n_points``. The mean and sd are those of the underlying
    sample population, not a mean of per-sequence means, so sequences with more
    samples pull proportionally harder.
    """
    n = rows["n_points"].to_numpy(dtype=float)
    m = rows[f"{metric}_mean"].to_numpy(dtype=float)
    s = rows[f"{metric}_std"].to_numpy(dtype=float)
    total = float(n.sum())
    if total <= 0:
        return 0.0, math.nan, math.nan
    mean = float((n * m).sum() / total)
    ex2 = float((n * (s ** 2 + m ** 2)).sum() / total)
    return total, mean, math.sqrt(max(ex2 - mean ** 2, 0.0))


def select(run: RunData, condition: str, objective: str | None = None,
           sequence: str | None = None) -> pd.DataFrame:
    """Pooled rows for one condition (+ objective / sequence filter)."""
    df = run.pooled
    sel = df["condition"] == condition
    if objective is None:
        sel &= df["attack_loss"].isna() | (df["attack_loss"] == "")
    else:
        sel &= df["attack_loss"] == objective
    if sequence is not None:
        sel &= df["sequence"] == sequence
    return df[sel]


def curve(run: RunData, condition: str, metric: str, objective: str | None = None,
          sequence: str | None = None) -> pd.DataFrame:
    """(epsilon, n, mean, sd, drift) curve, micro-averaged over sequences.

    Pass ``sequence`` to get that sequence alone. Rows are ordered by epsilon.
    """
    rows = select(run, condition, objective, sequence)
    if rows.empty:
        return pd.DataFrame(columns=["epsilon", "n", "mean", "std", "drift"])
    out = []
    for eps, grp in rows.groupby("epsilon", dropna=False):
        n, mean, std = _pool_rows(grp, metric)
        drift_n = grp["n_points"].to_numpy(dtype=float)
        drift = float((grp["count_drift_mean"].to_numpy(dtype=float) * drift_n).sum()
                      / max(drift_n.sum(), 1.0))
        out.append({"epsilon": eps, "n": n, "mean": mean, "std": std, "drift": drift})
    return pd.DataFrame(out).sort_values("epsilon", na_position="first").reset_index(drop=True)


def clean_value(run: RunData, metric: str, sequence: str | None = None) -> tuple[float, float]:
    """(mean, sd) of the clean baseline, epsilon-independent."""
    rows = select(run, "clean", None, sequence)
    if rows.empty:
        return math.nan, math.nan
    _, mean, std = _pool_rows(rows, metric)
    return mean, std


def infer_events_per_sample(run: RunData) -> float | None:
    """Estimate total events per sample, for the "% events injected" axis.

    Every voxel of the input tensor is shifted by +/-epsilon, so the recorded
    ``count_drift`` (= |sum(E_adv) - sum(E)|) is ~ epsilon * numel; the ratio at
    small epsilon recovers the voxel count. 
    
    Multiplying by the mean per-voxel event count from ``epsilon_calibration.json`` 
    gives the event total. Both steps are approximations (drift loses a little to clamping),
    so the axis is labelled as approximate.
    """
    if not run.calibration:
        return None
    mean_count = (run.calibration.get("count_statistics", {})
                  .get("combined", {}).get("mean"))
    if not mean_count:
        return None
    rows = run.pooled[(run.pooled["condition"] == run.attack)
                      & (run.pooled["epsilon"] > 0)]
    if rows.empty:
        return None
    numel = (rows["count_drift_max"] / rows["epsilon"]).max()
    return float(numel * mean_count)


# =============================================================================
# Plot helpers
# =============================================================================

@dataclass
class Ctx:
    """Everything the figure functions need beyond the data itself."""
    theme: Theme
    outdir: str
    formats: tuple = ("png",)
    dpi: int = 200
    band: str = "std"                 # std | sem | none
    xscale: str = "log"               # log | linear, for the epsilon axis
    attr_absolute: bool = False
    gradmap_clip: float = 99.5
    total_events: float | None = None
    data_root: str | None = None      # dataset root, for the activity figure
    activity_samples: int = 40        # tensors read per sequence for that figure
    eps_units: str = "percent"        # percent | epsilon, for the budget axis
    eps_pct: dict | None = None       # epsilon -> % of clean events, when known
    attr_objective: str | None = None  # objective drawn by attribution_temporal


def save(fig, ctx: Ctx, name: str) -> list[str]:
    os.makedirs(ctx.outdir, exist_ok=True)
    paths = []
    for ext in ctx.formats:
        path = os.path.join(ctx.outdir, f"{name}.{ext}")
        fig.savefig(path, dpi=ctx.dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def grid_axes(n: int, *, ncols: int | None = None, panel=(3.1, 2.3),
              sharex=True, sharey=True):
    """Small-multiple grid sized to the panel count; extras are hidden."""
    ncols = ncols or min(3, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                             figsize=(panel[0] * ncols, panel[1] * nrows))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    if sharex:
        # A ragged last row would otherwise leave the columns above the gap
        # with no x tick labels at all.
        for col in range(ncols):
            in_col = [i for i in range(col, n, ncols)]
            if in_col:
                axes[in_col[-1]].tick_params(labelbottom=True)
    return fig, axes[:n]


def band_bounds(df: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    if kind == "none" or df.empty:
        return None
    spread = df["std"].to_numpy(dtype=float)
    if kind == "sem":
        spread = spread / np.sqrt(np.maximum(df["n"].to_numpy(dtype=float), 1.0))
    m = df["mean"].to_numpy(dtype=float)
    return m - spread, m + spread


def visible_epsilons(run: RunData, ctx: Ctx) -> list[float]:
    """Epsilons to put on the x-axis.

    A log axis cannot show epsilon = 0, and that point is by construction the
    clean value, which every figure already draws as a reference rule, so it
    is dropped there rather than special-cased.
    """
    eps = run.epsilons
    return [e for e in eps if e > 0] if ctx.xscale == "log" else eps


def on_axis(df: pd.DataFrame, ctx: Ctx, x: str = "epsilon") -> pd.DataFrame:
    """Drop rows a log x-axis cannot render."""
    if ctx.xscale == "log" and not df.empty:
        return df[df[x] > 0]
    return df


def draw_curve(ax, df: pd.DataFrame, *, color: str, label: str, ctx: Ctx,
               marker: str = "o", x: str = "epsilon", zorder: float = 3,
               linestyle: str = "-", band: bool = True, clip_axis: bool = True):
    # clip_axis=False for a symlog axis, which can render the zero point that a
    # plain log axis cannot -- see eps_axis(include_zero=True).
    if clip_axis:
        df = on_axis(df, ctx, x)
    if df.empty:
        return
    xs = df[x].to_numpy(dtype=float)
    ys = df["mean"].to_numpy(dtype=float)
    if band:
        bounds = band_bounds(df, ctx.band)
        if bounds is not None:
            ax.fill_between(xs, bounds[0], bounds[1], color=color, alpha=0.13,
                            linewidth=0, zorder=zorder - 1)
    ax.plot(xs, ys, color=color, label=label, marker=marker, linestyle=linestyle,
            markeredgecolor=ctx.theme.surface, markeredgewidth=1.0, zorder=zorder)


def draw_reference(ax, value: float, *, color: str, label: str, ctx: Ctx):
    """Flat baseline (clean) as a thin solid rule -- never dashed."""
    ax.axhline(value, color=color, linewidth=1.4, label=label, zorder=2)


def end_labels(ax, items: Sequence[tuple], ctx: Ctx, min_gap_pt: float = 9.0):
    """Direct-label series endpoints, dropping any that would collide.

    ``items`` is (x, y, text, colour). Labels are placed right of the last
    point; where two endpoints land within ``min_gap_pt`` of each other the
    legend carries identity instead, so labels never overprint.
    """
    if not items:
        return
    # Top-down, so when a cluster has to be thinned the highest (headline)
    # series keeps its label rather than whichever happened to sort first.
    ordered = sorted(items, key=lambda it: it[1], reverse=True)
    kept, last_disp = [], None
    for x, y, text, color in ordered:
        disp = ax.transData.transform((x, y))[1] * 72.0 / ax.figure.dpi
        if last_disp is not None and abs(disp - last_disp) < min_gap_pt:
            continue
        kept.append((x, y, text, color))
        last_disp = disp
    for x, y, text, color in kept:
        ax.annotate(text, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                    color=color, fontsize=8, va="center", ha="left",
                    fontweight="semibold", annotation_clip=False)


def titled(fig, title: str, subtitle: str | None, ctx: Ctx, *,
           align: str = "left", gap: float = 1.0):
    """Header above the plot area, aligned to it, in points of clearance.
    """
    fig.canvas.draw()          # resolve tight/constrained layout first
    visible = [ax.get_position() for ax in fig.axes if ax.get_visible()]
    if align == "center":
        x = ((min(b.x0 for b in visible) + max(b.x1 for b in visible)) / 2
             if visible else 0.5)
        ha = "center"
    else:
        x = min((b.x0 for b in visible), default=fig.subplotpars.left)
        ha = "left"
    h = fig.get_figheight()
    fig.text(x, 1.0 + gap * 0.44 / h, title, color=ctx.theme.ink, fontsize=11,
             fontweight="semibold", ha=ha, va="bottom")
    if subtitle:
        fig.text(x, 1.0 + gap * 0.13 / h, subtitle, color=ctx.theme.muted,
                 fontsize=8, ha=ha, va="bottom")


def slide_title(fig, title: str, ctx: Ctx):
    """Header for a figure that will stand alone on a slide.
    """
    titled(fig, title, None, ctx, align="center", gap=0.55)


def panel_legend(fig, ax, ctx: Ctx, ncol: int | None = None):
    """One shared legend for a small-multiple grid, clear of the shared labels.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, labels, loc="upper center", ncol=ncol or len(labels),
               bbox_to_anchor=(0.5, -0.015))


def fmt_eps(e: float) -> str:
    return f"{e:.3g}"


def percent_map(run: RunData, calibration: dict | None) -> dict | None:
    """Swept epsilon -> % of the clean event count it injects. See 
    calibrate_epsilon.py for how the calibration is built.
    """
    if not calibration:
        return None
    table = {float(b["epsilon"]): float(b["percent_of_clean_events"])
             for b in calibration.get("budget", [])}
    voxels = calibration.get("voxels_per_sample")
    events = calibration.get("events_per_sample")
    out = {}
    for eps in run.epsilons:
        if eps <= 0:
            out[eps] = 0.0
            continue
        hit = next((v for k, v in table.items() if math.isclose(k, eps, rel_tol=1e-6)),
                   None)
        if hit is None and voxels and events:
            hit = eps * 0.5 * float(voxels) / float(events) * 100.0
        if hit is not None:
            out[eps] = hit
    return out or None


def fmt_budget(e: float, ctx: Ctx) -> str:
    """Tick label for one budget value, in whichever unit the axis is using."""
    if ctx.eps_units == "percent" and ctx.eps_pct and e in ctx.eps_pct:
        return f"{ctx.eps_pct[e]:g}"
    return fmt_eps(e)


def budget_label(ctx: Ctx, short: bool = False) -> str:
    """Axis label for the budget, matching the unit the ticks are in."""
    if ctx.eps_units == "percent" and ctx.eps_pct:
        return ("Events injected (% of clean)" if short else
                "Perturbation budget  (% of the clean event count injected)")
    return ("Perturbation budget  $\\epsilon$" if short else
            "Perturbation budget  $\\epsilon$  (events / voxel)")


def eps_axis(ax, epsilons: Sequence[float], ctx: Ctx, include_zero: bool = False):
    """Tick the swept epsilons themselves -- a geometric sweep, so log by default.
    """
    ticks = list(epsilons)
    if ctx.xscale == "log":
        positive = [e for e in ticks if e > 0]
        if include_zero and positive:
            ax.set_xscale("symlog", linthresh=min(positive), linscale=0.45)
            ticks = [0.0] + positive
        else:
            ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt_budget(e, ctx) for e in ticks],
                       rotation=0 if ctx.xscale == "log" else 45,
                       ha="center" if ctx.xscale == "log" else "right")
    ax.xaxis.set_minor_locator(NullLocator())


def sample_note(run: RunData) -> str:
    seqs = run.sequences
    n = int(select(run, "clean")["n_points"].sum())
    return f"{n} samples over {len(seqs)} sequence{'s' if len(seqs) != 1 else ''}"


# =============================================================================
# Figure registry
# =============================================================================

FIGURES: dict[str, tuple[Callable, str]] = {}


def figure(name: str, description: str):
    def deco(fn):
        FIGURES[name] = (fn, description)
        return fn
    return deco


class Skip(Exception):
    """Raised by a figure when its inputs are absent."""


# ---------------------------------------------------------------- 1. curves --

def _degradation(run: RunData, ctx: Ctx, metric: str, name: str,
                 standalone: bool = False):
    label = METRICS[metric][1]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    style_axes(ax)

    cmean, _ = clean_value(run, metric)
    draw_reference(ax, cmean, color=ctx.theme.ink2, label="clean", ctx=ctx)

    # Standalone: keep the unperturbed point on the curve rather than leaving it
    # implicit in the clean rule, so the reader can see the attack depart from
    # the baseline instead of inferring it from where the first marker sits.
    show_zero = standalone and ctx.xscale == "log"

    tips = []
    for obj in run.objectives:
        df = curve(run, run.attack, metric, objective=obj)
        if not show_zero:
            df = on_axis(df, ctx)
        color = objective_color(ctx.theme, obj)
        draw_curve(ax, df, color=color, label=f"{run.attack.upper()} ({obj})",
                   ctx=ctx, clip_axis=not show_zero)
        if not df.empty:
            tips.append((df["epsilon"].iloc[-1], df["mean"].iloc[-1], obj, color))

    eps_axis(ax, run.epsilons if show_zero else visible_epsilons(run, ctx), ctx,
             include_zero=show_zero)
    ax.set_xlabel(budget_label(ctx))
    ax.set_ylabel(label)
    handles, labels = ax.get_legend_handles_labels()
    if standalone and ctx.band != "none":
        handles.append(Patch(facecolor=ctx.theme.muted, alpha=0.13, linewidth=0))
        labels.append(f"+/-1 {ctx.band} across samples")
    ax.legend(handles, labels, loc="upper left")
    end_labels(ax, tips, ctx)                          # selective direct labels
    title = f"{label.split(' (')[0]} vs attack budget"
    if standalone:
        # Reclaim the default subplot margin above the axes first:.
        fig.tight_layout()
        slide_title(fig, title, ctx)
        print(f"         {name}: {run.attack.upper()} on DSEC, micro-averaged "
              f"over {sample_note(run)}; clean {metric} = {cmean:.3f}")
    else:
        titled(fig, title,
               f"{run.attack.upper()} on DSEC -- micro-averaged over "
               f"{sample_note(run)}; band = +/-1 sd across samples", ctx)
    return save(fig, ctx, name)


@figure("epe_vs_epsilon", "EPE vs epsilon, averaged (clean + one line per objective)")
def fig_epe_vs_epsilon(run, ctx):
    return _degradation(run, ctx, "EPE", "epe_vs_epsilon", standalone=True)


@figure("angular_vs_epsilon", "Angular error vs epsilon, averaged")
def fig_angular_vs_epsilon(run, ctx):
    return _degradation(run, ctx, "angular_deg", "angular_vs_epsilon")


def _degradation_per_sequence(run: RunData, ctx: Ctx, metric: str, name: str):
    seqs = run.sequences
    label = METRICS[metric][1]
    fig, axes = grid_axes(len(seqs))
    for ax, seq in zip(axes, seqs):
        style_axes(ax)
        cmean, _ = clean_value(run, metric, sequence=seq)
        draw_reference(ax, cmean, color=ctx.theme.ink2, label="clean", ctx=ctx)
        for obj in run.objectives:
            df = curve(run, run.attack, metric, objective=obj, sequence=seq)
            draw_curve(ax, df, color=objective_color(ctx.theme, obj),
                       label=f"{run.attack.upper()} ({obj})", ctx=ctx, marker="o")
        n = int(select(run, "clean", sequence=seq)["n_points"].sum())
        ax.set_title(f"{seq}  ({n})", color=ctx.theme.ink, fontsize=9)
        eps_axis(ax, visible_epsilons(run, ctx), ctx)

    fig.supxlabel(budget_label(ctx, short=True), color=ctx.theme.ink2, fontsize=9)
    fig.supylabel(label, color=ctx.theme.ink2, fontsize=9)
    fig.tight_layout()
    panel_legend(fig, axes[0], ctx)
    titled(fig, f"{label.split(' (')[0]} vs budget, per sequence",
           "Shared axes -- panel title gives the sample count", ctx)
    return save(fig, ctx, name)


@figure("epe_vs_epsilon_per_sequence", "EPE vs epsilon, small multiples per sequence")
def fig_epe_per_sequence(run, ctx):
    return _degradation_per_sequence(run, ctx, "EPE", "epe_vs_epsilon_per_sequence")


@figure("angular_vs_epsilon_per_sequence",
        "Angular error vs epsilon, small multiples per sequence")
def fig_angular_per_sequence(run, ctx):
    return _degradation_per_sequence(run, ctx, "angular_deg",
                                     "angular_vs_epsilon_per_sequence")


@figure("epe_vs_events_injected",
        "EPE vs injected events (physical x-axis) instead of raw epsilon")
def fig_epe_vs_drift(run, ctx):
    total = ctx.total_events or infer_events_per_sample(run)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    style_axes(ax)

    cmean, _ = clean_value(run, "EPE")
    draw_reference(ax, cmean, color=ctx.theme.ink2, label="clean", ctx=ctx)

    scale = (100.0 / total) if total else 1.0
    tips = []
    for obj in run.objectives:
        df = curve(run, run.attack, "EPE", objective=obj).copy()
        df = df[df["drift"] > 0]
        if df.empty:
            continue
        df["x"] = df["drift"] * scale
        color = objective_color(ctx.theme, obj)
        draw_curve(ax, df, color=color, x="x",
                   label=f"{run.attack.upper()} ({obj})", ctx=ctx)
        tips.append((df["x"].iloc[-1], df["mean"].iloc[-1], obj, color))
    if not tips:
        plt.close(fig)
        raise Skip("no count_drift recorded")

    if ctx.xscale == "log":
        ax.set_xscale("log")
    end_labels(ax, tips, ctx)
    ax.set_xlabel("Events injected per sample (% of clean event count, approx.)"
                  if total else "Events injected per sample (count)")
    ax.set_ylabel(METRICS["EPE"][1])
    ax.legend(loc="upper left")
    titled(fig, "Degradation vs injected event mass",
           ("Perturbation expressed physically rather than as epsilon; "
            + (f"total ~ {total:,.0f} events/sample" if total
               else "pass --total-events or --calibration for a % axis")), ctx)
    return save(fig, ctx, "epe_vs_events_injected")


# --------------------------------------------------- 2. attack vs control ----

@figure("attack_vs_random", "FGSM vs magnitude-matched random-sign control vs epsilon")
def fig_attack_vs_random(run, ctx):
    if not run.has_random:
        raise Skip("no random-sign control in this run")
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    style_axes(ax)

    cmean, _ = clean_value(run, "EPE")
    draw_reference(ax, cmean, color=ctx.theme.ink2, label="clean", ctx=ctx)

    rnd = on_axis(curve(run, "random", "EPE"), ctx)
    draw_curve(ax, rnd, color=ctx.theme.muted, label="random sign (same $\\epsilon$)",
               ctx=ctx, marker="s", zorder=3)

    tips = []
    for obj in run.objectives:
        df = on_axis(curve(run, run.attack, "EPE", objective=obj), ctx)
        color = objective_color(ctx.theme, obj)
        draw_curve(ax, df, color=color, label=f"{run.attack.upper()} ({obj})",
                   ctx=ctx, zorder=4)
        if not df.empty:
            tips.append((df["epsilon"].iloc[-1], df["mean"].iloc[-1], obj, color))
    if not rnd.empty:
        tips.append((rnd["epsilon"].iloc[-1], rnd["mean"].iloc[-1], "random",
                     ctx.theme.muted))

    eps_axis(ax, visible_epsilons(run, ctx), ctx)
    ax.set_xlabel(budget_label(ctx))
    ax.set_ylabel(METRICS["EPE"][1])
    ax.legend(loc="upper left")
    end_labels(ax, tips, ctx)
    titled(fig, "Gradient attack vs random-sign control",
           f"Same L-inf magnitude; the gap is what the gradient buys. "
           f"{sample_note(run)}", ctx)
    return save(fig, ctx, "attack_vs_random")


@figure("attack_vs_random_delta",
        "Damage over clean (log): separates the gradient advantage from control noise")
def fig_attack_vs_random_delta(run, ctx):
    if not run.has_random:
        raise Skip("no random-sign control in this run")
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    style_axes(ax)

    cmean, _ = clean_value(run, "EPE")

    dropped = []

    def delta(df, who=""):
        out = df.copy()
        out["mean"] = out["mean"] - cmean
        keep = out["mean"] > 0
        # A log axis cannot show a point at or below the clean baseline. Those
        # are meaningful here (the control sometimes does nothing at all), so
        # they are reported in the subtitle rather than silently vanishing.
        for eps in out.loc[~keep, "epsilon"]:
            dropped.append(f"{who} at eps={fmt_eps(eps)}")
        return out[keep]

    tips = []
    rnd = delta(on_axis(curve(run, "random", "EPE"), ctx), "random")
    draw_curve(ax, rnd, color=ctx.theme.muted, label="random sign (same $\\epsilon$)",
               ctx=ctx, marker="s", zorder=3, band=False)
    for obj in run.objectives:
        df = delta(on_axis(curve(run, run.attack, "EPE", objective=obj), ctx), obj)
        color = objective_color(ctx.theme, obj)
        draw_curve(ax, df, color=color, label=f"{run.attack.upper()} ({obj})",
                   ctx=ctx, zorder=4, band=False)
        if not df.empty:
            tips.append((df["epsilon"].iloc[-1], df["mean"].iloc[-1], obj, color))
    if not rnd.empty:
        tips.append((rnd["epsilon"].iloc[-1], rnd["mean"].iloc[-1], "random",
                     ctx.theme.muted))

    # Flag any epsilon where the control is *worse* than a larger one.
    r = rnd.reset_index(drop=True)
    for i in range(len(r) - 1):
        if r["mean"].iloc[i] > r["mean"].iloc[i + 1]:
            ax.annotate("control non-monotone here",
                        xy=(r["epsilon"].iloc[i], r["mean"].iloc[i]),
                        xytext=(0, -18), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color=ctx.theme.ink2,
                        arrowprops=dict(arrowstyle="-", color=ctx.theme.axis,
                                        linewidth=0.8))
            break

    ax.set_yscale("log")
    eps_axis(ax, visible_epsilons(run, ctx), ctx)
    ax.set_xlabel(budget_label(ctx))
    ax.set_ylabel("EPE above clean (px)")
    ax.legend(loc="upper left")
    end_labels(ax, tips, ctx)
    titled(fig, "Damage above the clean baseline, gradient vs chance",
           f"Log y: the vertical gap is the multiple of the control's damage the "
           f"gradient buys. Clean EPE = {cmean:.3f} px. {sample_note(run)}"
           + (f". Off-scale (no damage over clean): {', '.join(dropped)}"
              if dropped else ""), ctx)
    return save(fig, ctx, "attack_vs_random_delta")


@figure("objective_transfer",
        "Does matching the attack objective to the metric matter? (gain vs epsilon)")
def fig_objective_transfer(run, ctx):
    """How much damage is lost by attacking with the *other* objective.

    Each objective is measured under every metric, so for the pair (epe,
    angular) there is a matched and a mismatched number for each metric. The
    ratio of their damage above clean is the only thing that answers "does the
    objective choice matter" -- comparing raw error levels hides it, because
    both objectives move the metric a long way at large epsilon.
    """
    pairs = [("EPE", "epe"), ("angular_deg", "angular")]
    pairs = [(m, o) for m, o in pairs if o in run.objectives]
    if len(run.objectives) < 2 or not pairs:
        raise Skip("needs >= 2 attack objectives measured under both metrics")

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    style_axes(ax)

    tips = []
    for metric, matched in pairs:
        base, _ = clean_value(run, metric)
        others = [o for o in run.objectives if o != matched]
        if not others:
            continue
        m_curve = curve(run, run.attack, metric, objective=matched).set_index("epsilon")
        xs, ys = [], []
        for eps in visible_epsilons(run, ctx):
            if eps not in m_curve.index:
                continue
            d_match = float(m_curve["mean"].loc[eps]) - base
            worst = None
            for o in others:
                c = curve(run, run.attack, metric, objective=o).set_index("epsilon")
                if eps in c.index:
                    d = float(c["mean"].loc[eps]) - base
                    worst = d if worst is None else max(worst, d)
            if worst is None or worst <= 0:
                continue
            xs.append(eps)
            ys.append(100.0 * (d_match / worst - 1.0))
        if not xs:
            continue
        color = objective_color(ctx.theme, matched)
        ax.plot(xs, ys, color=color, marker="o",
                markeredgecolor=ctx.theme.surface, markeredgewidth=1.0, zorder=4,
                label=f"{METRICS[metric][2]} (matched: {matched})")
        tips.append((xs[-1], ys[-1], METRICS[metric][2], color))

    ax.axhline(0.0, color=ctx.theme.ink2, linewidth=1.4, zorder=3,
               label="objectives interchangeable")
    eps_axis(ax, visible_epsilons(run, ctx), ctx)
    ax.set_xlabel(budget_label(ctx))
    ax.set_ylabel("extra damage from the matched objective (%)")
    ax.legend(loc="upper right")
    end_labels(ax, tips, ctx)
    titled(fig, "Does the attack objective matter?", ctx)
    return save(fig, ctx, "objective_transfer")


@figure("gradient_advantage", "Bar: adv EPE - random EPE per epsilon (gradient advantage)")
def fig_gradient_advantage(run, ctx):
    if not run.has_random:
        raise Skip("no random-sign control in this run")
    eps = [e for e in run.epsilons if e > 0]
    objs = run.objectives
    if not eps or not objs:
        raise Skip("no non-zero epsilons")

    rnd = curve(run, "random", "EPE").set_index("epsilon")["mean"]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    style_axes(ax)

    x = np.arange(len(eps), dtype=float)
    # 2px surface gap between adjacent bars, no borders.
    width = 0.8 / len(objs)
    for k, obj in enumerate(objs):
        adv = curve(run, run.attack, "EPE", objective=obj).set_index("epsilon")["mean"]
        vals = [adv.get(e, np.nan) - rnd.get(e, np.nan) for e in eps]
        ax.bar(x + (k - (len(objs) - 1) / 2) * width, vals, width * 0.94,
               color=objective_color(ctx.theme, obj), label=obj, zorder=3)
    ax.axhline(0, color=ctx.theme.axis, linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_budget(e, ctx) for e in eps], rotation=45, ha="right")
    ax.set_xlabel(budget_label(ctx, short=True))
    ax.set_ylabel("adv EPE - random EPE (px)")
    ax.legend(loc="upper left", title="attack objective", title_fontsize=8)
    titled(fig, "Gradient advantage over a same-magnitude random sign",
           f"Positive = the gradient direction beats chance. {sample_note(run)}", ctx)
    return save(fig, ctx, "gradient_advantage")


@figure("robustness_ratio", "Normalised degradation: adv EPE / clean EPE vs epsilon")
def fig_robustness_ratio(run, ctx):
    cmean, _ = clean_value(run, "EPE")
    if not np.isfinite(cmean) or cmean <= 0:
        raise Skip("no clean baseline")
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    style_axes(ax)

    tips = []
    for obj in run.objectives:
        df = on_axis(curve(run, run.attack, "EPE", objective=obj), ctx).copy()
        if df.empty:
            continue
        df["mean"] = df["mean"] / cmean
        df["std"] = df["std"] / cmean
        color = objective_color(ctx.theme, obj)
        draw_curve(ax, df, color=color, label=f"{run.attack.upper()} ({obj})", ctx=ctx)
        tips.append((df["epsilon"].iloc[-1], df["mean"].iloc[-1], obj, color))
    if run.has_random:
        rnd = on_axis(curve(run, "random", "EPE"), ctx).copy()
        rnd["mean"] /= cmean
        rnd["std"] /= cmean
        draw_curve(ax, rnd, color=ctx.theme.muted, label="random sign", ctx=ctx,
                   marker="s")
        if not rnd.empty:
            tips.append((rnd["epsilon"].iloc[-1], rnd["mean"].iloc[-1], "random",
                         ctx.theme.muted))
    draw_reference(ax, 1.0, color=ctx.theme.ink2, label="clean (= 1.0)", ctx=ctx)

    eps_axis(ax, visible_epsilons(run, ctx), ctx)
    ax.set_xlabel(budget_label(ctx, short=True))
    ax.set_ylabel("EPE / clean EPE")
    ax.legend(loc="upper left")
    end_labels(ax, tips, ctx)
    titled(fig, "Robustness ratio", "Degradation normalised by each run's clean "
           "error, so sequences and metrics are comparable", ctx)
    return save(fig, ctx, "robustness_ratio")


@figure("robustness_ratio_per_sequence",
        "Robustness ratio per sequence (bold average over faint per-sequence lines)")
def fig_robustness_ratio_overlay(run, ctx):
    seqs = run.sequences
    fig, axes = grid_axes(len(run.objectives), ncols=min(3, len(run.objectives)),
                          panel=(3.4, 2.7))
    for ax, obj in zip(axes, run.objectives):
        style_axes(ax)
        color = objective_color(ctx.theme, obj)
        for seq in seqs:                                   # faint per-sequence
            base, _ = clean_value(run, "EPE", sequence=seq)
            df = on_axis(curve(run, run.attack, "EPE", objective=obj,
                               sequence=seq), ctx)
            if df.empty or not np.isfinite(base) or base <= 0:
                continue
            ax.plot(df["epsilon"], df["mean"] / base, color=color, alpha=0.30,
                    linewidth=1.2, zorder=2,
                    label="per sequence" if seq == seqs[0] else None)
        base, _ = clean_value(run, "EPE")                  # bold micro-average
        df = on_axis(curve(run, run.attack, "EPE", objective=obj), ctx).copy()
        df["mean"] /= base
        df["std"] /= base
        draw_curve(ax, df, color=color, label="micro-average", ctx=ctx, zorder=4,
                   band=False)
        ax.axhline(1.0, color=ctx.theme.ink2, linewidth=1.0, zorder=1,
                   label="clean (= 1.0)" if obj == run.objectives[0] else None)
        ax.set_title(OBJECTIVE_LABEL.get(obj, obj), color=ctx.theme.ink, fontsize=9)
        eps_axis(ax, visible_epsilons(run, ctx), ctx)

    axes[0].legend(loc="upper left")
    fig.supxlabel(budget_label(ctx, short=True), color=ctx.theme.ink2, fontsize=9)
    fig.supylabel("EPE / clean EPE", color=ctx.theme.ink2, fontsize=9)
    fig.tight_layout()
    titled(fig, "Robustness ratio -- spread across sequences",
           f"Faint lines: each of the {len(seqs)} sequence"
           f"{'s' if len(seqs) != 1 else ''}. Bold: sample-weighted average", ctx)
    return save(fig, ctx, "robustness_ratio_per_sequence")


# ------------------------------------------------------- 3. attribution ------

def _attr_profile(run: RunData, axis: str, *, absolute: bool,
                  sequence: str | None = None) -> pd.DataFrame:
    """Mean gradient profile along ``axis`` ("time" or "polarity").

    Per-sample magnitudes span orders of magnitude, so by default each sample's
    profile is normalised to its own total (a *share* of |dJ/dE|) before
    averaging; ``absolute`` keeps the raw sums instead.
    """
    if run.attribution is None:
        raise Skip("no attribution_*.csv in this run")
    df = run.attribution[run.attribution["axis"] == axis].copy()
    if sequence is not None:
        df = df[df["sequence"] == sequence]
    if df.empty:
        raise Skip(f"no '{axis}' attribution rows")
    if not absolute:
        totals = df.groupby(["sample_index", "objective"])["grad_sum"].transform("sum")
        df["value"] = 100.0 * df["grad_sum"] / totals.replace(0, np.nan)
    else:
        df["value"] = df["grad_sum"]
    out = (df.groupby(["objective", "index"])["value"]
             .agg(mean="mean", std=lambda s: float(s.std(ddof=0)), n="size")
             .reset_index())
    return out


def _attr_ylabel(absolute: bool, axis: str) -> str:
    if absolute:
        return "mean $|\\partial J/\\partial E|$"
    return f"share of $|\\partial J/\\partial E|$ (%)"


def mass_window(values: np.ndarray, frac: float = 0.75) -> tuple[int, int, float]:
    """Shortest contiguous index window holding at least ``frac`` of the mass.
    """
    v = np.asarray(values, dtype=float)
    total = v.sum()
    if total <= 0:
        return 0, len(v) - 1, 0.0
    target = frac * total
    best = (0, len(v) - 1, 1.0)
    for lo in range(len(v)):
        run_sum = 0.0
        for hi in range(lo, len(v)):
            run_sum += v[hi]
            if run_sum >= target:
                if (hi - lo) < (best[1] - best[0]):
                    best = (lo, hi, run_sum / total)
                break
    return best[0], best[1], best[2]


def contiguous_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of each contiguous True run in ``flags``."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


@figure("attribution_temporal", "Bar: mean |grad| per temporal bin, averaged")
def fig_attr_temporal(run, ctx):
    """Single objective by design.
    """
    prof = _attr_profile(run, "time", absolute=ctx.attr_absolute)
    available = [o for o in run.objectives if o in set(prof["objective"])] or \
        sorted(prof["objective"].unique())
    obj = ctx.attr_objective if ctx.attr_objective in available else available[0]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    style_axes(ax)
    sub = prof[prof["objective"] == obj].sort_values("index")
    vals = sub["mean"].to_numpy(dtype=float)
    idx = sub["index"].to_numpy(dtype=int)

    # Reference: the profile the events themselves have.
    uniform = 100.0 / max(len(vals), 1) if not ctx.attr_absolute else None

    # Shade exactly the bins that beat the rule.
    if uniform is not None:
        marked = vals > uniform
        note = (f"{int(marked.sum())} of {len(vals)} bins carry more than a "
                f"uniform share: {vals[marked].sum():.0f}% of the total")
    else:
        lo, hi, share = mass_window(vals, 0.75)
        marked = np.zeros(len(vals), dtype=bool)
        marked[lo:hi + 1] = True
        note = f"bins {idx[lo]}-{idx[hi]} of {len(vals)} hold {share * 100:.0f}%"
    for lo, hi in contiguous_runs(marked):
        ax.axvspan(idx[lo] - 0.5, idx[hi] + 0.5, color=ctx.theme.grid,
                   linewidth=0, zorder=1)

    ax.bar(idx, vals, width=0.82, color=objective_color(ctx.theme, obj), zorder=3)
    ax.errorbar(idx, vals, yerr=sub["std"], fmt="none", ecolor=ctx.theme.muted,
                elinewidth=0.8, capsize=0, zorder=4)
    if uniform is not None:
        ax.axhline(uniform, color=ctx.theme.ink2, linewidth=1.4, zorder=5)
        ax.annotate("uniform over bins", xy=(idx[0] - 0.4, uniform),
                    xytext=(0, 3), textcoords="offset points", ha="left",
                    va="bottom", fontsize=8, color=ctx.theme.ink2)
    ax.annotate(note, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color=ctx.theme.ink2, annotation_clip=False)
    ax.set_xlabel("temporal bin  (0 = oldest)")
    ax.set_ylabel(_attr_ylabel(ctx.attr_absolute, "time"))
    fig.tight_layout()
    slide_title(fig, "Where the gradient concentrates in time", ctx)
    return save(fig, ctx, "attribution_temporal")


@figure("attribution_temporal_cumulative",
        "Cumulative share of |grad| over temporal bins vs the uniform diagonal")
def fig_attr_temporal_cumulative(run, ctx):
    prof = _attr_profile(run, "time", absolute=False)
    objs = [o for o in run.objectives if o in set(prof["objective"])] or \
        sorted(prof["objective"].unique())
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    style_axes(ax)

    tips, n_bins = [], 0
    for obj in objs:
        sub = prof[prof["objective"] == obj].sort_values("index")
        idx = sub["index"].to_numpy(dtype=int)
        cum = np.cumsum(sub["mean"].to_numpy(dtype=float))
        n_bins = max(n_bins, len(idx))
        color = objective_color(ctx.theme, obj)
        ax.plot(idx, cum, color=color, marker="o", markersize=4,
                markeredgecolor=ctx.theme.surface, markeredgewidth=1.0,
                label=OBJECTIVE_LABEL.get(obj, obj), zorder=4)
        tips.append((idx[-1], cum[-1], obj, color))
    if n_bins == 0:
        plt.close(fig)
        raise Skip("no temporal attribution rows")

    ref = np.arange(1, n_bins + 1) * 100.0 / n_bins
    ax.plot(np.arange(n_bins), ref, color=ctx.theme.ink2, linewidth=1.4,
            label="uniform over bins", zorder=3)
    ax.set_xlabel("temporal bin  (0 = oldest)")
    ax.set_ylabel("cumulative share of $|\\partial J/\\partial E|$ (%)")
    ax.set_xticks(np.arange(0, n_bins, max(1, n_bins // 10)))
    ax.legend(loc="upper left")
    end_labels(ax, tips, ctx)
    titled(fig, "The gradient is not spread evenly over the input window", ctx)
    return save(fig, ctx, "attribution_temporal_cumulative")


@figure("attribution_temporal_per_sequence",
        "Temporal |grad| profile, small multiples per sequence")
def fig_attr_temporal_seq(run, ctx):
    if run.attribution is None:
        raise Skip("no attribution_*.csv in this run")
    seqs = sorted(run.attribution["sequence"].dropna().unique().tolist())
    objs = run.objectives
    fig, axes = plt.subplots(len(seqs), len(objs), sharex=True, sharey="row",
                             squeeze=False,
                             figsize=(3.2 * len(objs), 2.1 * len(seqs)))
    for r, seq in enumerate(seqs):
        prof = _attr_profile(run, "time", absolute=ctx.attr_absolute, sequence=seq)
        for c, obj in enumerate(objs):
            ax = axes[r][c]
            style_axes(ax)
            sub = prof[prof["objective"] == obj].sort_values("index")
            ax.bar(sub["index"], sub["mean"], width=0.82,
                   color=objective_color(ctx.theme, obj), zorder=3)
            if r == 0:
                ax.set_title(OBJECTIVE_LABEL.get(obj, obj), color=ctx.theme.ink,
                             fontsize=9)
            if c == 0:
                ax.set_ylabel(seq, color=ctx.theme.ink2, fontsize=8)
            if r == len(seqs) - 1:
                ax.set_xlabel("temporal bin")
    fig.tight_layout()
    titled(fig, "Temporal gradient concentration, per sequence",
           _attr_ylabel(ctx.attr_absolute, "time") + " -- rows share a y-scale", ctx)
    return save(fig, ctx, "attribution_temporal_per_sequence")


@figure("attribution_polarity", "Bar: ON/OFF polarity vs mean |grad|, averaged")
def fig_attr_polarity(run, ctx):
    prof = _attr_profile(run, "polarity", absolute=ctx.attr_absolute)
    objs = [o for o in run.objectives if o in set(prof["objective"])] or \
        sorted(prof["objective"].unique())
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    style_axes(ax)

    pols = sorted(prof["index"].unique())
    x = np.arange(len(objs), dtype=float)
    width = 0.62 / max(len(pols), 1)
    for k, pol in enumerate(pols):
        vals, errs = [], []
        for obj in objs:
            row = prof[(prof["objective"] == obj) & (prof["index"] == pol)]
            vals.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
            errs.append(float(row["std"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (k - (len(pols) - 1) / 2) * width, vals, width * 0.94,
               color=ctx.theme.series[k % len(ctx.theme.series)],
               label=POLARITY_LABEL.get(int(pol), f"ch {pol}"), zorder=3)
        ax.errorbar(x + (k - (len(pols) - 1) / 2) * width, vals, yerr=errs,
                    fmt="none", ecolor=ctx.theme.muted, elinewidth=0.8, zorder=4)
    # Parity rule: the null this figure exists to test. Without it a reader has
    # to eyeball two near-equal bars and guess whether the gap means anything.
    gaps = []
    if not ctx.attr_absolute and pols:
        ax.axhline(100.0 / len(pols), color=ctx.theme.ink2, linewidth=1.4,
                   zorder=5, label="parity")
        for obj in objs:
            sub = prof[prof["objective"] == obj].set_index("index")["mean"]
            if len(sub) == 2:
                lead = POLARITY_LABEL.get(int(sub.idxmax()), "?")
                gaps.append(f"{obj}: {sub.max():.0f}/{sub.min():.0f} "
                            f"toward {lead}")

    ax.set_xticks(x)
    ax.set_xticklabels([OBJECTIVE_LABEL.get(o, o) for o in objs])
    ax.set_ylabel(_attr_ylabel(ctx.attr_absolute, "polarity"))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=len(pols) + 1,
              title="polarity", title_fontsize=8)
    titled(fig, "Gradient split across event polarity -- a small, consistent ON bias",
           f"ON = channel 0, OFF = channel 1"
           + (f". Split {'; '.join(gaps)}" if gaps else "")
           + f". The bias is well inside the per-sample spread (bars: 1 sd), so "
           f"neither channel is a materially softer target than the other. "
           f"{sample_note(run)}", ctx)
    return save(fig, ctx, "attribution_polarity")


@figure("attribution_polarity_per_sequence",
        "ON/OFF |grad| split, small multiples per sequence")
def fig_attr_polarity_seq(run, ctx):
    if run.attribution is None:
        raise Skip("no attribution_*.csv in this run")
    seqs = sorted(run.attribution["sequence"].dropna().unique().tolist())
    objs = run.objectives
    fig, axes = grid_axes(len(seqs), panel=(3.0, 2.4))
    pols = None
    for ax, seq in zip(axes, seqs):
        style_axes(ax)
        prof = _attr_profile(run, "polarity", absolute=ctx.attr_absolute, sequence=seq)
        pols = sorted(prof["index"].unique())
        x = np.arange(len(objs), dtype=float)
        width = 0.8 / max(len(pols), 1)
        for k, pol in enumerate(pols):
            vals = [float(prof[(prof["objective"] == o) & (prof["index"] == pol)]
                          ["mean"].iloc[0])
                    if len(prof[(prof["objective"] == o) & (prof["index"] == pol)])
                    else np.nan for o in objs]
            ax.bar(x + (k - (len(pols) - 1) / 2) * width, vals, width * 0.94,
                   color=ctx.theme.series[k % len(ctx.theme.series)],
                   label=POLARITY_LABEL.get(int(pol), f"ch {pol}"), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(objs, rotation=20, ha="right")
        ax.set_title(seq, color=ctx.theme.ink, fontsize=9)
    fig.supylabel(_attr_ylabel(ctx.attr_absolute, "polarity"), color=ctx.theme.ink2,
                  fontsize=9)
    fig.tight_layout()
    panel_legend(fig, axes[0], ctx)
    titled(fig, "Polarity split, per sequence", "Shared axes across panels", ctx)
    return save(fig, ctx, "attribution_polarity_per_sequence")


def _gradmap_grid(run, ctx: Ctx, *, shared: bool):
    """Build the sequence x objective heatmap grid.
    """
    if not run.gradmaps:
        raise Skip("no gradmap_*.npy in this run")
    seqs = sorted({s for _, s in run.gradmaps})
    objs = [o for o in run.objectives if any(o == k[0] for k in run.gradmaps)] or \
        sorted({o for o, _ in run.gradmaps})
    cmap = sequential_cmap(ctx.theme)

    def clip_max(arrays):
        return float(np.percentile(np.concatenate([a.ravel() for a in arrays]),
                                   ctx.gradmap_clip))

    vmax = clip_max([m for (o, _), m in run.gradmaps.items() if o in objs])

    fig, axes = plt.subplots(len(seqs), len(objs), squeeze=False,
                             figsize=(3.1 * len(objs) + 0.9, 2.3 * len(seqs)),
                             layout="constrained")
    im = None
    for r, seq in enumerate(seqs):
        maps = [run.gradmaps.get((o, seq)) for o in objs]
        present = [m for m in maps if m is not None]
        if not present:
            continue
        row_max = vmax if shared else clip_max(present)
        for c, (obj, m) in enumerate(zip(objs, maps)):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(ctx.theme.axis)
                side.set_linewidth(0.8)
            if m is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(m, cmap=cmap, interpolation="nearest", aspect="equal",
                           vmin=0.0, vmax=row_max)
            if r == 0:
                ax.set_title(OBJECTIVE_LABEL.get(obj, obj), color=ctx.theme.ink,
                             fontsize=9)
            if c == 0:
                ax.set_ylabel(seq, color=ctx.theme.ink2, fontsize=8)
        if not shared and im is not None:
            _gradmap_colorbar(fig, im, list(axes[r]), ctx)
    if shared and im is not None:
        _gradmap_colorbar(fig, im, axes.ravel().tolist(), ctx)
    return fig


@figure("gradmap_spatial",
        "Heatmap grid: where per-pixel |grad| concentrates (each sequence scaled "
        "to itself)")
def fig_gradmaps(run, ctx):
    fig = _gradmap_grid(run, ctx, shared=False)
    titled(fig, "Where the gradient concentrates in space",
           f"Sample-mean per-pixel gradient magnitude. Each sequence is scaled "
           f"to its own {ctx.gradmap_clip:g}th percentile, so structure is "
           f"legible in every scene but intensity does not compare between them",
           ctx)
    return save(fig, ctx, "gradmap_spatial")


@figure("gradmap_spatial_shared",
        "Same grid on one shared colour scale: how big the gradient is between "
        "sequences and objectives")
def fig_gradmaps_shared(run, ctx):
    fig = _gradmap_grid(run, ctx, shared=True)
    titled(fig, "How large the gradient is, on one shared scale",
           f"Identical colour bounds on every panel ({ctx.gradmap_clip:g}th "
           f"percentile over all of them), so intensity compares directly "
           f"across sequences and objectives -- scenes whose gradients are "
           f"orders of magnitude smaller stay near-blank, which is the result",
           ctx)
    return save(fig, ctx, "gradmap_spatial_shared")


def _gradmap_colorbar(fig, im, axs, ctx: Ctx):
    cb = fig.colorbar(im, ax=axs, fraction=0.024, pad=0.012)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=ctx.theme.muted, labelcolor=ctx.theme.muted,
                      labelsize=7)
    cb.set_label("mean $|\\partial J/\\partial E|$ per pixel",
                 color=ctx.theme.ink2, fontsize=7)
    return cb


@figure("gradmap_average", "Heatmap: per-pixel |grad| averaged over sequences (summary)")
def fig_gradmaps_avg(run, ctx):
    if not run.gradmaps:
        raise Skip("no gradmap_*.npy in this run")
    # Objectives this sweep actually ran; a directory can still hold gradmaps
    # from an earlier sweep, and averaging those over their own (smaller) set of
    # sequences beside the current ones would put unlike panels side by side.
    objs = [o for o in run.objectives if any(o == k[0] for k in run.gradmaps)] or \
        sorted({o for o, _ in run.gradmaps}, key=lambda o: OBJECTIVE_SLOT.get(o, 99))
    maps = {o: [m for (oo, _), m in run.gradmaps.items() if oo == o] for o in objs}
    shapes = {m.shape for ms in maps.values() for m in ms}
    if len(shapes) != 1:
        raise Skip(f"gradmaps have mixed shapes {shapes}")
    counts = {len(ms) for ms in maps.values()}
    cmap = sequential_cmap(ctx.theme)

    fig, axes = plt.subplots(1, len(objs), squeeze=False,
                             figsize=(3.1 * len(objs) + 0.9, 2.7),
                             layout="constrained")
    stacks = {o: np.mean(ms, axis=0) for o, ms in maps.items()}
    vmax = float(np.percentile(np.concatenate([m.ravel() for m in stacks.values()]),
                               ctx.gradmap_clip))
    im = None
    for c, obj in enumerate(objs):
        ax = axes[0][c]
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(ctx.theme.axis)
            side.set_linewidth(0.8)
        im = ax.imshow(stacks[obj], cmap=cmap, vmin=0.0, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        ax.set_title(OBJECTIVE_LABEL.get(obj, obj), color=ctx.theme.ink, fontsize=9)
    cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.024, pad=0.012)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=ctx.theme.muted, labelcolor=ctx.theme.muted, labelsize=7)
    cb.set_label("mean $|\\partial J/\\partial E|$ per pixel", color=ctx.theme.ink2,
                 fontsize=7)
    span = (f"{min(counts)}-{max(counts)}" if len(counts) > 1 else f"{max(counts)}")
    titled(fig, "Spatial gradient concentration -- averaged over sequences",
           f"Mean of {span} per-sequence maps per panel; scene structure is "
           "clearer in the per-sequence grid", ctx)
    return save(fig, ctx, "gradmap_average")


def _scene_maps(ctx: Ctx, sequence: str, shape: tuple) -> tuple | None:
    """(event-density map, GT-valid fraction map) for one sequence, or None.
    """
    if not ctx.data_root:
        return None
    ev_dir = os.path.join(ctx.data_root, "event_tensors", "11frames")
    mask_dir = os.path.join(ctx.data_root, "mask_tensors")
    ev_files = sorted(glob.glob(os.path.join(ev_dir, f"{sequence}_*.npy")))
    if not ev_files:
        return None
    ev_files = ev_files[:max(ctx.activity_samples, 1)]

    density = np.zeros(shape, dtype=np.float64)
    for path in ev_files:
        arr = np.load(path)                       # [T, C, H, W]
        if arr.shape[-2:] != shape:
            return None
        density += arr.sum(axis=(0, 1))
    density /= len(ev_files)

    valid = None
    mask_files = [os.path.join(mask_dir, os.path.basename(p)) for p in ev_files]
    mask_files = [p for p in mask_files if os.path.isfile(p)]
    if mask_files:
        acc = np.zeros(shape, dtype=np.float64)
        for path in mask_files:
            m = np.load(path).astype(np.float64).squeeze()
            if m.shape != shape:
                acc = None
                break
            acc += m
        if acc is not None:
            valid = acc / len(mask_files)
    return density, valid


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without pulling in scipy just for one number."""
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


@figure("gradient_vs_activity",
        "Where the attack has leverage: |grad| by event-density decile, and by GT validity")
def fig_gradient_vs_activity(run, ctx):
    """Turns the heatmaps' visual claim into a number.
    """
    if not run.gradmaps:
        raise Skip("no gradmap_*.npy in this run")
    if not ctx.data_root:
        raise Skip("needs --data-root (event/mask tensors) to bin by activity")

    obj = run.objectives[0] if run.objectives else None
    if obj is None:
        raise Skip("no attack objective")

    grads, dens, valids, used = [], [], [], []
    for (o, seq), gmap in sorted(run.gradmaps.items()):
        if o != obj:
            continue
        scene = _scene_maps(ctx, seq, gmap.shape)
        if scene is None:
            continue
        density, valid = scene
        grads.append(gmap.ravel().astype(np.float64))
        dens.append(density.ravel())
        valids.append(valid.ravel() if valid is not None else None)
        used.append(seq)
    if not used:
        raise Skip(f"no sequence under {ctx.data_root} matched a gradmap")

    g = np.concatenate(grads)
    d = np.concatenate(dens)
    rho = _rank_corr(g, d)

    have_mask = all(v is not None for v in valids)
    ncols = 2 if have_mask else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols + 0.6, 3.9),
                             squeeze=False)
    axes = axes[0]
    color = objective_color(ctx.theme, obj)

    # -- panel 1: gradient by event-density decile -----------------------------
    ax = axes[0]
    style_axes(ax)
    # Binned by *rank*, not by quantile edges.
    order = np.argsort(d, kind="mergesort")
    which = np.empty(len(d), dtype=int)
    which[order] = np.minimum((np.arange(len(d)) * 10) // len(d), 9)
    means = np.array([g[which == k].mean() if np.any(which == k) else np.nan
                      for k in range(10)])
    ax.bar(np.arange(10), means, width=0.82, color=color, zorder=3)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f"{k + 1}" for k in range(10)])
    ax.set_xlabel("event-density decile  (1 = quietest pixels)")
    ax.set_ylabel("mean $|\\partial J/\\partial E|$ per pixel")
    lo, hi = means[0], means[-1]
    if np.isfinite(lo) and np.isfinite(hi) and hi > 0:
        ax.annotate(f"quietest / busiest = {lo / hi:.0f}x",
                    xy=(0.5, 0.97), xycoords="axes fraction", ha="center",
                    va="top", fontsize=8, color=ctx.theme.ink2)

    # -- panel 2: where the leverage sits relative to scored pixels ------------
    mask_note = ""
    if have_mask:
        v = np.concatenate(valids) > 0.5
        ax = axes[1]
        style_axes(ax)
        shares = [100.0 * g[v].sum() / g.sum(), 100.0 * g[~v].sum() / g.sum()]
        cover = [100.0 * v.mean(), 100.0 * (~v).mean()]
        x = np.arange(2, dtype=float)
        ax.bar(x - 0.19, shares, 0.36, color=color, zorder=3,
               label="share of total $|\\partial J/\\partial E|$")
        ax.bar(x + 0.19, cover, 0.36, color=ctx.theme.muted, zorder=3,
               label="share of pixels")
        ax.set_xticks(x)
        ax.set_xticklabels(["GT valid\n(scored by EPE)", "GT invalid\n(not scored)"])
        ax.set_ylabel("% of frame / of gradient mass")
        ax.legend(loc="upper left")
        mask_note = (f"; gradient mass tracks area rather than favouring either "
                     f"region ({shares[1]:.0f}% of it on the {cover[1]:.0f}% of "
                     f"pixels EPE never scores), so most of the attack's leverage "
                     f"reaches the scored pixels through spatial context")

    fig.tight_layout()
    slide_title(fig, "Where the attack has leverage", ctx)
    print(f"         gradient_vs_activity: rho(|grad|, event density) = {rho:+.2f}; "
          f"{len(used)} sequence(s) [{', '.join(used)}], "
          f"{ctx.activity_samples} samples each{mask_note}")
    return save(fig, ctx, "gradient_vs_activity")


# --------------------------------------------------- 4. per-sequence view ----

@figure("per_sequence_epe_bars",
        "Per-sequence EPE vs epsilon bars (mean +/- sd), small multiples")
def fig_per_sequence_bars(run, ctx):
    seqs = run.sequences
    eps = run.epsilons
    objs = run.objectives
    if not eps:
        raise Skip("no epsilons swept")

    fig, axes = grid_axes(len(seqs), panel=(3.4, 2.6))
    x = np.arange(len(eps), dtype=float)
    width = 0.8 / max(len(objs), 1)
    for ax, seq in zip(axes, seqs):
        style_axes(ax)
        for k, obj in enumerate(objs):
            df = curve(run, run.attack, "EPE", objective=obj,
                       sequence=seq).set_index("epsilon")
            vals = [df["mean"].get(e, np.nan) for e in eps]
            errs = [df["std"].get(e, np.nan) for e in eps]
            pos = x + (k - (len(objs) - 1) / 2) * width
            ax.bar(pos, vals, width * 0.94, color=objective_color(ctx.theme, obj),
                   label=obj, zorder=3)
            ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=ctx.theme.muted,
                        elinewidth=0.8, zorder=4)
        cmean, _ = clean_value(run, "EPE", sequence=seq)
        ax.axhline(cmean, color=ctx.theme.ink2, linewidth=1.2, zorder=5,
                   label="clean")
        n = int(select(run, "clean", sequence=seq)["n_points"].sum())
        ax.set_title(f"{seq}  ({n} samples)", color=ctx.theme.ink, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_budget(e, ctx) for e in eps], rotation=45, ha="right")

    fig.supxlabel(budget_label(ctx, short=True), color=ctx.theme.ink2, fontsize=9)
    fig.supylabel(METRICS["EPE"][1], color=ctx.theme.ink2, fontsize=9)
    fig.tight_layout()
    panel_legend(fig, axes[0], ctx)
    titled(fig, "Which sequences are most vulnerable",
           "Shared axes; error bars are +/-1 sd across that sequence's samples; "
           "the rule is its clean EPE", ctx)
    return save(fig, ctx, "per_sequence_epe_bars")


@figure("per_sequence_ranking",
        "Ranked bars: EPE degradation per sequence at the largest epsilon")
def fig_per_sequence_ranking(run, ctx):
    eps = run.epsilons
    if not eps:
        raise Skip("no epsilons swept")
    if len(run.sequences) < 3:
        # One or two bars is not a ranking; per_sequence_epe_bars already
        # carries those numbers.
        raise Skip(f"needs >= 3 sequences (found {len(run.sequences)})")
    top = max(eps)
    obj = run.objectives[0] if run.objectives else None
    if obj is None:
        raise Skip("no attack objective")

    rows = []
    for seq in run.sequences:
        adv = curve(run, run.attack, "EPE", objective=obj,
                    sequence=seq).set_index("epsilon")
        base, _ = clean_value(run, "EPE", sequence=seq)
        if top not in adv.index or not np.isfinite(base) or base <= 0:
            continue
        rows.append({"sequence": seq, "clean": base,
                     "adv": float(adv["mean"].loc[top]),
                     "ratio": float(adv["mean"].loc[top]) / base,
                     "n": int(select(run, "clean", sequence=seq)["n_points"].sum())})
    if not rows:
        raise Skip("no per-sequence rows at the top epsilon")
    df = pd.DataFrame(rows).sort_values("ratio")

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(df) + 2.2))
    style_axes(ax, xgrid=True)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    y = np.arange(len(df), dtype=float)
    ax.barh(y, df["ratio"], height=0.5, color=objective_color(ctx.theme, obj),
            zorder=3)
    ax.axvline(1.0, color=ctx.theme.ink2, linewidth=1.2, zorder=4)
    ax.annotate("clean (= 1.0)", xy=(1.0, len(df) - 0.4), xytext=(4, 0),
                textcoords="offset points", color=ctx.theme.ink2, fontsize=8,
                va="bottom", ha="left", annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  ({n})" for s, n in zip(df["sequence"], df["n"])])
    for yi, r in zip(y, df["ratio"]):                    # value at the bar end
        ax.annotate(f"{r:.2f}x", xy=(r, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=ctx.theme.ink2)
    ax.set_xlabel("EPE / clean EPE")
    titled(fig, f"Sequence fragility at a {fmt_budget(top, ctx)}"
                f"{'% event budget' if ctx.eps_units == 'percent' and ctx.eps_pct else ' epsilon budget'}",
           f"{run.attack.upper()} ({obj}); sample count in parentheses. Read "
           f"beside fragility_vs_clean -- fragility here tracks how good the "
           f"clean prediction was, so scene content is confounded", ctx)
    return save(fig, ctx, "per_sequence_ranking")


@figure("fragility_vs_clean",
        "Scatter: is a sequence 'robust' only because its clean error was already high?")
def fig_fragility_vs_clean(run, ctx):
    """Fragility against clean accuracy, per sequence.
    """
    eps = [e for e in run.epsilons if e > 0]
    obj = run.objectives[0] if run.objectives else None
    if not eps or obj is None:
        raise Skip("no attack objective / epsilons swept")
    if len(run.sequences) < 3:
        raise Skip(f"needs >= 3 sequences (found {len(run.sequences)})")
    top = max(eps)

    rows = []
    for seq in run.sequences:
        base, _ = clean_value(run, "EPE", sequence=seq)
        adv = curve(run, run.attack, "EPE", objective=obj, sequence=seq).set_index("epsilon")
        if top not in adv.index or not np.isfinite(base) or base <= 0:
            continue
        rows.append({"sequence": seq, "clean": base,
                     "ratio": float(adv["mean"].loc[top]) / base,
                     "n": int(select(run, "clean", sequence=seq)["n_points"].sum())})
    if len(rows) < 3:
        raise Skip("fewer than 3 sequences at the top epsilon")
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    style_axes(ax, xgrid=True)
    color = objective_color(ctx.theme, obj)
    # Area encodes sample count -- the per-sequence n spans 40..964 here, and a
    # 40-sample point should not read as heavily as a 964-sample one.
    sizes = 30.0 + 170.0 * df["n"] / df["n"].max()
    ax.scatter(df["clean"], df["ratio"], s=sizes, color=color,
               edgecolor=ctx.theme.surface, linewidth=1.2, zorder=4)
    # Labels sit beside the markers, not above them.
    fig.canvas.draw()
    pts = [(ax.transData.transform((r["clean"], r["ratio"])) * 72.0 / fig.dpi, r)
           for _, r in df.iterrows()]
    for (px, py), r in pts:
        blocked = any(0 < qx - px < 130 and abs(qy - py) < 10
                      for (qx, qy), _ in pts)
        ax.annotate(f"{r['sequence']}  ({r['n']})", xy=(r["clean"], r["ratio"]),
                    xytext=(-12 if blocked else 12, 0), textcoords="offset points",
                    ha="right" if blocked else "left", va="center",
                    fontsize=8, color=ctx.theme.ink2, annotation_clip=False)
    ax.axhline(1.0, color=ctx.theme.ink2, linewidth=1.4, zorder=3)
    ax.annotate("clean (= 1.0)", xy=(ax.get_xlim()[0], 1.0), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=ctx.theme.ink2,
                va="bottom", ha="left")

    rho = float(np.corrcoef(df["clean"].rank(), df["ratio"].rank())[0, 1]) \
        if len(df) > 2 else float("nan")
    ax.set_xlabel("Clean EPE (px) -- how well the model did before the attack")
    ax.set_ylabel(f"EPE / clean EPE at {fmt_budget(top, ctx)}"
                  + ("% events injected" if ctx.eps_units == "percent" and ctx.eps_pct
                     else " epsilon"))
    titled(fig, "Fragility against clean accuracy",
           f"Marker area = sample count. Spearman rho = {rho:+.2f}: sequences the "
           f"model was already worse on degrade proportionally less, so any "
           f"per-sequence robustness claim is confounded with headroom", ctx)
    return save(fig, ctx, "fragility_vs_clean")


# =============================================================================
# Driver
# =============================================================================

def render(run: RunData, ctx: Ctx, only: Sequence[str] | None,
           skip: Sequence[str] | None) -> None:
    names = list(FIGURES)
    if only:
        unknown = [n for n in only if n not in FIGURES]
        if unknown:
            raise SystemExit(f"Unknown figure(s): {', '.join(unknown)} "
                             f"(see --list)")
        names = [n for n in names if n in set(only)]
    if skip:
        names = [n for n in names if n not in set(skip)]

    print(f"\n{run.attack.upper()} @ {run.root}")
    print(f"  {sample_note(run)} | objectives: {', '.join(run.objectives) or '-'} "
          f"| epsilons: {', '.join(f'{e:g}' for e in run.epsilons) or '-'}")
    for name in names:
        fn, _ = FIGURES[name]
        try:
            paths = fn(run, ctx)
        except Skip as exc:
            print(f"  [skip] {name:38} {exc}")
            continue
        print(f"  [ok]   {name:38} -> {', '.join(os.path.basename(p) for p in paths)}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", default=None,
                   help="Sweep directory/ies to plot, e.g. results/sweep_fgsm. "
                        "Required (except with --list); see --results-dir for "
                        "how to find them.")
    p.add_argument("--results-dir", default="results",
                   help="Results root (default: results). Only supplies the "
                        "defaults for --outdir and --calibration; runs are "
                        "never discovered from it.")
    p.add_argument("--outdir", default=None,
                   help="Figure output root (default: <results-dir>/figures).")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    p.add_argument("--formats", nargs="+", default=["png"],
                   help="Output formats, e.g. png pdf svg.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--band", choices=["std", "sem", "none"], default="std",
                   help="Shaded band around averaged curves (default: +/-1 sd).")
    p.add_argument("--xscale", choices=["log", "linear"], default="log",
                   help="Epsilon axis scale. The sweep is geometric, so log "
                        "(default) spaces it evenly; log drops the epsilon = 0 "
                        "point, which is the clean baseline already drawn as a rule.")
    p.add_argument("--attr-absolute", action="store_true",
                   help="Attribution bars in raw |dJ/dE| instead of per-sample share.")
    p.add_argument("--attr-objective", default=None,
                   help="Objective drawn by attribution_temporal, which is a "
                        "single-objective figure (the profiles coincide). "
                        "Default: the run's first objective.")
    p.add_argument("--gradmap-clip", type=float, default=99.5,
                   help="Percentile the heatmap colour scale is clipped at "
                        "(default 99.5: the top 0.5%% of pixels saturate, so a "
                        "single hot pixel cannot set the scale for the rest).")
    p.add_argument("--total-events", type=float, default=None,
                   help="Events per clean sample, for the %% -injected x-axis.")
    p.add_argument("--data-root", default=None,
                   help="Dataset root holding event_tensors/ and mask_tensors/ "
                        "(e.g. data/dataset/saved_flow_data). Only the "
                        "gradient_vs_activity figure needs it; sequences whose "
                        "tensors are absent are dropped from that figure.")
    p.add_argument("--eps-units", choices=["percent", "epsilon"], default="percent",
                   help="Unit for the budget axis. 'percent' (default) labels it "
                        "as the %% of the clean event count the attack injects, "
                        "which is what the epsilons were calibrated to and is "
                        "readable without knowing the tensor layout; it needs the "
                        "calibration record (see --calibration) and falls back to "
                        "raw epsilon without it.")
    p.add_argument("--activity-samples", type=int, default=40,
                   help="Event tensors read per sequence for gradient_vs_activity "
                        "(default 40). Each is ~27 MB, so this bounds the read.")
    p.add_argument("--calibration", default=None,
                   help="epsilon_calibration.json, used to estimate --total-events "
                        "(default: <results-dir>/epsilon_calibration.json if present).")
    p.add_argument("--only", nargs="+", default=None, help="Only these figures.")
    p.add_argument("--skip", nargs="+", default=None, help="Skip these figures.")
    p.add_argument("--list", action="store_true", help="List figures and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        width = max(len(n) for n in FIGURES)
        for name, (_, desc) in FIGURES.items():
            print(f"  {name:{width}}  {desc}")
        return

    if not args.run:
        # Not defaulted to "everything under results/": a full run is slow, so
        # the directory is named explicitly. Point at the candidates instead.
        found = discover_runs(args.results_dir)
        hint = ("\nSweep directories under "
                f"{args.results_dir!r}:\n  " + "\n  ".join(found)) if found else (
            f"\nNo sweep directories found under {args.results_dir!r} "
            f"(looked for per_sequence_*.csv / sweep_*.csv).")
        raise SystemExit("--run is required, e.g. "
                         "python plots.py --run results/sweep_fgsm" + hint)
    runs = args.run

    calibration = args.calibration
    if calibration is None:
        default = os.path.join(args.results_dir, "epsilon_calibration.json")
        calibration = default if os.path.isfile(default) else None

    out_root = args.outdir or os.path.join(args.results_dir, "figures")
    apply_theme(THEMES[args.theme])

    for run_dir in runs:
        try:
            run = load_run(run_dir, calibration_path=calibration)
        except FileNotFoundError as exc:
            raise SystemExit(f"{exc}\nSweep directories under "
                             f"{args.results_dir!r}: "
                             f"{', '.join(discover_runs(args.results_dir)) or 'none'}")
        ctx = Ctx(
            theme=THEMES[args.theme],
            outdir=os.path.join(out_root, os.path.basename(os.path.normpath(run_dir))),
            formats=tuple(args.formats),
            dpi=args.dpi,
            band=args.band,
            xscale=args.xscale,
            attr_absolute=args.attr_absolute,
            gradmap_clip=args.gradmap_clip,
            total_events=args.total_events,
            data_root=args.data_root,
            activity_samples=args.activity_samples,
            eps_units=args.eps_units,
            eps_pct=percent_map(run, run.calibration),
            attr_objective=args.attr_objective,
        )
        if args.eps_units == "percent" and not ctx.eps_pct:
            print(f"  [note] no epsilon->% mapping available "
                  f"({calibration or 'no calibration record'}); "
                  f"budget axis stays in raw epsilon")
        render(run, ctx, args.only, args.skip)
        print(f"  figures in {ctx.outdir}")


if __name__ == "__main__":
    main()
