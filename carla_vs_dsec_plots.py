"""Figures comparing flow-prediction error on a CARLA capture against a DSEC sequence.

    python carla_vs_dsec_plots.py --capture 50761933 [--dsec thun_00_a] [--theme light|dark]

Reads the CSVs written by carla_epe_eval.py and draws:

  epe_vs_magnitude   error against |GT| flow magnitude for both datasets, which compares them
                     at matched motion instead of matched dataset. Raw averages conflate
                     accuracy with how fast things happened to be moving, and the two datasets
                     differ substantially on the latter.
  epe_distribution   spread of per-window error, CARLA against DSEC.
  epe_vs_state       error against ego speed and against range to the pedestrian, whole-image
                     and pedestrian-only, showing where in the scenario accuracy degrades.

Styling helpers come from plots.py so these match the rest of the figures.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots import (METRICS, THEMES, Ctx, apply_theme, draw_reference, grid_axes, save,
                   style_axes, titled)

EVAL_DIR = os.path.join("results", "carla_eval")
BASELINE = os.path.join("results", "valid-set-eval", "baseline_per_sequence.csv")
POOLED_BASELINE = os.path.join("results", "valid-set-eval", "baseline_metrics.csv")
# Mask name -> (label, categorical slot). "dense" and "dsec_matched" are the same model on the
# same frames scored over different pixel populations, so they must stay visually distinct.
MASK_STYLE = {"dense": ("CARLA, all non-sky pixels", 1),
              "dsec_matched": ("CARLA, DSEC-shaped region", 0),
              "ped": ("CARLA, pedestrian only", 2)}


def _mid(lo, hi):
    """Bin centre, with the open-ended top bin placed just past its lower edge."""
    hi = np.where(np.isfinite(hi), hi, lo * 1.5)
    return (lo + hi) / 2.0


def fig_epe_vs_magnitude(capture, dsec, ctx):
    cm = pd.read_csv(os.path.join(EVAL_DIR, f"{capture}_epe_vs_magnitude.csv"))
    dm = pd.read_csv(os.path.join(EVAL_DIR, f"dsec_{dsec}_epe_vs_magnitude.csv"))

    # Marker area tracks each bin's share of pixels. Without it the sparsely populated tail
    # bins -- a fraction of a percent of the image -- draw as confidently as bins carrying a
    # quarter of it, and the tail is where the curves diverge most.
    def draw(sub, color, label, marker):
        share = sub.n_pixels.to_numpy() / sub.n_pixels.sum()
        x = _mid(sub.bin_lo.to_numpy(), sub.bin_hi.to_numpy())
        ax.plot(x, sub.EPE, color=color, linewidth=1.2, zorder=2)
        ax.scatter(x, sub.EPE, s=12 + 320 * np.sqrt(share), color=color, marker=marker,
                   edgecolor=ctx.theme.surface, linewidth=1.0, zorder=3, label=label)

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for mask in ("dense", "dsec_matched"):
        sub = cm[cm["mask"] == mask]
        if not sub.empty:
            label, slot = MASK_STYLE[mask]
            draw(sub, ctx.theme.series[slot], label, "o")
    draw(dm[dm["mask"] == "dsec"], ctx.theme.ink2, f"DSEC {dsec}", "s")

    # y = x marks the point where error equals the motion being measured, i.e. where a
    # prediction stops being better than predicting no flow at all.
    lim = ax.get_xlim()
    xs = np.linspace(max(lim[0], 0.1), lim[1], 50)
    ax.plot(xs, xs, color=ctx.theme.muted, linewidth=1.0, linestyle=(0, (4, 3)),
            label="error = magnitude (no-flow control)", zorder=1)

    ax.set_xlabel("Ground-truth flow magnitude (px per 100 ms)")
    ax.set_ylabel(METRICS["EPE"][1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax, xgrid=True)
    leg = ax.legend(loc="upper left", fontsize=8)
    for h in leg.legend_handles:
        if hasattr(h, "set_sizes"):   # scatter swatches only; the reference line is a Line2D
            h.set_sizes([26])         # uniform in the key, since size encodes data in the plot
    titled(fig, "Error scales with the motion being measured",
           "Per-pixel error binned by ground-truth magnitude, same model and metric "
           "throughout. Marker area = share of pixels in that bin", ctx)
    return save(fig, ctx, "epe_vs_magnitude")


def fig_epe_distribution(capture, dsec, ctx):
    cw = pd.read_csv(os.path.join(EVAL_DIR, f"{capture}_per_window.csv"))
    dw = pd.read_csv(os.path.join(EVAL_DIR, f"dsec_{dsec}_per_window.csv"))

    series = [(f"DSEC {dsec}", dw["EPE_dsec"], ctx.theme.ink2)]
    for mask in ("dsec_matched", "dense"):
        col = f"EPE_{mask}"
        if col in cw:
            label, slot = MASK_STYLE[mask]
            series.append((label, cw[col], ctx.theme.series[slot]))

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    parts = ax.violinplot([s[1].dropna().to_numpy() for s in series],
                          showextrema=False, widths=0.7)
    for body, (_, _, color) in zip(parts["bodies"], series):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
    for i, (_, vals, color) in enumerate(series, start=1):
        v = vals.dropna().to_numpy()
        ax.scatter([i], [v.mean()], s=22, color=color, zorder=3,
                   edgecolor=ctx.theme.surface, linewidth=1.0)
    ax.set_xticks(range(1, len(series) + 1))
    ax.set_xticklabels([s[0].replace("CARLA, ", "CARLA\n") for s in series], fontsize=8)
    ax.set_ylabel(METRICS["EPE"][1])
    style_axes(ax)
    titled(fig, "Per-window error distribution",
           "Dots mark the mean; each CARLA column is the same predictions scored over a "
           "different pixel population", ctx)
    return save(fig, ctx, "epe_distribution")


def fig_epe_vs_state(capture, ctx):
    cw = pd.read_csv(os.path.join(EVAL_DIR, f"{capture}_per_window.csv"))

    # Ego speed is deliberately not an axis here: a single scenario holds one cruising speed,
    # so error against speed collapses to a vertical line. It only becomes informative across
    # captures recorded at different speeds.
    panels = [("window", "Window (100 ms each)"),
              ("dist_to_ped_m", "Range to pedestrian (m)")]
    panels = [p for p in panels if p[0] in cw]
    if not panels:
        return []
    # sharex must stay off, or inverting the range axis flips the other panel with it.
    fig, axes = grid_axes(len(panels), ncols=len(panels), panel=(3.3, 2.6),
                          sharex=False, sharey=True)

    for ax, (col, xlabel) in zip(axes, panels):
        for mask in ("dsec_matched", "ped"):
            ycol = f"EPE_{mask}"
            if ycol not in cw:
                continue
            label, slot = MASK_STYLE[mask]
            sub = cw[[col, ycol]].dropna().sort_values(col)
            ax.plot(sub[col], sub[ycol], marker="o", markersize=2.5, linewidth=1.0,
                    color=ctx.theme.series[slot], label=label,
                    markeredgecolor=ctx.theme.surface, markeredgewidth=0.6)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")     # the pedestrian series spans nearly two decades
        style_axes(ax, xgrid=True)
    axes[0].set_ylabel(METRICS["EPE"][1])
    if "dist_to_ped_m" in cw:
        axes[-1].invert_xaxis()      # so the approach reads left to right
    axes[0].legend(loc="upper left", fontsize=8)
    titled(fig, "Where in the scenario the error grows",
           "Pedestrian-region error against the wider region, as the vehicle closes in", ctx)
    return save(fig, ctx, "epe_vs_state")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="capture id, e.g. 50761933")
    ap.add_argument("--dsec", default="thun_00_a")
    ap.add_argument("--theme", default="light", choices=sorted(THEMES))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    theme = THEMES[args.theme]
    apply_theme(theme)
    outdir = args.outdir or os.path.join("results", "figures", f"carla_{args.capture}")
    ctx = Ctx(theme=theme, outdir=outdir, dpi=args.dpi)

    written = []
    for fn in (lambda: fig_epe_vs_magnitude(args.capture, args.dsec, ctx),
               lambda: fig_epe_distribution(args.capture, args.dsec, ctx),
               lambda: fig_epe_vs_state(args.capture, ctx)):
        try:
            written += fn()
        except FileNotFoundError as e:
            print("skipped a figure, missing input: %s" % e)
    for p in written:
        print("wrote %s" % p)


if __name__ == "__main__":
    main()
