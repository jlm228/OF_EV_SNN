"""Compare attack degradation between two conditions (e.g. easy vs. hard scene).

Takes two ``attack_eval_<name>.csv`` files produced by ``evaluate_attack.py``
(same attack config, different ``--split``/``--outdir``) and prints a
side-by-side table of clean EPE, attacked EPE, and relative degradation
(%dEPE) for each condition, so the *only* varied axis -- scene difficulty --
is visible without touching the model, checkpoint, or attack parameters.

Usage::

    python evaluate_attack.py --attack pgd --epsilon 2.0 --alpha 0.5 --iters 7 \\
        --split test_instances/thun_00_a.csv --outdir results/easy
    python evaluate_attack.py --attack pgd --epsilon 2.0 --alpha 0.5 --iters 7 \\
        --split test_instances/zurich_city_02_a.csv --outdir results/hard
    python attacks/compare_easy_hard.py results/easy/attack_eval_pgd.csv \\
        results/hard/attack_eval_pgd.csv --labels easy hard
"""

import argparse
import csv


def load_metrics(path):
    metrics = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "metric":
                continue
            key = row[0]
            if key in ("n_samples", "max_count_drift"):
                continue
            metrics[key] = (float(row[1]), float(row[2]))  # (clean, attacked)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_a", help="attack_eval_*.csv for condition A (e.g. easy).")
    p.add_argument("csv_b", help="attack_eval_*.csv for condition B (e.g. hard).")
    p.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("LABEL_A", "LABEL_B"))
    return p.parse_args()


def main():
    args = parse_args()
    label_a, label_b = args.labels
    metrics_a = load_metrics(args.csv_a)
    metrics_b = load_metrics(args.csv_b)

    keys = [k for k in metrics_a if k in metrics_b]

    header = (f"{'metric':<16}"
              f"{label_a + '_clean':>14}{label_a + '_atk':>12}{label_a + '_%d':>10}"
              f"{label_b + '_clean':>14}{label_b + '_atk':>12}{label_b + '_%d':>10}")
    print(header)
    print("-" * len(header))

    for k in keys:
        ca, aa = metrics_a[k]
        cb, ab = metrics_b[k]
        pct_a = 100.0 * (aa - ca) / ca if ca != 0 else float("nan")
        pct_b = 100.0 * (ab - cb) / cb if cb != 0 else float("nan")
        print(f"{k:<16}{ca:>14.4f}{aa:>12.4f}{pct_a:>+9.1f}%"
              f"{cb:>14.4f}{ab:>12.4f}{pct_b:>+9.1f}%")

    if "EPE" in metrics_a and "EPE" in metrics_b:
        ca, aa = metrics_a["EPE"]
        cb, ab = metrics_b["EPE"]
        pct_a = 100.0 * (aa - ca) / ca if ca != 0 else float("nan")
        pct_b = 100.0 * (ab - cb) / cb if cb != 0 else float("nan")
        print(f"\n%dEPE under attack: {label_a} = {pct_a:+.1f}%, {label_b} = {pct_b:+.1f}%"
              f" (difference = {pct_b - pct_a:+.1f} points)")


if __name__ == "__main__":
    main()
