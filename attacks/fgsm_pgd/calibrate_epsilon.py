"""Calibrate an L-infinity epsilon budget for the fgsm/pgd threats.

FGSM/PGD in this repo perturb raw event *counts* directly, not normalised
pixel intensities, so the usual ``eps/255`` convention doesn't transfer. This
script grounds ``epsilon`` in the data instead: it scans a sample of the
*training* split's event tensors, computes per-polarity-channel statistics of
the nonzero bin counts, and prints/saves a suggested epsilon table expressed
as small multiples of the mean nonzero count.

Usage::

    python -m attacks.fgsm_pgd.calibrate_epsilon --n-samples 200

Output is written to ``results/epsilon_calibration.json``.
"""

import argparse
import json
import os

import numpy as np
import torch

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/dataset/saved_flow_data")
    p.add_argument("--split", default="train_split_doubleseq.csv")
    p.add_argument("--num-frames-per-ts", type=int, default=11)
    p.add_argument("--n-samples", type=int, default=200,
                   help="Number of chunks to scan (a subset, chosen with --seed).")
    p.add_argument("--seed", type=int, default=2305)
    p.add_argument("--multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                   help="Suggested epsilon = multiplier * mean_nonzero_count.")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dataset = DSECDatasetLite(root=args.root, file_list=args.split,
                              num_frames_per_ts=args.num_frames_per_ts,
                              stereo=False, transform=None)

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)

    on_counts = []
    off_counts = []
    all_counts = []

    for idx in indices:
        chunk, _mask, _label = dataset[int(idx)]
        chunk = torch.as_tensor(chunk).float()  # [T, C=2, H, W]
        on = chunk[:, 0]
        off = chunk[:, 1]
        on_nz = on[on > 0]
        off_nz = off[off > 0]
        if on_nz.numel() > 0:
            on_counts.append(on_nz.numpy())
        if off_nz.numel() > 0:
            off_counts.append(off_nz.numpy())
        both = chunk[chunk > 0]
        if both.numel() > 0:
            all_counts.append(both.numpy())

    def _stats(arrs):
        if not arrs:
            return {}
        cat = np.concatenate(arrs)
        return {
            "mean": float(cat.mean()),
            "std": float(cat.std()),
            "p50": float(np.percentile(cat, 50)),
            "p90": float(np.percentile(cat, 90)),
            "p95": float(np.percentile(cat, 95)),
            "p99": float(np.percentile(cat, 99)),
            "max": float(cat.max()),
        }

    stats = {
        "on_polarity": _stats(on_counts),
        "off_polarity": _stats(off_counts),
        "combined": _stats(all_counts),
    }

    mean_nonzero = stats["combined"]["mean"]
    suggested_epsilon = {f"{m}x_mean": round(m * mean_nonzero, 4) for m in args.multipliers}

    result = {
        "root": args.root,
        "split": args.split,
        "n_samples_scanned": n,
        "count_statistics": stats,
        "suggested_epsilon": suggested_epsilon,
    }

    out_path = os.path.join(args.outdir, "epsilon_calibration.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scanned {n} chunks from '{args.split}'.")
    print(f"Nonzero event-count statistics (combined ON+OFF):")
    for k, v in stats["combined"].items():
        print(f"  {k:>6} = {v:.4f}")
    print(f"\nSuggested epsilon values (multiples of mean nonzero count = "
          f"{mean_nonzero:.4f}):")
    for k, v in suggested_epsilon.items():
        print(f"  {k:<12} epsilon = {v}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
