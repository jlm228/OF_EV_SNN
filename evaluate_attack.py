"""Evaluate the optical-flow SNN clean vs. under a cybersecurity threat.

Runs the network over the DSEC validation split twice per sample -- once on the
clean event tensor and once on the adversarial tensor produced by a selected
threat -- and reports flow-error metrics for both, plus the degradation.

Examples
--------
Clean baseline::

    python evaluate_attack.py --attack none

White-box FGSM / PGD on the raw event-count tensor (see
``attacks/calibrate_epsilon.py`` for choosing ``--epsilon``)::

    python evaluate_attack.py --attack fgsm --epsilon 2.0
    python evaluate_attack.py --attack pgd --epsilon 2.0 --alpha 0.5 --iters 7

The threat is selected by name from the modular ``attacks`` registry, so new
threats are usable here without changing this file.
"""

import argparse
import csv
import math
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from spikingjelly.clock_driven import functional

from eval.vector_loss_functions import (
    mod_loss_function,
    angular_loss_function,
    cosine_loss_function,
)
from attacks.cli_common import (
    add_common_attack_args,
    add_common_model_args,
    build_threat_from_args,
    load_model_and_data,
)

RAD2DEG = 180.0 / math.pi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_attack_args(p)
    add_common_model_args(p)
    p.add_argument("--visualize", action="store_true",
                   help="Also write clean/attacked flow videos to results/.")
    p.add_argument("--fps", type=int, default=10, help="Video frame rate.")
    p.add_argument("--per-sample", action="store_true",
                   help="Also write per-sample and per-sequence-summary CSVs to results/.")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def load_sample_names(args):
    """Ordered (sequence, filename) for each loader sample, from the split CSV.

    ``DSECDatasetLite`` iterates the split rows in order (``shuffle=False``) and takes each
    sample's label/M from the *second* file in the row, so that column is the canonical
    per-sample identity. Reconstructing here avoids changing the dataset's return signature.
    """
    split_path = os.path.join(args.root, "sequence_lists", args.split)
    rows = pd.read_csv(split_path, header=None)
    names = rows.iloc[:, 1].tolist()
    seqs = [re.sub(r"_\d+\.npy$", "", nm) for nm in names]
    return seqs, names


@torch.no_grad()
def predict(net, E):
    """Reset state and return the finest flow field pred_1, shape [B, 2, H, W]."""
    functional.reset_net(net)
    return net(E)[-1]


def metrics(pred, label, M):
    return (
        mod_loss_function(pred, label, M).item(),               # magnitude / EPE proxy
        angular_loss_function(pred, label, M).item() * RAD2DEG,  # angular error (deg)
        cosine_loss_function(pred, label, M).item(),            # 1 - cos(pred, label)
    )


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Creating validation dataset ...")
    device, net, dataset, loader = load_model_and_data(args)
    print(f"Device: {device}")

    threat = build_threat_from_args(args)
    print(f"Threat: {threat}")

    # Metric accumulators (clean, attacked).
    keys = ["EPE", "angular_deg", "one_minus_cos"]
    clean_sum = {k: 0.0 for k in keys}
    adv_sum = {k: 0.0 for k in keys}
    n = 0
    max_count_drift = 0.0

    # Per-sample identity + rows, only when requested.
    seqs = names = None
    if args.per_sample:
        seqs, names = load_sample_names(args)
    per_sample = []

    # Sequences for optional visualisation.
    label_seq, mask_seq, pred_clean_seq, pred_adv_seq = [], [], [], []

    for E, M, label in tqdm(loader, desc="Evaluating"):
        E = torch.transpose(E, 1, 2)              # [B, C, T, H, W]
        M = torch.unsqueeze(M, dim=1)               # [B, 1, H, W] (pixels with valid GT flow)
        E = E.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)  # [B, 2, H, W]
        M = M.to(device=device)

        # Clean prediction.
        pred_clean = predict(net, E)

        # Adversarial event tensor + prediction.
        E_adv = threat(E, model=net, label=label, M=M)
        max_count_drift = max(max_count_drift,
                              (E_adv.sum() - E.sum()).abs().item())
        pred_adv = predict(net, E_adv)

        cm = metrics(pred_clean, label, M)
        am = metrics(pred_adv, label, M)
        for k, cv, av in zip(keys, cm, am):
            clean_sum[k] += cv
            adv_sum[k] += av

        if args.per_sample:
            seq = seqs[n] if n < len(seqs) else ""
            fname = names[n] if n < len(names) else ""
            per_sample.append([n, seq, fname, cm[0], cm[1], cm[2], am[0], am[1], am[2]])

        n += 1

        if args.visualize:
            label_seq.append(torch.squeeze(label[0]).cpu().numpy())
            pred_clean_seq.append(torch.squeeze(pred_clean[0]).cpu().numpy())
            pred_adv_seq.append(torch.squeeze(pred_adv[0]).cpu().numpy())
            mask_seq.append(torch.squeeze(M[0]).cpu().numpy())

        if args.max_chunks is not None and n >= args.max_chunks:
            break

    if n == 0:
        raise RuntimeError("No samples were evaluated; check --root / --split.")

    clean_avg = {k: clean_sum[k] / n for k in keys}
    adv_avg = {k: adv_sum[k] / n for k in keys}

    # ---- Report -----------------------------------------------------------
    print(f"\nEvaluated {n} samples | attack = '{args.attack}'")
    print(f"Count drift: max |sum(E_adv) - sum(clean)| = {max_count_drift:.6g} "
          f"(0 = perfectly rate-preserving; retiming_* attacks should be ~0, "
          f"a large value is *expected* for additive attacks like fgsm/pgd)\n")
    header = f"{'metric':<16}{'clean':>12}{'attacked':>12}{'delta':>12}"
    print(header)
    print("-" * len(header))
    for k in keys:
        delta = adv_avg[k] - clean_avg[k]
        print(f"{k:<16}{clean_avg[k]:>12.4f}{adv_avg[k]:>12.4f}{delta:>+12.4f}")

    # ---- CSV --------------------------------------------------------------
    csv_path = os.path.join(args.outdir, f"attack_eval_{args.attack}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "clean", "attacked", "delta"])
        for k in keys:
            w.writerow([k, clean_avg[k], adv_avg[k], adv_avg[k] - clean_avg[k]])
        w.writerow(["n_samples", n, n, 0])
        w.writerow(["max_count_drift", max_count_drift, max_count_drift, 0])
    print(f"\nMetrics written to {csv_path}")

    # ---- Per-sample + per-sequence CSVs -----------------------------------
    if args.per_sample:
        ps_cols = ["index", "sequence", "filename",
                   "clean_EPE", "clean_angular_deg", "clean_one_minus_cos",
                   "adv_EPE", "adv_angular_deg", "adv_one_minus_cos"]
        ps_path = os.path.join(args.outdir, f"per_sample_{args.attack}.csv")
        with open(ps_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(ps_cols)
            w.writerows(per_sample)
        print(f"Per-sample metrics ({len(per_sample)} rows) written to {ps_path}")

        # Group by sequence in first-appearance order; mean each metric, plus clean->E_adv delta.
        groups = OrderedDict()
        for row in per_sample:
            groups.setdefault(row[1], []).append(row)
        seq_path = os.path.join(args.outdir, f"per_sequence_{args.attack}.csv")
        with open(seq_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "n_samples",
                        "clean_EPE", "clean_angular_deg", "clean_one_minus_cos",
                        "adv_EPE", "adv_angular_deg", "adv_one_minus_cos",
                        "delta_EPE", "delta_angular_deg", "delta_one_minus_cos"])
            for seq, rows in groups.items():
                cnt = len(rows)
                ce, ca, cc, ae, aa, ac = (sum(r[c] for r in rows) / cnt for c in range(3, 9))
                w.writerow([seq, cnt, ce, ca, cc, ae, aa, ac,
                            ae - ce, aa - ca, ac - cc])
        print(f"Per-sequence summary ({len(groups)} sequences) written to {seq_path}")

    # ---- Optional videos --------------------------------------------------
    if args.visualize:
        from eval.progress_plot_full_v2 import plot_evolution, plot_gt_vs_predictions
        label_arr = np.array(label_seq)
        clean_path = os.path.join(args.outdir, "flow_clean.mp4")
        adv_path = os.path.join(args.outdir, f"flow_{args.attack}.mp4")
        plot_evolution(label_arr, np.array(pred_clean_seq), mask_seq, args.fps, clean_path)
        plot_evolution(label_arr, np.array(pred_adv_seq), mask_seq, args.fps, adv_path)
        print(f"Detailed videos (gt/pred/error) written to {clean_path} and {adv_path}")

        if args.attack != "none":
            gt_vs_pred_path = os.path.join(args.outdir, f"flow_gt_vs_{args.attack}.mp4")
            plot_gt_vs_predictions(
                label_arr,
                [np.array(pred_clean_seq), np.array(pred_adv_seq)],
                ["No Attack", f"Attacked ({args.attack})"],
                args.fps,
                gt_vs_pred_path,
            )
            print(f"Ground-truth vs. predictions video written to {gt_vs_pred_path}")


if __name__ == "__main__":
    main()
