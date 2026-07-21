"""Evaluate the optical-flow SNN clean vs. under a cybersecurity threat.

Runs the network over the DSEC validation split twice per sample -- once on the
clean event tensor and once on the adversarial tensor produced by a selected
threat -- and reports flow-error metrics for both, plus the degradation.

Examples
--------
Clean baseline::

    python evaluate_attack.py --attack none

Black-box spike-retiming with qualitative videos::

    python evaluate_attack.py --attack retiming_blackbox --budget 2 --visualize

White-box projected-in-the-loop retiming::

    python evaluate_attack.py --attack retiming_pil --budget 2 --iters 10

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

import numpy as np
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
    p.add_argument("--outdir", default="results")
    return p.parse_args()


@torch.no_grad()
def predict(net, chunk):
    """Reset state and return the finest flow field pred_1, shape [B, 2, H, W]."""
    functional.reset_net(net)
    return net(chunk)[-1]


def metrics(pred, label, mask):
    return (
        mod_loss_function(pred, label, mask).item(),               # magnitude / EPE proxy
        angular_loss_function(pred, label, mask).item() * RAD2DEG,  # angular error (deg)
        cosine_loss_function(pred, label, mask).item(),            # 1 - cos(pred, label)
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

    # Sequences for optional visualisation.
    label_seq, mask_seq, pred_clean_seq, pred_adv_seq = [], [], [], []

    for chunk, mask, label in tqdm(loader, desc="Evaluating"):
        chunk = torch.transpose(chunk, 1, 2)              # [B, C, T, H, W]
        mask = torch.unsqueeze(mask, dim=1)               # [B, 1, H, W]
        chunk = chunk.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)  # [B, 2, H, W]
        mask = mask.to(device=device)

        # Clean prediction.
        pred_clean = predict(net, chunk)

        # Adversarial event tensor + prediction.
        adv = threat(chunk, model=net, label=label, mask=mask)
        max_count_drift = max(max_count_drift,
                              (adv.sum() - chunk.sum()).abs().item())
        pred_adv = predict(net, adv)

        cm = metrics(pred_clean, label, mask)
        am = metrics(pred_adv, label, mask)
        for k, cv, av in zip(keys, cm, am):
            clean_sum[k] += cv
            adv_sum[k] += av
        n += 1

        if args.visualize:
            label_seq.append(torch.squeeze(label[0]).cpu().numpy())
            pred_clean_seq.append(torch.squeeze(pred_clean[0]).cpu().numpy())
            pred_adv_seq.append(torch.squeeze(pred_adv[0]).cpu().numpy())
            mask_seq.append(torch.squeeze(mask[0]).cpu().numpy())

        if args.max_chunks is not None and n >= args.max_chunks:
            break

    if n == 0:
        raise RuntimeError("No samples were evaluated; check --root / --split.")

    clean_avg = {k: clean_sum[k] / n for k in keys}
    adv_avg = {k: adv_sum[k] / n for k in keys}

    # ---- Report -----------------------------------------------------------
    print(f"\nEvaluated {n} samples | attack = '{args.attack}'")
    print(f"Count drift: max |sum(adv) - sum(clean)| = {max_count_drift:.6g} "
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
