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

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d
from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite
from eval.vector_loss_functions import (
    mod_loss_function,
    angular_loss_function,
    cosine_loss_function,
)
from attacks import build_attack

RAD2DEG = 180.0 / math.pi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attack", default="none",
                   help="Threat name (e.g. none, retiming_blackbox, retiming_pil).")
    p.add_argument("--budget", type=int, default=2, help="Max temporal shift in bins.")
    p.add_argument("--granularity", default=None,
                   help="pixel | block | global (default: threat's own default).")
    p.add_argument("--mode", default="random",
                   help="Black-box sampling mode: random | worst_of_n.")
    p.add_argument("--n-samples", type=int, default=8, help="Candidates for worst_of_n.")
    p.add_argument("--iters", type=int, default=10, help="PIL optimisation steps.")
    p.add_argument("--loss", default="angular", help="PIL objective: angular | cosine.")
    p.add_argument("--seed", type=int, default=2305, help="Attack sampling seed.")
    p.add_argument("--root", default="data/dataset/saved_flow_data",
                   help="Dataset root (relative path).")
    p.add_argument("--split", default="valid_split_thun_00_a.csv", help="Sequence list CSV.")
    p.add_argument("--num-frames-per-ts", type=int, default=11)
    p.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    p.add_argument("--multiply-factor", type=float, default=35.0)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Limit number of samples (useful on CPU).")
    p.add_argument("--visualize", action="store_true",
                   help="Also write clean/attacked flow videos to results/.")
    p.add_argument("--fps", type=int, default=10, help="Video frame rate.")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def build_threat(args):
    """Assemble threat kwargs, only overriding granularity when the user set it."""
    cfg = dict(budget=args.budget, mode=args.mode, n_samples=args.n_samples,
               iters=args.iters, loss=args.loss, seed=args.seed)
    if args.granularity is not None:
        cfg["granularity"] = args.granularity
    return build_attack(args.attack, **cfg)


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
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Device: {device}")
    print("Creating validation dataset ...")
    dataset = DSECDatasetLite(root=args.root, file_list=args.split,
                              num_frames_per_ts=args.num_frames_per_ts,
                              stereo=False, transform=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         drop_last=False, pin_memory=True)

    net = NeuronPool_Separable_Pool3d(multiply_factor=args.multiply_factor).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    threat = build_threat(args)
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
    print(f"Rate-preservation check: max |sum(adv) - sum(clean)| = "
          f"{max_count_drift:.6g} (0 = perfectly rate-preserving)\n")
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
