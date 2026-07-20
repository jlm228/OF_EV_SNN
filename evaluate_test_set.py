"""Evaluate the optical-flow SNN over a folder of test data instances.

An "instance" is one single-sequence sequence-list CSV under
``<root>/sequence_lists/<instances-dir>/`` (e.g. one DSEC scene such as
``thun_00_a``). For each instance, this script runs the network over every
frame pair, renders a ground-truth-vs-prediction video, and computes
continuous flow-error metrics (EPE, angular error, cosine error). Results are
written to one summary CSV (one row per instance, plus an overall average
row) alongside one video per instance.

This does not use the attacks/ module -- it is a plain baseline evaluation.

Examples
--------
Full run over all instances in the default folder::

    python evaluate_test_set.py

Quick CPU smoke test (3 frames per instance)::

    python evaluate_test_set.py --max-chunks-per-instance 3

Explicit instance list, bypassing the folder scan::

    python evaluate_test_set.py --instances thun_00_a.csv
"""

import argparse
import csv
import glob
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from spikingjelly.clock_driven import functional

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d
from data.dsec_dataset_with_camera import DSECDatasetWithCamera
from eval.vector_loss_functions import (
    mod_loss_function,
    angular_loss_function,
    cosine_loss_function,
)
from eval.progress_plot_full_v2 import plot_evolution, plot_gt_pred_events_camera

RAD2DEG = 180.0 / math.pi
KEYS = ["EPE", "angular_deg", "one_minus_cos"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/dataset/saved_flow_data",
                   help="Dataset root (relative path).")
    p.add_argument("--images-root", default="data/dataset/train",
                   help="Root containing <sequence>/images/left/rectified/ camera frames.")
    p.add_argument("--instances-dir", default="test_instances",
                   help="Folder of single-sequence CSVs, relative to <root>/sequence_lists/.")
    p.add_argument("--instances", nargs="+", default=None,
                   help="Explicit sequence-list CSV filenames (relative to "
                        "<root>/sequence_lists/), bypassing --instances-dir discovery.")
    p.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    p.add_argument("--multiply-factor", type=float, default=35.0)
    p.add_argument("--num-frames-per-ts", type=int, default=11)
    p.add_argument("--fps", type=int, default=10, help="Video frame rate.")
    p.add_argument("--outdir", default="results")
    p.add_argument("--max-chunks-per-instance", type=int, default=None,
                   help="Limit number of frame pairs per instance (useful on CPU).")
    return p.parse_args()


def discover_instances(root, instances_dir, explicit):
    """Return sorted [(instance_name, csv_relpath_under_sequence_lists), ...]."""
    seq_lists_root = os.path.join(root, "sequence_lists")

    if explicit:
        pairs = [(os.path.splitext(os.path.basename(f))[0], f) for f in explicit]
    else:
        pattern = os.path.join(seq_lists_root, instances_dir, "*.csv")
        csv_paths = sorted(glob.glob(pattern))
        pairs = [
            (os.path.splitext(os.path.basename(p))[0],
             os.path.join(instances_dir, os.path.basename(p)))
            for p in csv_paths
        ]

    if not pairs:
        raise RuntimeError(
            f"No test instances found. Populate '{os.path.join(seq_lists_root, instances_dir)}' "
            "with one single-sequence CSV per instance, or pass --instances explicitly."
        )

    return pairs


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


def evaluate_instance(net, device, root, images_root, sequence, csv_relpath,
                      num_frames_per_ts, max_chunks, desc):
    dataset = DSECDatasetWithCamera(root=root, file_list=csv_relpath, images_root=images_root,
                                    sequence=sequence, num_frames_per_ts=num_frames_per_ts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         drop_last=False, pin_memory=True)

    metric_sum = {k: 0.0 for k in KEYS}
    label_seq, pred_seq, mask_seq, events_seq, camera_seq = [], [], [], [], []
    n = 0

    for chunk, mask, label, image in tqdm(loader, desc=desc, leave=False):
        chunk = torch.transpose(chunk, 1, 2)              # [B, C, T, H, W]
        mask = torch.unsqueeze(mask, dim=1)               # [B, 1, H, W]
        chunk = chunk.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)  # [B, 2, H, W]
        mask = mask.to(device=device)

        pred = predict(net, chunk)

        for k, v in zip(KEYS, metrics(pred, label, mask)):
            metric_sum[k] += v
        n += 1

        label_seq.append(torch.squeeze(label[0]).cpu().numpy())
        pred_seq.append(torch.squeeze(pred[0]).cpu().numpy())
        mask_seq.append(torch.squeeze(mask[0]).cpu().numpy())
        events_seq.append(torch.sum(chunk[0], dim=1).cpu().numpy())  # [C, T, H, W] -> [C, H, W]
        camera_seq.append(torch.squeeze(image[0]).numpy())           # [H, W, 3] BGR

        if max_chunks is not None and n >= max_chunks:
            break

    if n == 0:
        raise RuntimeError(f"Instance '{csv_relpath}' has no samples.")

    avg = {k: metric_sum[k] / n for k in KEYS}
    return {
        "avg": avg,
        "n_samples": n,
        "label_seq": np.array(label_seq),
        "pred_seq": np.array(pred_seq),
        "mask_seq": np.array(mask_seq),
        "events_seq": np.array(events_seq),
        "camera_seq": np.array(camera_seq),
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Device: {device}")
    instances = discover_instances(args.root, args.instances_dir, args.instances)
    print(f"Discovered {len(instances)} test instance(s): {[name for name, _ in instances]}")

    net = NeuronPool_Separable_Pool3d(multiply_factor=args.multiply_factor).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    rows = []
    per_instance_avgs = []
    total_samples = 0

    for name, csv_relpath in tqdm(instances, desc="Instances"):
        result = evaluate_instance(net, device, args.root, args.images_root, name, csv_relpath,
                                   args.num_frames_per_ts, args.max_chunks_per_instance,
                                   desc=name)

        video_path = os.path.join(args.outdir, f"flow_{name}.mp4")
        plot_evolution(result["label_seq"], result["pred_seq"], result["mask_seq"],
                       args.fps, video_path)

        preview_path = os.path.join(args.outdir, f"preview_{name}.mp4")
        plot_gt_pred_events_camera(result["label_seq"], result["pred_seq"], result["events_seq"],
                                   result["camera_seq"], result["mask_seq"], args.fps, preview_path)

        rows.append([name, result["n_samples"]] +
                    [result["avg"][k] for k in KEYS] + [video_path])
        per_instance_avgs.append(result["avg"])
        total_samples += result["n_samples"]

    overall_avg = {k: sum(a[k] for a in per_instance_avgs) / len(per_instance_avgs)
                   for k in KEYS}
    rows.append(["OVERALL_AVERAGE", total_samples] +
                [overall_avg[k] for k in KEYS] + [""])

    csv_path = os.path.join(args.outdir, "test_set_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "n_samples"] + KEYS + ["video_path"])
        w.writerows(rows)

    print(f"\nPer-instance results:")
    header = f"{'instance':<28}{'n':>6}{'EPE':>10}{'angular_deg':>14}{'one_minus_cos':>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        name, n = row[0], row[1]
        epe, ang, cos = row[2], row[3], row[4]
        print(f"{name:<28}{n:>6}{epe:>10.4f}{ang:>14.4f}{cos:>16.4f}")

    print(f"\nMetrics written to {csv_path}")
    print(f"Detailed gt/pred/error videos written to {args.outdir}/flow_<instance>.mp4")
    print(f"Quick gt/pred/events preview videos written to {args.outdir}/preview_<instance>.mp4")


if __name__ == "__main__":
    main()
