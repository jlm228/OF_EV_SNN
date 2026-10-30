#!/usr/bin/env python
"""Convert a CARLA capture into the DSECDatasetLite tensor layout this model reads.

    python carla_voxelise.py <capture_dir> --out <capture_dir>/tensors [--capture-id ID]

Ground-truth labels come from CARLA-hpc-scripts (capture/groundtruth/inspect_capture.py), found
through $CARLA_SCRIPTS_ROOT or ../CARLA-hpc-scripts.
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
import pandas as pd


def bin_window_events(x, y, t, pol, t_start_us, t_end_us, num_bins=11, height=480, width=640):
    """Bin one window's events into a [num_bins, 2, H, W] ON/OFF count tensor."""
    # t is sorted: slice each bin by bisection.
    lo = int(np.searchsorted(t, t_start_us, side="left"))
    hi = int(np.searchsorted(t, t_end_us, side="left"))
    shape = (num_bins, 2, height, width)
    if hi <= lo:
        return np.zeros(shape, dtype=np.float32)

    xs, ys = x[lo:hi].astype(np.int64), y[lo:hi].astype(np.int64)
    ts, off = t[lo:hi], (~pol[lo:hi].astype(bool)).astype(np.int64)
    dt = (t_end_us - t_start_us) / num_bins
    b_idx = np.clip(((ts - t_start_us) / dt).astype(np.int64), 0, num_bins - 1)

    # One bincount over a flattened (bin, polarity, y, x) index.
    flat = ((b_idx * 2 + off) * height + ys) * width + xs
    counts = np.bincount(flat, minlength=num_bins * 2 * height * width)
    return counts.reshape(shape).astype(np.float32)


def _inspect_capture():
    """capture.groundtruth.inspect_capture from the CARLA-hpc-scripts checkout."""
    root = os.path.abspath(os.environ.get("CARLA_SCRIPTS_ROOT")
                           or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                           "CARLA-hpc-scripts"))
    if root not in sys.path:
        sys.path.insert(0, root)
    from capture.groundtruth import inspect_capture
    return inspect_capture


def run_voxelise(capture_dir, out_dir, capture_id=None, flow_y_flip=True, gt_offset=1,
                 depth_max=500.0, only_windows=None):
    """Write a capture in DSECDatasetLite's layout, applying flow_y_flip and gt_offset.

    only_windows, if given, limits the output to those 0-based window indices.
    """
    ic = _inspect_capture()
    keep = None if only_windows is None else set(only_windows)
    events, windows, meta = ic.load_capture(capture_dir)
    capture_id = capture_id or os.path.basename(os.path.normpath(capture_dir))
    height, width = meta["height"], meta["width"]
    window_s = meta["window_s"]

    ev_dir = os.path.join(out_dir, "event_tensors", "11frames")
    gt_dir = os.path.join(out_dir, "gt_tensors")
    mask_dir = os.path.join(out_dir, "mask_tensors")
    ped_dir = os.path.join(out_dir, "ped_mask_tensors")
    seqlist_dir = os.path.join(out_dir, "sequence_lists", "test_instances")
    for d in (ev_dir, gt_dir, mask_dir, ped_dir, seqlist_dir):
        os.makedirs(d, exist_ok=True)

    n_windows = len(windows)
    t_starts = windows["t_start_us"].to_numpy()
    # A contiguous copy: on the structured array's strided "t" field, searchsorted would copy the
    # whole column on every call.
    t_all = np.ascontiguousarray(events["t"])

    written = []
    for i in range(n_windows):
        if keep is not None and i not in keep:
            continue
        t_start = t_starts[i]
        if pd.isna(t_start):
            continue  # window recorded zero events, nothing to bin
        t_start = int(t_start)
        t_end = t_start + int(round(window_s * 1e6))

        # Skip windows with no label.
        gt = ic.labels_for_window(capture_dir, i, height, width, flow_y_flip=flow_y_flip,
                                  gt_offset=gt_offset, depth_max=depth_max)
        if gt is None:
            continue
        flow_2hw, mask, ped = gt

        frame = bin_window_events(events["x"], events["y"], t_all, events["pol"],
                                  t_start, t_end, num_bins=11, height=height, width=width)
        fname = "%s_%04d.npy" % (capture_id, i + 1)
        np.save(os.path.join(ev_dir, fname), frame)
        np.save(os.path.join(gt_dir, fname), flow_2hw)
        np.save(os.path.join(mask_dir, fname), mask)
        np.save(os.path.join(ped_dir, fname), ped)

        written.append(fname)

    rows = [(written[i - 1], written[i]) for i in range(1, len(written))]
    csv_path = os.path.join(seqlist_dir, "%s.csv" % capture_id)
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    rgb_src = os.path.join(capture_dir, "rgb")
    rgb_dst = os.path.join(out_dir, "rgb")
    if os.path.isdir(rgb_src) and not os.path.isdir(rgb_dst):
        shutil.copytree(rgb_src, rgb_dst)

    print("wrote %d event/gt/mask/ped tensor sets -> %s" % (len(written), out_dir))
    print("  flow y-flip: %s   gt_offset: %+d   depth_max: %.0f m"
          % (flow_y_flip, gt_offset, depth_max))
    print("wrote %d sequence-list rows -> %s" % (len(rows), csv_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir")
    ap.add_argument("--out", default=None, help="output dir (default: <capture_dir>/tensors)")
    ap.add_argument("--capture-id", default=None,
                    help="filename prefix for the tensors (default: capture_dir's basename)")
    ap.add_argument("--no-flow-y-flip", action="store_true",
                    help="do not negate the ground-truth flow's y channel")
    ap.add_argument("--gt-offset", type=int, default=1,
                    help="flow_gt frame that labels window i is i + gt_offset")
    ap.add_argument("--depth-max", type=float, default=500.0,
                    help="depth (m) above which a pixel is treated as sky and masked out")
    args = ap.parse_args()
    run_voxelise(args.capture_dir, args.out or os.path.join(args.capture_dir, "tensors"),
                 capture_id=args.capture_id, flow_y_flip=not args.no_flow_y_flip,
                 gt_offset=args.gt_offset, depth_max=args.depth_max)


if __name__ == "__main__":
    main()
