"""Run a trained OF_EV_SNN checkpoint on a voxelised CARLA capture and render a 4-panel
video (GT flow | predicted flow | input events | camera) for visual inspection of the
predicted flow.

    python carla_smoke_test.py <tensors_dir> [--checkpoint examples/checkpoint_epoch34.pth]
                                              [--capture-dir <raw capture dir, for rgb/>]
                                              [--out results/carla_smoke_test.mp4]

<tensors_dir> must contain event_tensors/11frames/, gt_tensors/, mask_tensors/,
sequence_lists/test_instances/, and (if not passing --capture-dir separately) rgb/.
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite
from eval.progress_plot_full_v2 import plot_gt_pred_events_camera
from models.flow_model import OfEvSnnAdapter

FNAME_RE = re.compile(r"_(\d{4})\.npy$")


def find_split_csv(tensors_dir):
    csvs = glob.glob(os.path.join(tensors_dir, "sequence_lists", "test_instances", "*.csv"))
    if len(csvs) != 1:
        raise SystemExit(
            "expected exactly one sequence-list CSV under sequence_lists/test_instances/, "
            "found %d: %s" % (len(csvs), csvs))
    return csvs[0]


def rgb_path_for(gt_filename, rgb_dir):
    m = FNAME_RE.search(gt_filename)
    window_idx = int(m.group(1)) - 1  # filenames are 1-based, windows.csv rows are 0-based
    return os.path.join(rgb_dir, "%05d.png" % window_idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tensors_dir")
    ap.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    ap.add_argument("--multiply-factor", type=float, default=35.0)
    ap.add_argument("--capture-dir", default=None,
                     help="raw capture dir for rgb/ (default: <tensors_dir>/rgb)")
    ap.add_argument("--out", default="results/carla_smoke_test.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    split_csv = find_split_csv(args.tensors_dir)
    print("sequence-list: %s" % split_csv)

    rgb_dir = os.path.join(args.capture_dir, "rgb") if args.capture_dir \
        else os.path.join(args.tensors_dir, "rgb")

    # DSECDatasetLite joins root/sequence_lists/<file_list>, and the CSV lives under
    # sequence_lists/test_instances/, so pass that relative path rather than just the basename.
    file_list = os.path.join("test_instances", os.path.basename(split_csv))
    dataset = DSECDatasetLite(root=args.tensors_dir, file_list=file_list,
                               num_frames_per_ts=11, stereo=False, transform=None)
    print("%d window pairs" % len(dataset))

    model = OfEvSnnAdapter(checkpoint_path=args.checkpoint,
                            multiply_factor=args.multiply_factor, device="cpu")

    gt_seq, pred_seq, events_seq, camera_seq, mask_seq = [], [], [], [], []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, (chunk, mask, label) in enumerate(tqdm(dataloader, desc="running model")):
        model.reset_state()
        x = torch.transpose(chunk, 1, 2)
        pred = model.forward(x)

        gt_seq.append(torch.squeeze(label[0]).numpy())
        pred_seq.append(torch.squeeze(pred[0]).cpu().numpy())
        mask_seq.append(torch.squeeze(mask[0]).numpy())

        # events panel: this window's own 11 bins, summed -- the last 11 of the 21-bin chunk.
        events_seq.append(chunk[0, -11:].sum(axis=0).numpy())

        frame2_name = dataset.files.iloc[idx, 1]
        img_path = rgb_path_for(frame2_name, rgb_dir)
        img = cv2.imread(img_path)
        if img is None:
            raise SystemExit("could not read camera frame: %s" % img_path)
        camera_seq.append(img)

    gt_seq = np.array(gt_seq)
    pred_seq = np.array(pred_seq)
    events_seq = np.array(events_seq)
    camera_seq = np.array(camera_seq)
    mask_seq = np.array(mask_seq)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_gt_pred_events_camera(gt_seq, pred_seq, events_seq, camera_seq, mask_seq,
                                fps=args.fps, filename=args.out)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
