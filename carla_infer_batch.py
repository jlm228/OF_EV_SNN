"""Clean OF_EV_SNN inference over many captures, keeping compact predictions.

    python carla_infer_batch.py --captures <list.txt> --out <dir> --scratch <dir> \
        [--shard I --shards N]
    python carla_infer_batch.py --check <capture_dir> --out <dir> --scratch <dir>

The capture loop, window selection, scoring and compaction are CARLA-hpc-scripts'
flow/batch_infer.py. This file supplies the converter (carla_voxelise.run_voxelise) and the
prediction loop of carla_epe_eval.run: one sample at a time, state reset before each.
--check runs flow/batch_infer.check on one capture instead.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

from carla_voxelise import run_voxelise
from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite
from models.flow_model import OfEvSnnAdapter

MODEL = "of_ev_snn"
DEFAULT_CARLA_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                     "CARLA-hpc-scripts")


def import_batch_infer(path=None):
    """flow.batch_infer from the CARLA-hpc-scripts checkout."""
    root = os.path.abspath(path or os.environ.get("CARLA_SCRIPTS_ROOT") or DEFAULT_CARLA_SCRIPTS)
    if not os.path.isfile(os.path.join(root, "flow", "batch_infer.py")):
        raise SystemExit("flow/batch_infer.py not found under %s; point $CARLA_SCRIPTS_ROOT at "
                         "your CARLA-hpc-scripts checkout." % root)
    sys.path.insert(0, root)
    from flow import batch_infer
    return batch_infer


def convert(capture_dir, tensors_dir, capture_id, windows):
    run_voxelise(capture_dir, tensors_dir, capture_id=capture_id, only_windows=windows)


def make_predict(model):
    def predict(name, tensors_dir, capture_id, pred_dir):
        """carla_epe_eval.run's loop, writing (2, H, W) float32 per target."""
        csvs = glob.glob(os.path.join(tensors_dir, "sequence_lists", "test_instances", "*.csv"))
        if len(csvs) != 1:
            raise RuntimeError("expected 1 sequence list in %s, found %d" % (tensors_dir, len(csvs)))
        ds = DSECDatasetLite(root=tensors_dir,
                             file_list=os.path.join("test_instances", os.path.basename(csvs[0])),
                             num_frames_per_ts=11, stereo=False, transform=None)
        files = ds.files.iloc[:, 1].tolist()
        os.makedirs(pred_dir, exist_ok=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        for idx, (chunk, _mask, _label) in enumerate(loader):
            model.reset_state()
            pred = model.forward(torch.transpose(chunk, 1, 2))
            np.save(os.path.join(pred_dir, os.path.basename(files[idx])),
                    pred[0].detach().cpu().numpy().astype(np.float32))
    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--captures", help="file listing capture directories, one per line")
    g.add_argument("--check", help="one capture directory to check end to end")
    ap.add_argument("--out", required=True, help="output root (pred/, epe/, done/, failed/)")
    ap.add_argument("--scratch", required=True, help="working space, emptied per capture")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    ap.add_argument("--multiply-factor", type=float, default=35.0)
    ap.add_argument("--prior", default=os.path.join("results", "carla_eval",
                                                     "dsec_validity_prior.npy"))
    ap.add_argument("--carla-scripts", default=None)
    args = ap.parse_args()

    for path in (args.checkpoint, args.prior):
        if not os.path.isfile(path):
            raise SystemExit("not found: %s" % path)
    bi = import_batch_infer(args.carla_scripts)
    model = OfEvSnnAdapter(checkpoint_path=args.checkpoint,
                           multiply_factor=args.multiply_factor)
    print("OF_EV_SNN on %s" % model.device, flush=True)
    predict = make_predict(model)

    if args.check:
        ok = bi.check(args.check, [MODEL], convert, predict, args.out, args.scratch,
                      prior=args.prior)
        raise SystemExit(0 if ok else 1)
    caps = bi.shard(bi.read_capture_list(args.captures), args.shard, args.shards)
    failures = bi.run(caps, [MODEL], convert, predict, args.out, args.scratch, prior=args.prior)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
