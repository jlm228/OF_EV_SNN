"""Measure flow-prediction error for a CARLA capture or a DSEC sequence.

    python carla_epe_eval.py --carla data/CARLA/50761933
    python carla_epe_eval.py --dsec thun_00_a

Both datasets run through the same loop and the same metrics, so their numbers are directly
comparable. Metrics come from eval/vector_loss_functions.py, the functions the published
baseline was computed with.

CARLA metrics are reported under two masks, since the choice of mask moves the result more
than anything else does:

  dense        every non-sky pixel (~81% of the image).
  dsec_matched restricted to a DSEC-shaped region (~21%, see build_dsec_validity_prior.py).
               DSEC's GT validity mask omits the sky and nearly all of the near-field road,
               which is where flow magnitude peaks, so a dense mask would score CARLA over a
               higher-flow pixel population than DSEC is ever scored over. DSEC has no GT
               outside its mask, so only the CARLA side can be moved.

A per-pixel histogram of endpoint error against |GT| magnitude is accumulated alongside, which
allows error to be compared at matched flow magnitude -- the confound that remains once the
masks agree. It is built during the run because obtaining it later means re-running the network.
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite
from eval.vector_loss_functions import angular_loss_function, mod_loss_function
from models.flow_model import OfEvSnnAdapter

RAD2DEG = 180.0 / math.pi
DSEC_ROOT = os.path.join("data", "dataset", "saved_flow_data")
# |GT flow| bin edges in px per 100 ms window; the open top bin absorbs near-field road pixels.
MAG_BINS = np.array([0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, np.inf])


def metrics(pred, label, mask):
    """Masked EPE and angular error in degrees. Shapes [B,2,H,W] and [B,1,H,W].

    rel_loss_function is deliberately not used here: it divides per pixel by |GT|, which is
    unbounded wherever ground-truth flow approaches zero. Dense ground truth contains such
    pixels in quantity -- the focus of expansion, distant static scenery -- so the result is
    dominated by them. Normalising by aggregate mean |GT| instead keeps the same intent
    without the singularity.
    """
    return (mod_loss_function(pred, label, mask).item(),
            angular_loss_function(pred, label, mask).item() * RAD2DEG)


def accumulate_magnitude_hist(pred, label, mask, sums, counts):
    """Add this sample's per-pixel endpoint errors into bins of |GT| magnitude.

    Binning individual pixels rather than per-window means keeps the error-versus-magnitude
    relationship intact; averaging within a window would flatten it.
    """
    with torch.no_grad():
        err = torch.sqrt((pred[:, 0] - label[:, 0]) ** 2 + (pred[:, 1] - label[:, 1]) ** 2)
        mag = torch.sqrt(label[:, 0] ** 2 + label[:, 1] ** 2)
        m = mask[:, 0] > 0
        e, g = err[m].cpu().numpy(), mag[m].cpu().numpy()
    idx = np.digitize(g, MAG_BINS) - 1
    np.add.at(sums, idx, e)
    np.add.at(counts, idx, 1)


def load_prior(path, target_fraction=None):
    """DSEC validity prior -> boolean region mask matched to DSEC's valid fraction."""
    prior = np.load(path)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    tau = meta["threshold_tau"]
    return prior >= tau, meta


def run(name, dataset, files, model, extra_masks, out_dir, join_df=None, max_samples=None,
        dump_pred=None):
    """Shared evaluation loop. extra_masks maps mask-name -> callable(idx) -> [H,W] or None.

    dump_pred writes each prediction as float32 (2, H, W) named after its ground-truth tensor,
    for score_flow.py, which computes the same metrics from dumped predictions. Run that with
    --verify-against on this loop's metrics CSV to confirm the two agree.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    n = len(loader) if max_samples is None else min(len(loader), max_samples)

    if dump_pred:
        os.makedirs(dump_pred, exist_ok=True)

    per_sample = []
    hists = {k: (np.zeros(len(MAG_BINS) - 1), np.zeros(len(MAG_BINS) - 1)) for k in extra_masks}

    for idx, (chunk, mask, label) in enumerate(tqdm(loader, total=n, desc=name)):
        if max_samples is not None and idx >= max_samples:
            break
        model.reset_state()
        pred = model.forward(torch.transpose(chunk, 1, 2))

        if dump_pred:
            np.save(os.path.join(dump_pred, os.path.basename(files[idx])),
                    pred[0].detach().cpu().numpy().astype(np.float32))
        label = label.to(device=model.device, dtype=torch.float32)
        base = torch.unsqueeze(mask, 1).to(model.device)

        row = {"index": idx, "filename": files[idx]}
        for mname, getter in extra_masks.items():
            extra = getter(idx)
            m = base if extra is None else base * torch.from_numpy(
                extra[None, None]).to(device=model.device, dtype=base.dtype)
            if m.sum() < 1:
                continue
            epe, ang = metrics(pred, label, m)
            gtmag = float((torch.sqrt(label[:, 0] ** 2 + label[:, 1] ** 2)
                           * m[:, 0]).sum().item() / m.sum().item())
            # Predicting zero flow everywhere scores exactly mean |GT|, so that doubles as the
            # control an informative prediction has to beat, and as the scale to normalise by.
            row.update({f"EPE_{mname}": epe, f"angular_deg_{mname}": ang,
                        f"gtmag_{mname}": gtmag, f"zeroflow_{mname}": gtmag,
                        f"EPEnorm_{mname}": epe / gtmag if gtmag > 1e-6 else np.nan,
                        f"npix_{mname}": int(m.sum().item())})
            accumulate_magnitude_hist(pred, label, m, *hists[mname])
        per_sample.append(row)

    df = pd.DataFrame(per_sample)
    if join_df is not None:
        df = df.merge(join_df, how="left", on="index")

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}_per_window.csv"), index=False)

    overall = {"dataset": name, "n_samples": len(df)}
    for c in sorted(df.columns):
        if c.startswith(("EPE_", "angular_deg_", "gtmag_", "zeroflow_")):
            overall[c] = float(df[c].mean())
    # Taken from the aggregates rather than averaging per-window ratios, which would weight
    # low-motion windows as heavily as the rest.
    for mname in {c.split("_", 1)[1] for c in df.columns if c.startswith("EPE_")}:
        if f"gtmag_{mname}" in df:
            overall[f"EPEnorm_{mname}"] = (float(df[f"EPE_{mname}"].mean())
                                            / float(df[f"gtmag_{mname}"].mean()))
    with open(os.path.join(out_dir, f"{name}_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in overall.items():
            w.writerow([k, v])

    rows = []
    for mname, (s, c) in hists.items():
        for b in range(len(MAG_BINS) - 1):
            if c[b]:
                rows.append({"mask": mname, "bin_lo": MAG_BINS[b], "bin_hi": MAG_BINS[b + 1],
                              "n_pixels": int(c[b]), "EPE": s[b] / c[b]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}_epe_vs_magnitude.csv"), index=False)
    return overall, df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--carla", help="raw capture dir (expects <dir>/tensors alongside)")
    g.add_argument("--dsec", help="DSEC sequence name, e.g. thun_00_a")
    ap.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    ap.add_argument("--multiply-factor", type=float, default=35.0)
    ap.add_argument("--outdir", default=os.path.join("results", "carla_eval"))
    ap.add_argument("--prior", default=os.path.join("results", "carla_eval",
                                                     "dsec_validity_prior.npy"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--dump-pred", default=None,
                    help="also write raw (2,H,W) predictions here, for "
                         "CARLA-hpc-scripts/score_flow.py")
    args = ap.parse_args()

    model = OfEvSnnAdapter(checkpoint_path=args.checkpoint,
                            multiply_factor=args.multiply_factor, device=args.device)

    if args.dsec:
        # DSEC carries its own per-window validity mask, so no extra region is applied.
        split = f"{args.dsec}_all.csv"
        split_path = os.path.join(DSEC_ROOT, "sequence_lists", split)
        if not os.path.exists(split_path):
            gts = sorted(os.path.basename(f) for f in
                          glob.glob(os.path.join(DSEC_ROOT, "gt_tensors", f"{args.dsec}_*.npy")))
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            with open(split_path, "w", newline="") as fh:
                csv.writer(fh).writerows([(gts[i - 1], gts[i]) for i in range(1, len(gts))])
            print(f"built {split_path} ({len(gts) - 1} pairs)")
        ds = DSECDatasetLite(root=DSEC_ROOT, file_list=split, num_frames_per_ts=11,
                              stereo=False, transform=None)
        files = ds.files.iloc[:, 1].tolist()
        overall, _ = run(f"dsec_{args.dsec}", ds, files, model,
                          {"dsec": lambda i: None}, args.outdir,
                          max_samples=args.max_samples, dump_pred=args.dump_pred)
    else:
        cap = args.carla.rstrip("/\\")
        tdir = os.path.join(cap, "tensors")
        csvs = glob.glob(os.path.join(tdir, "sequence_lists", "test_instances", "*.csv"))
        if len(csvs) != 1:
            raise SystemExit(f"expected 1 sequence-list CSV in {tdir}, found {len(csvs)}")
        ds = DSECDatasetLite(root=tdir,
                              file_list=os.path.join("test_instances", os.path.basename(csvs[0])),
                              num_frames_per_ts=11, stereo=False, transform=None)
        files = ds.files.iloc[:, 1].tolist()

        region, pmeta = load_prior(args.prior)
        print("dsec_matched region: %.1f%% of image (DSEC's own mean was %.1f%%)"
              % (100 * region.mean(), 100 * pmeta["dsec_mean_valid_fraction"]))

        ped_dir = os.path.join(tdir, "ped_mask_tensors")
        ped = lambda i: np.load(os.path.join(ped_dir, files[i]))

        # "ped" restricts to the pedestrian silhouette; there is no DSEC equivalent, so it is
        # only comparable across CARLA captures.
        overall, df = run(os.path.basename(cap), ds, files, model,
                           {"dense": lambda i: None,
                            "dsec_matched": lambda i: region.astype(np.float32),
                            "ped": ped},
                           args.outdir,
                           join_df=_window_state(cap, files),
                           max_samples=args.max_samples, dump_pred=args.dump_pred)

    print("\n=== %s ===" % overall["dataset"])
    for k, v in overall.items():
        if k not in ("dataset", "n_samples"):
            print("  %-24s %.4f" % (k, v))
    print("  n_samples                %d" % overall["n_samples"])


def _window_state(capture_dir, files):
    """Ego and pedestrian state per sample, keyed by loader position.

    Lets error be plotted against speed or range. Tensor filenames are numbered from 1 over
    windows.csv's rows, which are numbered from 0.
    """
    w = pd.read_csv(os.path.join(capture_dir, "windows.csv"))
    keep = [c for c in ("window", "ego_speed_ms", "dist_to_ped_m", "visible_px", "n_events")
            if c in w.columns]
    w = w[keep].copy()
    win_of_file = [int(re.search(r"_(\d{4})\.npy$", f).group(1)) - 1 for f in files]
    return pd.DataFrame({"index": range(len(files)), "window": win_of_file}).merge(
        w, how="left", on="window")


if __name__ == "__main__":
    main()
