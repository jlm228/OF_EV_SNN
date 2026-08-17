"""Derive a DSEC-shaped image region for restricting dense CARLA metrics.

    python build_dsec_validity_prior.py [--sequence thun_00_a] [--out results/carla_eval]

DSEC's ground truth is only valid over ~21% of the image, and that region is far from uniform:
it drops the sky and nearly all of the near-field road, where flow magnitude is highest. Metrics
computed over dense CARLA ground truth therefore cover a different, faster-moving pixel
population than any DSEC metric does. Since DSEC has no ground truth to add outside its mask,
comparability has to come from narrowing the CARLA side.

Averaging DSEC's per-window masks gives a per-pixel validity rate. Thresholding that rate at the
quantile matching DSEC's own valid fraction yields a region of the same size and shape, which
carla_epe_eval.py applies as an additional mask.
"""
import argparse
import glob
import json
import os

import numpy as np

DEFAULT_MASK_DIR = os.path.join("data", "dataset", "saved_flow_data", "mask_tensors")
# Sequences making up the DSEC validation split. Those without preprocessed masks on disk are
# skipped, so the prior is built from whichever subset is available.
VALID_SPLIT_SEQUENCES = ("thun_00_a", "zurich_city_02_d", "zurich_city_03_a",
                          "zurich_city_08_a", "zurich_city_11_b")


def build_prior(mask_dir, sequences):
    """Per-pixel P(valid) averaged over every available mask of the given sequences."""
    files = []
    used = {}
    for seq in sequences:
        f = sorted(glob.glob(os.path.join(mask_dir, "%s_*.npy" % seq)))
        if f:
            used[seq] = len(f)
            files += f
    if not files:
        raise SystemExit(
            "No mask_tensors found for %s under %s.\nRun DSEC preprocessing first, or pass "
            "--sequence for one that is present locally." % (list(sequences), mask_dir))

    acc = None
    for f in files:
        m = (np.load(f) > 0).astype(np.float64)
        acc = m if acc is None else acc + m
    prior = acc / len(files)
    return prior, used


def choose_threshold(prior, target_fraction):
    """Threshold whose region covers target_fraction of the image.

    Taken from the prior's own quantiles, so it adapts to whichever masks were available
    rather than assuming a fixed level.
    """
    tau = float(np.quantile(prior, 1.0 - target_fraction))
    achieved = float((prior >= tau).mean())
    return tau, achieved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-dir", default=DEFAULT_MASK_DIR)
    ap.add_argument("--sequence", action="append", default=None,
                     help="restrict to this sequence (repeatable). Default: all validation-split "
                          "sequences that have local masks.")
    ap.add_argument("--out", default=os.path.join("results", "carla_eval"))
    args = ap.parse_args()

    sequences = tuple(args.sequence) if args.sequence else VALID_SPLIT_SEQUENCES
    prior, used = build_prior(args.mask_dir, sequences)

    target = float(np.mean([ (np.load(f) > 0).mean()
                             for seq in used
                             for f in sorted(glob.glob(os.path.join(args.mask_dir, "%s_*.npy" % seq))) ]))
    tau, achieved = choose_threshold(prior, target)

    print("DSEC validity prior")
    print("  sequences used      : %s" % ", ".join("%s (%d masks)" % (k, v) for k, v in used.items()))
    missing = [s for s in sequences if s not in used]
    if missing:
        print("  no local masks for  : %s" % ", ".join(missing))
    print("  DSEC mean valid fraction : %.1f%%" % (100 * target))
    print("  chosen threshold tau     : %.4f  -> matched fraction %.1f%%" % (tau, 100 * achieved))
    print("\n  validity rate by image row band:")
    for a, b in ((0, 120), (120, 240), (240, 360), (360, 480)):
        print("    rows %3d-%3d : %5.1f%%" % (a, b, 100 * prior[a:b].mean()))

    os.makedirs(args.out, exist_ok=True)
    prior_path = os.path.join(args.out, "dsec_validity_prior.npy")
    np.save(prior_path, prior.astype(np.float32))
    meta = {
        "sequences_used": used,
        "sequences_missing": missing,
        "n_masks": int(sum(used.values())),
        "dsec_mean_valid_fraction": target,
        "threshold_tau": tau,
        "matched_fraction": achieved,
    }
    with open(os.path.join(args.out, "dsec_validity_prior.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("\nwrote %s" % prior_path)


if __name__ == "__main__":
    main()
