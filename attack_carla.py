"""Adversarially perturb OF_EV_SNN's input over a CARLA capture and dump the flow it predicts.

    python attack_carla.py --capture <capture_dir> --objective div --sign suppress \
        --attack pgd --epsilons 0.0 0.05 0.1 --clean-pred results/carla_eval/pred/of_ev_snn \
        --out results/attack/of_ev_snn --report results/attack/of_ev_snn/reports

The objective and the optimisation loop live in CARLA-hpc-scripts/attack_core, shared with
SDformerFlow, so "the same attack across three models" is true by construction. This file is
only the model-specific half: the capture loader, the adapter, and the epsilon budget's units.

Attacked predictions are dumped in the same (2,H,W) float32 layout as clean ones, so nothing in
`avoidance/` changes -- point `run_case --pred` at the output directory.

Epsilon is applied to the RAW ON/OFF event-count tensor, clamped non-negative. One unit means
"add one event to every pixel, in every polarity channel, in every time bin".
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

from attacks.base import build_attack
from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite
from eval.vector_loss_functions import mod_loss_function
from models.flow_model import OfEvSnnAdapter

DEFAULT_CARLA_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "CARLA-hpc-scripts")


def import_attack_core(path=None):
    """Import attack_core from the CARLA-hpc-scripts checkout.

    Located by $CARLA_SCRIPTS_ROOT, falling back to ../CARLA-hpc-scripts, the same mechanism
    SDformerFlow/carla_eval/carla_to_voxel.py uses to reach inspect_capture. The objective must
    exist exactly once or the cross-model comparison quietly stops being one.
    """
    root = os.path.abspath(path or os.environ.get("CARLA_SCRIPTS_ROOT")
                           or DEFAULT_CARLA_SCRIPTS)
    if not os.path.isdir(os.path.join(root, "attack_core")):
        raise SystemExit(
            "attack_core not found under %s.\n"
            "Point $CARLA_SCRIPTS_ROOT at your CARLA-hpc-scripts checkout, or pass "
            "--carla-scripts." % root)
    sys.path.insert(0, root)
    import attack_core                                                    # noqa: E402
    from attack_core import band as band_mod, runner                      # noqa: E402
    return attack_core, band_mod, runner


def build_capture_loader(capture_dir, device):
    """(load_window, capture_id, n_windows) over a voxelised capture.

    Reads exactly what carla_epe_eval.py reads (the same DSECDatasetLite over
    <capture>/tensors), so the attacked run walks the windows the clean run walked, in the
    same representation. `ped_mask_tensors` carries the hazard mask, written by
    inspect_capture.labels_for_window.
    """
    tdir = os.path.join(capture_dir, "tensors")
    csvs = [f for f in os.listdir(os.path.join(tdir, "sequence_lists", "test_instances"))
            if f.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise SystemExit("expected 1 sequence-list CSV in %s, found %d" % (tdir, len(csvs)))

    dataset = DSECDatasetLite(root=tdir,
                              file_list=os.path.join("test_instances", csvs[0]),
                              num_frames_per_ts=11, stereo=False, transform=None)
    files = dataset.files.iloc[:, 1].tolist()
    capture_id = re.sub(r"_\d{4}\.npy$", "", files[0])
    ped_dir = os.path.join(tdir, "ped_mask_tensors")

    # Tensor filenames are 1-based over windows.csv rows, which are 0-based.
    by_window = {int(re.search(r"_(\d{4})\.npy$", f).group(1)) - 1: k
                 for k, f in enumerate(files)}

    def load_window(i):
        k = by_window.get(i)
        if k is None:
            return None
        chunk, valid, label = dataset[k]
        # (21, 2, H, W) -> (1, 2, 21, H, W), the transpose carla_epe_eval.py applies.
        x = torch.transpose(torch.as_tensor(chunk).unsqueeze(0), 1, 2).to(
            device=device, dtype=torch.float32)
        gt = torch.as_tensor(label).unsqueeze(0).to(device=device, dtype=torch.float32)
        valid = torch.as_tensor(valid).unsqueeze(0).unsqueeze(0).to(
            device=device, dtype=torch.float32)
        haz = torch.from_numpy(np.load(os.path.join(ped_dir, files[k]))).unsqueeze(0).unsqueeze(
            0).to(device=device, dtype=torch.float32)
        return x, gt, valid, haz

    return load_window, capture_id, len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="capture dir, with <dir>/tensors alongside")
    ap.add_argument("--objective", required=True,
                    choices=["random_sign", "epe_global", "epe_masked", "div"])
    ap.add_argument("--sign", default="suppress", choices=["suppress", "inflate"],
                    help="div only: suppress reads tau LONG, inflate reads it SHORT")
    ap.add_argument("--attack", default="pgd", choices=["fgsm", "pgd"])
    ap.add_argument("--epsilons", type=float, nargs="+", required=True,
                    help="one run covers the whole ramp: the clean forward is computed once "
                         "per window and reused across every epsilon")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=None, help="default: epsilon / 4")
    ap.add_argument("--seed", type=int, default=2305)
    ap.add_argument("--support", default="all", choices=["all", "nonzero"])
    ap.add_argument("--band-lo", type=int, default=None)
    ap.add_argument("--band-hi", type=int, default=None)
    ap.add_argument("--band-json", default=None,
                    help="default: <capture>/attack_band.json, from attack_core.band")
    ap.add_argument("--clean-pred", required=True,
                    help="clean prediction dump; every output directory is seeded from it")
    ap.add_argument("--out", required=True, help="root for the attacked dumps")
    ap.add_argument("--report", default=None, help="default: <out>/reports")
    ap.add_argument("--dump-adv-tensors", default=None,
                    help="also write the perturbed INPUT tensors, for the Stage 6 transfer "
                         "check. Note: OF_EV_SNN's representation has no swin counterpart, so "
                         "these do not transfer to SDformerFlow -- see the plan")
    ap.add_argument("--capture-id", default=None)
    ap.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    ap.add_argument("--multiply-factor", type=float, default=35.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--carla-scripts", default=None)
    ap.add_argument("--round-trip", default=None, metavar="REPORT_JSON",
                    help="verify a finished run instead of attacking: re-score its dumped "
                         ".npy and compare against what the objective reported")
    args = ap.parse_args()

    _core, band_mod, runner = import_attack_core(args.carla_scripts)

    if args.round_trip:
        from attack_core.reference import round_trip
        with open(args.round_trip) as fh:
            rep = json.load(fh)
        ok, rows = round_trip(args.round_trip, rep["pred_dir"], rep["capture_id"],
                              mask_dir=os.path.join(args.capture, "tensors",
                                                    "ped_mask_tensors"))
        worst = max((r.get("rel_delta", 0.0) for r in rows), default=0.0)
        print("round trip: %d windows | worst relative div error %.3e | %s"
              % (len(rows), worst, "PASS" if ok else "FAIL"))
        for r in rows:
            if not r.get("passed", True):
                print("  window %s: reported %.6g, recomputed %.6g"
                      % (r["window"], r.get("div_reported"), r.get("div_recomputed")))
        raise SystemExit(0 if ok else 1)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    load_window, capture_id, n_windows = build_capture_loader(args.capture, device)
    capture_id = args.capture_id or capture_id

    if args.band_lo is not None and args.band_hi is not None:
        lo, hi = args.band_lo, args.band_hi
    else:
        path = args.band_json or os.path.join(args.capture, "attack_band.json")
        if not os.path.exists(path):
            raise SystemExit(
                "no band at %s. Compute it once, in an environment with avoidance's "
                "dependencies:\n  python -m attack_core.band --capture %s"
                % (path, args.capture))
        lo, hi, _meta = band_mod.read(path)

    model = OfEvSnnAdapter(checkpoint_path=args.checkpoint,
                           multiply_factor=args.multiply_factor, device=str(device))
    control = build_attack("random_sign", epsilon=args.epsilons[0], seed=args.seed)

    def random_sign_fn(x, eps, seed):
        control.epsilon = eps
        return control(x)

    print("of_ev_snn | objective %s%s | attack %s | band [%d, %d] of %d windows"
          % (args.objective, "/" + args.sign if args.objective == "div" else "",
             args.attack, lo, hi, n_windows))
    print("epsilons: %s" % " ".join("%g" % e for e in args.epsilons))

    reports, _dirs = runner.run_sweep(
        band=(lo, hi), load_window=load_window,
        forward_grad=model.forward_grad, forward_eval=model.forward,
        epe_fn=mod_loss_function,
        objective=args.objective, sign=args.sign, attack=args.attack,
        epsilons=args.epsilons, iters=args.iters, alpha=args.alpha, seed=args.seed,
        clean_pred_dir=args.clean_pred, out_root=args.out, capture_id=capture_id,
        model_name="of_ev_snn",
        # Event counts cannot be negative. There is no upper clamp: the count tensor is
        # unbounded above, and capping it would be an event-consistency constraint, not an
        # L-infinity one.
        clip_min=0.0, clip_max=None, support_mode=args.support,
        dump_adv_tensors=args.dump_adv_tensors, random_sign_fn=random_sign_fn)

    paths = runner.write_reports(reports, args.report or os.path.join(args.out, "reports"),
                                 reports[args.epsilons[0]]["label"])
    print("\nreports:")
    for eps in args.epsilons:
        print("  %s" % paths[eps])


if __name__ == "__main__":
    main()
