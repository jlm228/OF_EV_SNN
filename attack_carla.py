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
    from attack_core import preflight, surrogates                         # noqa: E402
    return attack_core, band_mod, runner, preflight, surrogates


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


def _surrogate_kwargs(args):
    """Constructor arguments for the chosen surrogate."""
    if args.surrogate in ("assg", "assgs"):
        if args.assg_A is None:
            raise SystemExit(
                "--surrogate %s needs --assg-A. It is tuned per model and then frozen, so "
                "there is no default: a bound in (0,1) on the atan base, or the sharpness "
                "scale itself on the sigmoid base." % args.surrogate)
        return {"A": args.assg_A, "gamma": args.assg_gamma,
                "beta1": args.assg_betas[0], "beta2": args.assg_betas[1]}
    if args.surrogate == "pdsg":
        return {"mode": args.pdsg_mode, "channel_dim": args.pdsg_channel_dim}
    return {}


def preflight_input(capture_dir, model, window):
    """One model input for `window`, binned from events.npy in memory.

    The pre-flight runs before anything has been voxelised, and the pipeline deletes the voxels
    once inference is done, so `<capture>/tensors` cannot be relied on. This rebuilds the exact
    tensor the voxelised path would hand the attack: `bin_window_events` per window, as
    `run_voxelise` calls it, then the `concat(frame_prev, frame)[-21:]` DSECDatasetLite applies.

    It deliberately does NOT go through `input_from_events`, which splits one span at its
    midpoint and so assumes consecutive windows are adjacent.
    """
    from groundtruth.inspect_capture import bin_window_events, load_capture

    events, windows, meta = load_capture(capture_dir, mmap=True)
    t_starts = windows["t_start_us"].to_numpy()
    if window < 1 or window >= len(t_starts):
        raise SystemExit(
            "--preflight-window %d is out of range: this model needs window w and the one "
            "before it, so w must be in [1, %d]" % (window, len(t_starts) - 1))

    window_us = int(round(float(meta["window_s"]) * 1e6))
    stride = (int(t_starts[window]) - int(t_starts[window - 1])
              if not np.isnan(t_starts[window - 1]) else None)
    print("capture geometry: window %d us, row stride %s us%s"
          % (window_us, stride,
             " (windows overlap)" if stride is not None and stride < window_us else ""))

    frames = []
    for w in (window - 1, window):
        if np.isnan(t_starts[w]):
            raise SystemExit("window %d recorded no events; pick another "
                             "--preflight-window" % w)
        t0 = int(t_starts[w])
        frames.append(bin_window_events(events["x"], events["y"], events["t"], events["pol"],
                                        t0, t0 + window_us, num_bins=model.num_bins,
                                        height=model.height, width=model.width))

    # [-21:] as DSECDatasetLite does: 10 bins of the previous window, then this window's 11.
    chunk = np.concatenate(frames, axis=0)[-21:]
    return torch.transpose(torch.from_numpy(chunk).unsqueeze(0), 1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="capture dir, with <dir>/tensors alongside")
    ap.add_argument("--objective", required=True,
                    choices=["random_sign", "epe_global", "epe_masked", "div"])
    ap.add_argument("--sign", default="suppress", choices=["suppress", "inflate", "none"],
                    help="div only: suppress reads tau LONG, inflate reads it SHORT. 'none' is "
                         "the placeholder the sweep manifest carries for objectives that have "
                         "no direction, and is ignored unless --objective is div")
    ap.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "sapgd", "sda"])
    ap.add_argument("--surrogate", default="native",
                    choices=["native", "pdsg", "assg", "assgs"],
                    help="gradient substitute during the attack. assg is the Atan base, assgs "
                         "the sigmoid one, which is this model's own family (native Sigmoid, "
                         "alpha=4)")
    ap.add_argument("--assg-A", type=float, default=None,
                    help="ASSG sharpness setting. A bound in (0,1) on the atan base; the "
                         "sharpness scale itself on the sigmoid base. Required for assg/assgs")
    ap.add_argument("--assg-gamma", type=float, default=1.5)
    ap.add_argument("--assg-betas", type=float, nargs=2, default=(0.9, 0.9),
                    metavar=("BETA1", "BETA2"))
    ap.add_argument("--pdsg-mode", default="channel", choices=["channel", "layer"])
    ap.add_argument("--pdsg-channel-dim", type=int, default=1,
                    help="the polarity axis of (B, 2, T, H, W) after the batch axis")
    ap.add_argument("--sda-tau-factors", type=float, nargs="+", default=None,
                    help="target mode: how far the planner must be made to misread tau. "
                         "suppress drives div to div_clean/f, inflate to f*div_clean")
    ap.add_argument("--sda-budgets", type=float, nargs="+", default=None,
                    help="budget mode: events per window, from calibrate --events")
    ap.add_argument("--sda-epe-margin", type=float, default=None,
                    help="target mode with --objective epe_masked: extra masked EPE, in pixels")
    ap.add_argument("--sda-directions", default="inject_only",
                    choices=["inject_only", "inject_remove"])
    ap.add_argument("--sda-k-init", type=int, default=10,
                    help="candidates tested per round grow as (n+1)*k_init; the search width, "
                         "not the target")
    ap.add_argument("--sda-iters", type=int, default=500)
    ap.add_argument("--sda-fd-batch", type=int, default=8,
                    help="candidates per forward pass. The reference uses 64, which "
                         "does not fit: one window peaks near 12.5 GB")
    ap.add_argument("--sda-time-budget", type=float, default=None,
                    help="seconds before a window is censored; keeps one slow window "
                         "from consuming the job and starving every window after it")
    ap.add_argument("--sda-rank", default="grad", choices=["grad", "random"],
                    help="random is the gradient-free control at matched event mass")
    ap.add_argument("--rhos", type=float, nargs="+", default=None,
                    help="the rho each epsilon was calibrated from, recorded in the reports")
    ap.add_argument("--scene-mass", type=float, default=None,
                    help="events per window, so realised rho can be reported")
    ap.add_argument("--preflight-window", type=int, default=None,
                    help="window to measure on; default: the band's first, or 1")
    ap.add_argument("--preflight", action="store_true",
                    help="measure swap coverage, mean |u| per spiking layer, timing and peak "
                         "memory on one window, then exit without attacking")
    ap.add_argument("--epsilons", type=float, nargs="+", default=None,
                    help="one run covers the whole ramp: the clean forward is computed once "
                         "per window and reused across every epsilon")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=None, help="default: epsilon / 4")
    ap.add_argument("--no-rand-init", action="store_true",
                    help="start PGD at the clean input, not a random point in the ball")
    ap.add_argument("--seed", type=int, default=2305)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate arguments, band, tensors, checkpoint and surrogate "
                         "swap, then exit without attacking")
    ap.add_argument("--support", default="all", choices=["all", "nonzero"])
    ap.add_argument("--band-lo", type=int, default=None)
    ap.add_argument("--band-hi", type=int, default=None)
    ap.add_argument("--band-json", default=None,
                    help="default: <capture>/attack_band.json, from attack_core.band")
    ap.add_argument("--clean-pred", default=None,
                    help="clean prediction dump; every output directory is seeded from it")
    ap.add_argument("--out", default=None, help="root for the attacked dumps")
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

    # --preflight only loads the model and measures it; it writes no dumps, so it should not
    # demand the paths a real run needs.
    if args.attack == "sda":
        # SDA has no epsilon: its levels are tau factors or event budgets.
        if (args.sda_tau_factors is None) == (args.sda_budgets is None):
            ap.error("--attack sda needs exactly one of --sda-tau-factors "
                     "(target mode) or --sda-budgets (budget mode)")
    elif not args.preflight and args.epsilons is None:
        ap.error("--epsilons is required unless --attack sda or --preflight")

    if not args.preflight:
        for flag, value in (("--clean-pred", args.clean_pred), ("--out", args.out)):
            if value is None:
                ap.error("%s is required unless --preflight" % flag)

    # The manifest carries "none" for objectives with no direction; the objective builder
    # only accepts a real sign, and ignores it for everything but div.
    if args.sign == "none":
        args.sign = "suppress"

    _core, band_mod, runner, preflight, surrogates = import_attack_core(args.carla_scripts)

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
    # The pre-flight encodes its own window from events.npy, so it must not go through
    # the voxelised dataset: those tensors are written for inference and deleted
    # afterwards, and the pre-flight is meant to run before any of that.
    if args.preflight:
        load_window, capture_id, n_windows = None, args.capture_id or 'preflight', 0
    else:
        load_window, capture_id, n_windows = build_capture_loader(args.capture, device)
        capture_id = args.capture_id or capture_id

    if args.band_lo is not None and args.band_hi is not None:
        lo, hi = args.band_lo, args.band_hi
    else:
        path = args.band_json or os.path.join(args.capture, "attack_band.json")
        if os.path.exists(path):
            lo, hi, _meta = band_mod.read(path)
        elif args.preflight:
            # Measuring needs a window, not the band: any window with a predecessor
            # will do, and --preflight-window overrides this.
            lo, hi = 1, 1
        else:
            raise SystemExit(
                "no band at %s. Compute it once, in an environment with avoidance's "
                "dependencies:\n  python -m attack_core.band --capture %s"
                % (path, args.capture))

    model = OfEvSnnAdapter(checkpoint_path=args.checkpoint,
                           multiply_factor=args.multiply_factor, device=str(device))
    def forward_eval(x):
        """Prediction with state reset first.

        `OfEvSnnAdapter.forward` leaves the reset to its caller; every window must start from
        a clean membrane state or the prediction depends on evaluation order.
        """
        model.reset_state()
        return model.forward(x)

    # This model's sample is (1, 2, 21, H, W): positions 0-9 are the last 10 bins of window
    # i-1 and 10-20 are window i's own 11 bins, so consecutive samples share 10 bins. Attacking
    # each independently would put two different perturbations on the same bin.
    bin_layout = runner.BinLayout(axis=2, own=slice(10, 21),
                                  inherit_from=slice(11, 21), inherit_to=slice(0, 10))

    # `pool` is an IFNode with V_th = inf used as an integrator: its .v is read, not its spike,
    # so its incoming gradient is 0 -- but an adaptive surrogate there gives NaN, and 0 * NaN
    # would spread NaN over the whole input gradient.
    factory = surrogates.build_surrogate_factory(args.surrogate, **_surrogate_kwargs(args))
    handle = None
    if factory is not None:
        handle = surrogates.swap_surrogates(model.net, factory=factory, skip=("pool",))
        cov = handle.coverage
        print("surrogate %s on %d of %d spiking modules (skipped: %s)"
              % (args.surrogate, cov["n_swapped"], cov["n_candidates"],
                 ", ".join(cov["skipped"]) or "none"))

    if args.preflight:
        try:
            w = args.preflight_window if args.preflight_window is not None else max(lo, 1)
            x = preflight_input(args.capture, model, w).to(device)
            print("preflight on window %d, input %s" % (w, tuple(x.shape)))
            # forward_grad, not forward_eval: the latter is wrapped in no_grad, so there would
            # be nothing to differentiate and the forward+backward timing -- the number that
            # sets EPS_CHUNK -- would be skipped. A plain squared mean stands in for the real
            # objective, whose own cost is negligible beside the network's.
            # base="sigmoid": this model's native family, and the one whose omega is a raw
            # scale that has to be anchored to the measured E|u|.
            preflight.report(model.net, model.forward_grad, x,
                             loss_fn=lambda f: (f ** 2).mean(),
                             native_alpha=4.0, base="sigmoid", skip=("pool",),
                             device=str(device))
        finally:
            if handle is not None:
                surrogates.restore_surrogates(handle)
        raise SystemExit(0)


    # Every forward pass must tell the surrogate it is a new pass, or ASSG's per-time-step
    # moment index keeps climbing: each pass writes to a fresh (M, D) starting at (1, 0), so the
    # sharpness never adapts and the moments grow without bound.
    def with_begin_forward(fn):
        if handle is None:
            return fn

        def wrapped(*args, **kwargs):
            handle.begin_forward()
            return fn(*args, **kwargs)
        return wrapped

    # SDA has no epsilon, so the epsilon-ball control neither applies nor can be
    # built; run_sweep_sda never asks for it.
    random_sign_fn = None
    if args.epsilons:
        control = build_attack("random_sign", epsilon=args.epsilons[0],
                               seed=args.seed)

        def random_sign_fn(x, eps, seed):
            control.epsilon = eps
            return control(x)

    label = runner.attack_label(args.attack, args.surrogate)
    print("of_ev_snn | objective %s%s | attack %s | band [%d, %d] of %d windows"
          % (args.objective, "/" + args.sign if args.objective == "div" else "",
             label, lo, hi, n_windows))
    # The levels this run sweeps, in whichever units the attack uses: SDA has no epsilon,
    # so its reports are keyed by tau factor or event budget instead. Everything below
    # reads the list from here rather than assuming epsilons.
    if args.attack == "sda":
        levels = args.sda_tau_factors or args.sda_budgets
        level_name = "tau factors" if args.sda_tau_factors else "event budgets"
    else:
        levels = args.epsilons
        level_name = "epsilons"
    print("%s: %s" % (level_name, " ".join("%g" % v for v in levels)))

    if args.dry_run:
        # Everything above is setup a real run shares: arguments, the band, the tensors,
        # the checkpoint and the surrogate swap. Checking it here costs seconds on a login
        # node, rather than a GPU queue wait to discover a bad argument.
        print("dry run: setup OK, %d level(s) over band [%d, %d]"
              % (len(levels), lo, hi))
        if handle is not None:
            surrogates.restore_surrogates(handle)
        raise SystemExit(0)

    try:
        if args.attack == "sda":
            reports, _dirs = runner.run_sweep_sda(
                band=(lo, hi), load_window=load_window,
                forward_grad=with_begin_forward(model.forward_grad),
            forward_eval=with_begin_forward(forward_eval),
                epe_fn=mod_loss_function,
                objective=args.objective, sign=args.sign, attack=label, seed=args.seed,
                clean_pred_dir=args.clean_pred, out_root=args.out, capture_id=capture_id,
                model_name="of_ev_snn",
                tau_factors=args.sda_tau_factors, budgets=args.sda_budgets,
                epe_margin=args.sda_epe_margin,
                domain='count', directions=args.sda_directions,
                support_mode=args.support, k_init=args.sda_k_init, iters=args.sda_iters,
                fd_batch=args.sda_fd_batch, rank=args.sda_rank,
                time_budget_s=args.sda_time_budget,
                bin_layout=bin_layout, surrogate_ctx=handle, scene_mass=args.scene_mass,
                dump_adv_tensors=args.dump_adv_tensors, verbose=True)
        else:
            reports, _dirs = runner.run_sweep(
            band=(lo, hi), load_window=load_window,
            forward_grad=with_begin_forward(model.forward_grad),
            forward_eval=with_begin_forward(forward_eval),
            epe_fn=mod_loss_function,
            objective=args.objective, sign=args.sign, attack=label,
            epsilons=args.epsilons, iters=args.iters, alpha=args.alpha, seed=args.seed,
            clean_pred_dir=args.clean_pred, out_root=args.out, capture_id=capture_id,
            model_name="of_ev_snn",
            # Event counts cannot be negative. There is no upper clamp: the count tensor is
            # unbounded above, and capping it would be an event-consistency constraint, not an
            # L-infinity one.
            clip_min=0.0, clip_max=None, support_mode=args.support,
            dump_adv_tensors=args.dump_adv_tensors,
            bin_layout=bin_layout, surrogate_ctx=handle,
            rhos=args.rhos, scene_mass=args.scene_mass,
            rand_init=not args.no_rand_init, random_sign_fn=random_sign_fn)
    finally:
        # Must run even when the attack raises, or a failed window leaves the adaptive
        # surrogate installed for whatever runs next in this process.
        if handle is not None:
            surrogates.restore_surrogates(handle)

    paths = runner.write_reports(reports, args.report or os.path.join(args.out, "reports"),
                                 reports[levels[0]]["label"])
    print("\nreports:")
    for level in levels:
        print("  %s" % paths[level])


if __name__ == "__main__":
    main()
