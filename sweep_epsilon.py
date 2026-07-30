"""Sweep an additive L-infinity attack over an epsilon range, with a control.

Run the selected attack (``fgsm`` / ``pgd``) over the DSEC validation split for each 
epsilon in a range, and report clean vs. adversarial flow error as a curve (one row per epsilon).

The clean prediction is epsilon-independent, so it is computed once per sample
and shared across the whole sweep (single pass over the data).

The attack objective is a third axis: ``--losses`` runs one experiment per error
type (fgsm/pgd maximising EPE, angular, or cosine error), and *every* run is
measured under *all* three metrics. Clean is computed once; the random-sign
control is computed once per epsilon (it has no objective).

Outputs (all under ``--outdir``, for a run with ``--attack A``):

* ``raw_A_<sequence>.csv`` -- one file **per sequence**, the full raw data: one
  long-format row per (sample x condition) evaluation, including every individual
  random draw. Columns: ``sample_index, sequence, filename, condition,
  attack_loss, epsilon, draw, EPE, angular_deg, one_minus_cos, count_drift``
  (``condition`` in {clean, A, random}).
* ``per_sequence_A.csv`` -- pooled averages, one row per (sequence, condition,
  attack_loss, epsilon), with mean+std of each metric and count-drift.
* ``sweep_A.csv`` -- the split-level aggregate curve (one row per objective x
  epsilon); a quick convenience view, derivable from the pooled file.

Examples
--------
One experiment per error type (EPE-, angular-, cosine-objective attacks)::

    python sweep_epsilon.py --attack fgsm --losses epe angular cosine \
        --epsilons 0.005 0.01 0.02

PGD sweep (per-step size defaults to epsilon/4 unless --alpha is given)::

    python sweep_epsilon.py --attack pgd --iters 7 \
        --epsilons 0.005 0.01 0.02

See ``attacks/fgsm_pgd/calibrate_epsilon.py`` for choosing the epsilon values.
"""

import argparse
import csv
import math
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

from spikingjelly.clock_driven import functional

from eval.vector_loss_functions import (
    mod_loss_function,
    angular_loss_function,
    cosine_loss_function,
)
from attacks.base import build_attack
from attacks.cli_common import (
    add_common_attack_args,
    add_common_model_args,
    load_model_and_data,
    load_sample_names,
)

RAD2DEG = 180.0 / math.pi
KEYS = ["EPE", "angular_deg", "one_minus_cos"]

# Long-format raw record: one row per (sample x condition) evaluation.
RAW_COLS = ["sample_index", "sequence", "filename", "condition", "attack_loss",
            "epsilon", "draw", "EPE", "angular_deg", "one_minus_cos", "count_drift"]
# Per-sequence pooled record: one row per (sequence, condition, attack_loss, epsilon).
POOL_COLS = ["sequence", "condition", "attack_loss", "epsilon", "n_points",
             "EPE_mean", "EPE_std", "angular_deg_mean", "angular_deg_std",
             "one_minus_cos_mean", "one_minus_cos_std",
             "count_drift_mean", "count_drift_max"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_attack_args(p)
    add_common_model_args(p)
    p.add_argument("--epsilons", type=float, nargs="+", default=None,
                   help="Epsilon sweep values (space-separated). "
                        "If omitted, falls back to the single --epsilon.")
    p.add_argument("--losses", nargs="+", default=None,
                   choices=["epe", "angular", "cosine"],
                   help="Attack objective(s) to sweep -- one experiment per error "
                        "type (fgsm/pgd only). Each is measured under ALL metrics. "
                        "Default: the single --loss, else 'epe'.")
    p.add_argument("--rand-restarts", type=int, default=5,
                   help="R: random-sign control draws per sample per epsilon "
                        "(0 disables the control).")
    p.add_argument("--rand-seed", type=int, default=1234,
                   help="Seed for the random-sign control draws (reproducible).")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


@torch.no_grad()
def predict(net, E):
    """Reset state and return the finest flow field pred_1, shape [B, 2, H, W]."""
    functional.reset_net(net)
    return net(E)[-1]


def metrics(pred, label, M):
    return (
        mod_loss_function(pred, label, M).item(),                # EPE
        angular_loss_function(pred, label, M).item() * RAD2DEG,  # angular error (deg)
        cosine_loss_function(pred, label, M).item(),             # 1 - cos(pred, label)
    )


def build_attack_for(args, eps, loss):
    """Build the selected attack at a specific epsilon and objective.

    Mirrors ``cli_common.build_threat_from_args`` but forces ``epsilon = eps``
    and ``loss`` (the objective being swept). ``alpha``/``rand_init`` are only
    injected when the user set them, so PGD keeps its own defaults.
    """
    cfg = dict(iters=args.iters, seed=args.seed, epsilon=eps, loss=loss)
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.rand_init:
        cfg["rand_init"] = True
    return build_attack(args.attack, **cfg)


def _pool_add(pools, key, mvals, drift):
    """Accumulate one (EPE, angular, cosine) triple + count-drift into ``pools[key]``.

    ``pools`` is an ``OrderedDict`` keyed by (sequence, condition, attack_loss,
    epsilon); each entry keeps running sums / sums-of-squares (for mean + std)
    and the running / max count-drift. First insertion fixes the row's order.
    """
    p = pools.get(key)
    if p is None:
        p = {"n": 0, "sum": [0.0, 0.0, 0.0], "sumsq": [0.0, 0.0, 0.0],
             "drift_sum": 0.0, "drift_max": 0.0}
        pools[key] = p
    p["n"] += 1
    for j in range(3):
        p["sum"][j] += mvals[j]
        p["sumsq"][j] += mvals[j] * mvals[j]
    p["drift_sum"] += drift
    p["drift_max"] = max(p["drift_max"], drift)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    epsilons = args.epsilons if args.epsilons is not None else (
        [args.epsilon] if args.epsilon is not None else None)
    if not epsilons:
        raise SystemExit("Provide --epsilons e0 e1 ... (or a single --epsilon).")
    if args.attack not in ("fgsm", "pgd"):
        print(f"[warning] --attack '{args.attack}' is not an additive L-infinity "
              f"attack; epsilon has no effect for it.")
    R = max(0, args.rand_restarts)

    print("Creating validation dataset ...")
    device, net, dataset, loader = load_model_and_data(args)
    print(f"Device: {device}")
    losses = (args.losses if args.losses is not None
              else ([args.loss] if args.loss is not None else ["epe"]))
    print(f"Attack: {args.attack} | objectives: {losses} | epsilons: {epsilons} | "
          f"random-sign control R = {R}")

    # One attack object per (objective, epsilon); one shared seeded control
    # (epsilon set per use; the control is objective-independent).
    attacks = {(loss, eps): build_attack_for(args, eps, loss)
               for loss in losses for eps in epsilons}
    control = build_attack("random_sign", epsilon=epsilons[0],
                           seed=args.rand_seed) if R > 0 else None

    # Per-sample identity (sequence + filename), aligned to loader order.
    seqs, names = load_sample_names(args)

    # Raw per-sequence writers, opened lazily so the full raw dataset never has
    # to live in memory; one file `raw_<attack>_<sequence>.csv` per sequence.
    raw_writers, raw_handles = {}, {}

    def raw_writer(seq):
        w = raw_writers.get(seq)
        if w is None:
            fh = open(os.path.join(args.outdir, f"raw_{args.attack}_{seq}.csv"),
                      "w", newline="")
            w = csv.writer(fh)
            w.writerow(RAW_COLS)
            raw_writers[seq], raw_handles[seq] = w, fh
        return w

    # Per-sequence pooled accumulators, keyed (sequence, condition, loss, epsilon).
    pools = OrderedDict()

    # Aggregate accumulators (the retained sweep_<attack>.csv curve).
    # clean: once (objective- and epsilon-independent).
    # random: per epsilon (objective-independent). adv: per (objective, epsilon).
    clean_sum = {k: 0.0 for k in KEYS}
    rand_sum = {eps: {k: 0.0 for k in KEYS} for eps in epsilons}    # per-sample draw-mean, summed
    rand_std_sum = {eps: 0.0 for eps in epsilons}                  # sum over samples of within-draw EPE std
    adv_sum = {(loss, eps): {k: 0.0 for k in KEYS}
               for loss in losses for eps in epsilons}
    drift = {(loss, eps): 0.0 for loss in losses for eps in epsilons}
    n = 0

    for E, M, label in tqdm(loader, desc="Sweeping"):
        E = torch.transpose(E, 1, 2)                                # [B, C, T, H, W]
        M = torch.unsqueeze(M, dim=1)                               # [B, 1, H, W]
        E = E.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)       # [B, 2, H, W]
        M = M.to(device=device)

        seq = seqs[n] if n < len(seqs) else ""
        fname = names[n] if n < len(names) else ""
        wr = raw_writer(seq)

        # Clean prediction is objective and epsilon independent so compute once per sample
        cm = metrics(predict(net, E), label, M)
        for k, cv in zip(KEYS, cm):
            clean_sum[k] += cv
        wr.writerow([n, seq, fname, "clean", "", "", "", cm[0], cm[1], cm[2], 0.0])
        _pool_add(pools, (seq, "clean", "", ""), cm, 0.0)

        for eps in epsilons:
            # Random-sign control: once per epsilon (no objective, gradient-free);
            # every draw is emitted as its own raw row and pooled per sequence.
            if control is not None:
                control.epsilon = eps
                draw_acc = [0.0, 0.0, 0.0]
                epe_sq = 0.0
                for r in range(R):
                    E_rand = control(E)                            # model/label ignored
                    rm = metrics(predict(net, E_rand), label, M)
                    d = (E_rand.sum() - E.sum()).abs().item()
                    wr.writerow([n, seq, fname, "random", "", eps, r,
                                 rm[0], rm[1], rm[2], d])
                    _pool_add(pools, (seq, "random", "", eps), rm, d)
                    for j in range(3):
                        draw_acc[j] += rm[j]
                    epe_sq += rm[0] * rm[0]
                for k, s in zip(KEYS, draw_acc):
                    rand_sum[eps][k] += s / R                       # this sample's mean over R
                # Spread of the control across its own R draws for this sample
                # (0 when R==1); averaged over samples in the report below.
                m_epe = draw_acc[0] / R
                rand_std_sum[eps] += math.sqrt(max(epe_sq / R - m_epe * m_epe, 0.0))

            # Attack: one run per objective (fgsm/pgd), each measured under all metrics.
            for loss in losses:
                E_adv = attacks[(loss, eps)](E, model=net, label=label, M=M)
                d = (E_adv.sum() - E.sum()).abs().item()
                drift[(loss, eps)] = max(drift[(loss, eps)], d)
                am = metrics(predict(net, E_adv), label, M)
                for k, av in zip(KEYS, am):
                    adv_sum[(loss, eps)][k] += av
                wr.writerow([n, seq, fname, args.attack, loss, eps, "",
                             am[0], am[1], am[2], d])
                _pool_add(pools, (seq, args.attack, loss, eps), am, d)

        n += 1
        if args.max_chunks is not None and n >= args.max_chunks:
            break

    if n == 0:
        raise RuntimeError("No samples were evaluated; check --root / --split.")

    clean_avg = {k: clean_sum[k] / n for k in KEYS}

    # ---- Report -----------------------------------------------------------
    print(f"\nEvaluated {n} samples | attack = '{args.attack}' | R = {R}")
    print(f"Clean:  EPE {clean_avg['EPE']:.4f} | angular {clean_avg['angular_deg']:.4f} deg "
          f"| 1-cos {clean_avg['one_minus_cos']:.4f}\n")

    rows = []
    for loss in losses:
        print(f"== attack objective: maximise {loss} ==")
        hdr = (f"{'epsilon':>9}{'adv_EPE':>9}{'adv_ang':>9}{'adv_cos':>9}"
               f"{'rand_EPE':>10}{'adv-rand':>10}{'drift':>11}")
        print(hdr)
        print("-" * len(hdr))
        for eps in epsilons:
            adv_avg = {k: adv_sum[(loss, eps)][k] / n for k in KEYS}
            if control is not None:
                rand_avg = {k: rand_sum[eps][k] / n for k in KEYS}
                rand_std = rand_std_sum[eps] / n
            else:
                rand_avg = {k: float("nan") for k in KEYS}
                rand_std = float("nan")

            adv_vs_rand = adv_avg["EPE"] - rand_avg["EPE"]
            print(f"{eps:>9.4g}{adv_avg['EPE']:>9.4f}{adv_avg['angular_deg']:>9.4f}"
                  f"{adv_avg['one_minus_cos']:>9.4f}{rand_avg['EPE']:>10.4f}"
                  f"{adv_vs_rand:>+10.4f}{drift[(loss, eps)]:>11.4g}")

            rows.append({
                "attack": args.attack,
                "attack_loss": loss,
                "epsilon": eps,
                "clean_EPE": clean_avg["EPE"],
                "clean_angular_deg": clean_avg["angular_deg"],
                "clean_one_minus_cos": clean_avg["one_minus_cos"],
                "adv_EPE": adv_avg["EPE"],
                "adv_angular_deg": adv_avg["angular_deg"],
                "adv_one_minus_cos": adv_avg["one_minus_cos"],
                "rand_EPE_mean": rand_avg["EPE"],
                "rand_EPE_std": rand_std,
                "rand_angular_deg": rand_avg["angular_deg"],
                "rand_one_minus_cos": rand_avg["one_minus_cos"],
                "adv_minus_rand_EPE": adv_vs_rand,
                "delta_adv_EPE": adv_avg["EPE"] - clean_avg["EPE"],
                "delta_rand_EPE": rand_avg["EPE"] - clean_avg["EPE"],
                "max_count_drift": drift[(loss, eps)],
                "n_samples": n,
                "R": R,
            })
        print()

    # ---- Aggregate curve CSV ---------------------------------------------
    csv_path = os.path.join(args.outdir, f"sweep_{args.attack}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Aggregate curve written to {csv_path} ({len(rows)} rows)")

    # ---- Raw per-sequence files (close the streamed writers) --------------
    for fh in raw_handles.values():
        fh.close()
    print(f"Raw per-sample data written to {len(raw_handles)} file(s): "
          f"{os.path.join(args.outdir, f'raw_{args.attack}_<sequence>.csv')}")

    # ---- Per-sequence pooled averages CSV --------------------------------
    pool_path = os.path.join(args.outdir, f"per_sequence_{args.attack}.csv")
    with open(pool_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(POOL_COLS)
        for (seq, cond, aloss, eps), p in pools.items():
            nn = p["n"]
            mean = [p["sum"][j] / nn for j in range(3)]
            std = [math.sqrt(max(p["sumsq"][j] / nn - mean[j] * mean[j], 0.0))
                   for j in range(3)]
            w.writerow([seq, cond, aloss, eps, nn,
                        mean[0], std[0], mean[1], std[1], mean[2], std[2],
                        p["drift_sum"] / nn, p["drift_max"]])
    print(f"Per-sequence pooled averages written to {pool_path} ({len(pools)} rows)")


if __name__ == "__main__":
    main()
