"""Sweep an additive L-infinity attack over an epsilon range, with a control.

Extends the single-point comparison in ``evaluate_attack.py`` to the full
experiment described for FGSM/PGD:

* **Step 4 -- epsilon sweep.** Run the selected attack (``fgsm`` / ``pgd``) over
  the DSEC validation split for each epsilon in a range, and report clean vs.
  adversarial flow error as a curve (one row per epsilon).
* **Step 5 -- random-sign control.** For each epsilon, also apply the
  magnitude-matched control ``E_rand = clamp(E + epsilon * S, 0, inf)`` with
  ``S ~ Uniform{-1, +1}``, averaged over ``R`` seeded draws per sample. This
  isolates the value of the *gradient* sign from merely spending the epsilon
  budget: the headline comparison is adversarial EPE vs. random-sign EPE at the
  same epsilon.

The clean prediction is epsilon-independent, so it is computed once per sample
and shared across the whole sweep (single pass over the data).

Examples
--------
FGSM sweep with a 5-draw random-sign control::

    python sweep_epsilon.py --attack fgsm \
        --epsilons 0.0 0.002 0.005 0.01 0.02 0.05 --rand-restarts 5

PGD sweep (per-step size defaults to epsilon/4 unless --alpha is given)::

    python sweep_epsilon.py --attack pgd --iters 7 \
        --epsilons 0.005 0.01 0.02

See ``attacks/fgsm_pgd/calibrate_epsilon.py`` for choosing the epsilon values.
"""

import argparse
import csv
import math
import os

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
)

RAD2DEG = 180.0 / math.pi
KEYS = ["EPE", "angular_deg", "one_minus_cos"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_attack_args(p)
    add_common_model_args(p)
    p.add_argument("--epsilons", type=float, nargs="+", default=None,
                   help="Epsilon sweep values (space-separated). "
                        "If omitted, falls back to the single --epsilon.")
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


def build_attack_for_epsilon(args, eps):
    """Build the selected attack at a specific epsilon, carrying the CLI config.

    Mirrors ``cli_common.build_threat_from_args`` but forces ``epsilon = eps``.
    Extra keys (iters/alpha/rand_init/loss) are only injected when the user set
    them, so each attack keeps its own defaults.
    """
    cfg = dict(iters=args.iters, seed=args.seed, epsilon=eps)
    if args.loss is not None:
        cfg["loss"] = args.loss
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.rand_init:
        cfg["rand_init"] = True
    return build_attack(args.attack, **cfg)


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
    print(f"Attack: {args.attack} | epsilons: {epsilons} | "
          f"random-sign control R = {R}")

    # One attack object per epsilon; one shared, seeded control (epsilon set per use).
    attacks = {eps: build_attack_for_epsilon(args, eps) for eps in epsilons}
    control = build_attack("random_sign", epsilon=epsilons[0],
                           seed=args.rand_seed) if R > 0 else None

    # Accumulators.
    clean_sum = {k: 0.0 for k in KEYS}
    adv_sum = {eps: {k: 0.0 for k in KEYS} for eps in epsilons}
    rand_sum = {eps: {k: 0.0 for k in KEYS} for eps in epsilons}   # per-sample draw-mean, summed
    rand_std_sum = {eps: 0.0 for eps in epsilons}                  # sum over samples of within-draw EPE std
    drift = {eps: 0.0 for eps in epsilons}
    n = 0

    for E, M, label in tqdm(loader, desc="Sweeping"):
        E = torch.transpose(E, 1, 2)                                # [B, C, T, H, W]
        M = torch.unsqueeze(M, dim=1)                               # [B, 1, H, W]
        E = E.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)       # [B, 2, H, W]
        M = M.to(device=device)

        # Clean prediction is epsilon-independent -- compute once.
        cm = metrics(predict(net, E), label, M)
        for k, cv in zip(KEYS, cm):
            clean_sum[k] += cv

        for eps in epsilons:
            E_adv = attacks[eps](E, model=net, label=label, M=M)
            drift[eps] = max(drift[eps], (E_adv.sum() - E.sum()).abs().item())
            am = metrics(predict(net, E_adv), label, M)
            for k, av in zip(KEYS, am):
                adv_sum[eps][k] += av

            if control is not None:
                control.epsilon = eps
                draw_acc = [0.0, 0.0, 0.0]
                epe_sq = 0.0
                for _ in range(R):
                    E_rand = control(E)                            # model/label ignored
                    rm = metrics(predict(net, E_rand), label, M)
                    for j in range(3):
                        draw_acc[j] += rm[j]
                    epe_sq += rm[0] * rm[0]
                for k, s in zip(KEYS, draw_acc):
                    rand_sum[eps][k] += s / R                       # this sample's mean over R
                # Spread of the control across its own R draws for this sample
                # (0 when R==1); averaged over samples in the report below.
                m_epe = draw_acc[0] / R
                rand_std_sum[eps] += math.sqrt(max(epe_sq / R - m_epe * m_epe, 0.0))

        n += 1
        if args.max_chunks is not None and n >= args.max_chunks:
            break

    if n == 0:
        raise RuntimeError("No samples were evaluated; check --root / --split.")

    clean_avg = {k: clean_sum[k] / n for k in KEYS}

    # ---- Report -----------------------------------------------------------
    print(f"\nEvaluated {n} samples | attack = '{args.attack}' | R = {R}")
    print(f"Clean EPE = {clean_avg['EPE']:.4f}\n")
    hdr = (f"{'epsilon':>10}{'adv_EPE':>10}{'rand_EPE':>12}{'adv-rand':>10}"
           f"{'adv-clean':>11}{'drift':>12}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for eps in epsilons:
        adv_avg = {k: adv_sum[eps][k] / n for k in KEYS}
        if control is not None:
            rand_avg = {k: rand_sum[eps][k] / n for k in KEYS}
            rand_std = rand_std_sum[eps] / n
        else:
            rand_avg = {k: float("nan") for k in KEYS}
            rand_std = float("nan")

        adv_vs_rand = adv_avg["EPE"] - rand_avg["EPE"]
        print(f"{eps:>10.4g}{adv_avg['EPE']:>10.4f}"
              f"{rand_avg['EPE']:>9.4f}+-{rand_std:<0.3f}"
              f"{adv_vs_rand:>+10.4f}{adv_avg['EPE'] - clean_avg['EPE']:>+11.4f}"
              f"{drift[eps]:>12.4g}")

        rows.append({
            "epsilon": eps,
            "clean_EPE": clean_avg["EPE"],
            "adv_EPE": adv_avg["EPE"],
            "rand_EPE_mean": rand_avg["EPE"],
            "rand_EPE_std": rand_std,
            "adv_minus_rand_EPE": adv_vs_rand,
            "delta_adv_EPE": adv_avg["EPE"] - clean_avg["EPE"],
            "delta_rand_EPE": rand_avg["EPE"] - clean_avg["EPE"],
            "adv_angular_deg": adv_avg["angular_deg"],
            "adv_one_minus_cos": adv_avg["one_minus_cos"],
            "rand_angular_deg": rand_avg["angular_deg"],
            "rand_one_minus_cos": rand_avg["one_minus_cos"],
            "clean_angular_deg": clean_avg["angular_deg"],
            "clean_one_minus_cos": clean_avg["one_minus_cos"],
            "max_count_drift": drift[eps],
            "n_samples": n,
            "R": R,
        })

    # ---- CSV --------------------------------------------------------------
    csv_path = os.path.join(args.outdir, f"sweep_{args.attack}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSweep written to {csv_path}")


if __name__ == "__main__":
    main()
