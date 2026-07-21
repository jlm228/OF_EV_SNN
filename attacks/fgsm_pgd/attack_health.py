"""Attack-health diagnostic for fgsm/pgd: confirm the adversarial loss rises.

A weak attack effect on a deep temporal SNN can mean two very different
things: (a) the model is genuinely robust, or (b) the surrogate gradient is
vanishing/saturating and the attack is silently failing. This script tells
them apart by running FGSM (one step) and PGD (``--iters`` steps) at the same
``epsilon`` on a handful of validation samples, recording the loss and
input-gradient L-infinity norm at every PGD step, and checking that:

1. PGD's per-step loss is (broadly) non-decreasing -- the objective is
   actually climbing, not flatlining after step 1.
2. PGD's final loss is >= FGSM's one-step loss at the same epsilon -- extra
   iterations should never leave you worse off than a single step.

Per-sample loss-vs-iteration curves are saved to
``results/attack_health/sample_<i>.png``; a pass/fail summary is printed.

Usage::

    python -m attacks.fgsm_pgd.attack_health --epsilon 2.0 --n-samples 5
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d
from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite
from .attack import FGSMAttack, PGDAttack


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epsilon", type=float, required=True,
                   help="L-infinity budget in event-count units (see calibrate_epsilon.py).")
    p.add_argument("--alpha", type=float, default=None, help="PGD step size (default epsilon/4).")
    p.add_argument("--iters", type=int, default=10, help="PGD steps.")
    p.add_argument("--loss", default="epe", choices=["epe", "angular", "cosine"])
    p.add_argument("--root", default="data/dataset/saved_flow_data")
    p.add_argument("--split", default="valid_split_thun_00_a.csv")
    p.add_argument("--num-frames-per-ts", type=int, default=11)
    p.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    p.add_argument("--multiply-factor", type=float, default=35.0)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--outdir", default="results/attack_health")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    dataset = DSECDatasetLite(root=args.root, file_list=args.split,
                              num_frames_per_ts=args.num_frames_per_ts,
                              stereo=False, transform=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         drop_last=False, pin_memory=True)

    net = NeuronPool_Separable_Pool3d(multiply_factor=args.multiply_factor).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    fgsm = FGSMAttack(epsilon=args.epsilon, loss=args.loss, record_history=True)
    pgd = PGDAttack(epsilon=args.epsilon, alpha=args.alpha, iters=args.iters,
                    loss=args.loss, rand_init=False, record_history=True)

    n_pass = 0
    n_total = 0
    print(f"{'sample':>8}{'fgsm_loss':>12}{'pgd_final':>12}{'pgd_monotone':>14}{'pgd>=fgsm':>12}")

    for chunk, mask, label in tqdm(loader, desc="Attack health", total=min(args.n_samples, len(dataset))):
        if n_total >= args.n_samples:
            break

        chunk = torch.transpose(chunk, 1, 2)
        mask = torch.unsqueeze(mask, dim=1)
        chunk = chunk.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device)

        fgsm.perturb(chunk, model=net, label=label, mask=mask)
        pgd.perturb(chunk, model=net, label=label, mask=mask)

        fgsm_loss = fgsm.history[0][0]
        pgd_losses = [l for l, _g in pgd.history]
        pgd_final = pgd_losses[-1]

        # "Broadly non-decreasing": allow small numerical wobble, flag a real regression.
        monotone = all(b >= a - 1e-3 for a, b in zip(pgd_losses, pgd_losses[1:]))
        strengthens = pgd_final >= fgsm_loss - 1e-3

        n_total += 1
        n_pass += int(monotone and strengthens)
        print(f"{n_total:>8}{fgsm_loss:>12.4f}{pgd_final:>12.4f}"
              f"{str(monotone):>14}{str(strengthens):>12}")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(range(1, len(pgd_losses) + 1), pgd_losses, marker="o", label="PGD")
        ax.axhline(fgsm_loss, color="tab:red", linestyle="--", label="FGSM (1 step)")
        ax.set_xlabel("PGD iteration")
        ax.set_ylabel(f"{args.loss} loss (maximised)")
        ax.set_title(f"Sample {n_total} | epsilon={args.epsilon}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"sample_{n_total}.png"))
        plt.close(fig)

    print(f"\n{n_pass}/{n_total} samples passed the attack-health check "
          f"(PGD loss rises monotonically and PGD final loss >= FGSM loss).")
    if n_pass < n_total:
        print("WARNING: some samples failed -- this can indicate vanishing/"
              "saturating surrogate gradients rather than genuine model robustness.")


if __name__ == "__main__":
    main()
