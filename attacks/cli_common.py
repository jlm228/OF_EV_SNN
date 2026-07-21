"""Shared CLI surface and model/data loading for attack-facing scripts.

``evaluate_attack.py`` needs to: (1) pick and configure an attack from the
``attacks`` registry, and (2) load the trained checkpoint and DSEC validation
split. Factoring that out here keeps attack-facing tools' flags identical by
construction instead of by copy-paste discipline.
"""

import argparse

import torch

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d
from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite

from .base import build_attack


def add_common_attack_args(p: argparse.ArgumentParser) -> None:
    """Flags that configure *which* attack and how strong it is."""
    p.add_argument("--attack", default="none",
                   help="Threat name (e.g. none, retiming_blackbox, retiming_pil, fgsm, pgd).")
    p.add_argument("--budget", type=int, default=2, help="Max temporal shift in bins (retiming).")
    p.add_argument("--granularity", default=None,
                   help="pixel | block | global (default: threat's own default).")
    p.add_argument("--mode", default="random",
                   help="Black-box sampling mode: random | worst_of_n.")
    p.add_argument("--n-samples", type=int, default=8, help="Candidates for worst_of_n.")
    p.add_argument("--iters", type=int, default=10,
                   help="PIL optimisation steps / PGD steps.")
    p.add_argument("--loss", default=None,
                   help="Attack objective. retiming_pil: angular | cosine "
                        "(default angular). fgsm/pgd: epe | angular | cosine "
                        "(default epe). Leave unset to use each attack's own default.")
    p.add_argument("--seed", type=int, default=2305, help="Attack sampling seed.")
    p.add_argument("--epsilon", type=float, default=None,
                   help="fgsm/pgd: L-infinity budget, in event-count units.")
    p.add_argument("--alpha", type=float, default=None,
                   help="pgd: per-step size (default epsilon / 4).")
    p.add_argument("--rand-init", action="store_true",
                   help="pgd: start from a random point inside the epsilon-ball.")


def add_common_model_args(p: argparse.ArgumentParser) -> None:
    """Flags that configure which checkpoint/dataset split to load."""
    p.add_argument("--root", default="data/dataset/saved_flow_data",
                   help="Dataset root (relative path).")
    p.add_argument("--split", default="valid_split_thun_00_a.csv", help="Sequence list CSV.")
    p.add_argument("--num-frames-per-ts", type=int, default=11)
    p.add_argument("--checkpoint", default="examples/checkpoint_epoch34.pth")
    p.add_argument("--multiply-factor", type=float, default=35.0)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Limit number of validation samples (useful on CPU).")


def build_threat_from_args(args, **extra_cfg):
    """Assemble threat kwargs, only overriding optional fields when the user set them.

    ``loss``/``epsilon``/``alpha``/``rand_init`` are only injected when
    explicitly provided so each attack class keeps its own default (e.g.
    retiming_pil defaults to the "angular" objective, fgsm/pgd default to "epe").
    ``extra_cfg`` (e.g. ``record_history=True``) is merged in on top.
    """
    cfg = dict(budget=args.budget, mode=args.mode, n_samples=args.n_samples,
               iters=args.iters, seed=args.seed)
    if args.granularity is not None:
        cfg["granularity"] = args.granularity
    if args.loss is not None:
        cfg["loss"] = args.loss
    if args.epsilon is not None:
        cfg["epsilon"] = args.epsilon
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.rand_init:
        cfg["rand_init"] = True
    cfg.update(extra_cfg)
    return build_attack(args.attack, **cfg)


def load_model_and_data(args):
    """Load the checkpoint + DSEC validation split shared by both CLI tools.

    Returns ``(device, net, dataset, loader)``.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    dataset = DSECDatasetLite(root=args.root, file_list=args.split,
                              num_frames_per_ts=args.num_frames_per_ts,
                              stereo=False, transform=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         drop_last=False, pin_memory=True)

    net = NeuronPool_Separable_Pool3d(multiply_factor=args.multiply_factor).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    return device, net, dataset, loader
