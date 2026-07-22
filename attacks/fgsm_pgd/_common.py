"""Shared functions for the FGSM/PGD event-count attacks.

Both `FGSMAttack` and `PGDAttack`, in their pertub functions, need the same loss lookup, 
a way to freeze  the model's parameters, an input-gradient step, and an L-infinity-ball 
constraint report. 

See `fgsm.py` and `pgd.py` for the attack classes and the standalone script `calibrate_epsilon.py`
for picking a data-grounded epsilon budget.
"""

from typing import List, Tuple

import torch
from torch import Tensor

from spikingjelly.clock_driven import functional

from eval.vector_loss_functions import (
    mod_loss_function,
    angular_loss_function,
    cosine_loss_function,
)


_LOSS_FNS = {
    "epe": mod_loss_function,
    "angular": angular_loss_function,
    "cosine": cosine_loss_function,
}


def _loss_fn(name: str):
    try:
        return _LOSS_FNS[name]
    except KeyError:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {', '.join(sorted(_LOSS_FNS))}."
        )


class _FreezeParams:
    """Temporarily set every model parameter's `requires_grad` to False.

    Context manager which temporarily sets every parameter's `requires_grad` to False,
    then restores the original values on exit. 
    
    This is used to ensure that the model's parameters are not updated during the attack 
    perturbation process, which is important for white-box attacks like FGSM and PGD 
    where we want to compute gradients with respect to the input, not the model parameters.
    """

    def __init__(self, model):
        self.model = model
        self._prev: List[bool] = []

    def __enter__(self):
        for p in self.model.parameters():
            self._prev.append(p.requires_grad)
            p.requires_grad_(False)
        return self.model

    def __exit__(self, exc_type, exc, tb):
        for p, prev in zip(self.model.parameters(), self._prev):
            p.requires_grad_(prev)
        return False


def _input_grad(model, x: Tensor, label: Tensor, mask: Tensor, loss_fn) -> Tuple[Tensor, float]:
    """One forward/backward pass; returns (grad w.r.t. x, loss value)."""

    x = x.detach().clone().requires_grad_(True)
    functional.reset_net(model)
    pred = model(x)[-1]
    loss = loss_fn(pred, label, mask)
    loss.backward()
    grad = x.grad.detach()
    return grad, loss.item()


def _epsilon_ball_report(chunk: Tensor, adv: Tensor, epsilon: float, clip_min: float,
                         tol: float = 1e-3) -> dict:
    """Shared `verify_constraint` body for FGSMAttack and PGDAttack.

    Checks the perturbation stayed within the L-infinity epsilon-ball of the
    clean input and that event counts remain non-negative.
    """

    max_abs_delta = (adv - chunk).abs().max().item()
    min_value = adv.min().item()
    passed = (max_abs_delta <= epsilon + tol) and (min_value >= clip_min - tol)
    return {
        "passed": passed,
        "description": "L-infinity ball + non-negativity",
        "max_abs_delta": max_abs_delta,
        "epsilon": epsilon,
        "min_value": min_value,
        "clip_min": clip_min,
    }
