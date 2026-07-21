"""FGSM / PGD attacks on the event-based optical-flow SNN.

Reproduces the attack technique of Sharmin et al., "Inherent Adversarial
Robustness of Deep Spiking Neural Networks" (arXiv:2003.10399): white-box,
surrogate-gradient BPTT perturbation generation with the classic FGSM
single-step / PGD iterative-projected-step structure. Everything else is
changed to fit this repo: the target is the spiking
encoder-decoder (``network_3d.poolingNet_cat_1res.NeuronPool_Separable_Pool3d``)
doing dense optical-flow regression on DSEC event tensors, instead of a classifier on
static image pixels.

No new neuron model or surrogate gradient is introduced here -- the gradient
that FGSM/PGD needs is produced entirely by the model's own
``spikingjelly.clock_driven.neuron.IFNode`` units (hard reset,
``v_threshold=1.0``, ``v_reset=0.0``) and the default
``surrogate.Sigmoid(alpha=4.0)`` backward pass.

The perturbation space is an additive L-infinity ball of radius ``epsilon``
defined directly on the model-input event-count tensor ``chunk``
(``[B, C, T, H, W]``, non-negative integer-valued counts), rather than the
usual pixel-intensity ``eps/255`` convention. Because event counts cannot be
negative, ``clip_min=0`` is enforced after every perturbation step; no upper
clamp is applied since the clean data itself is already unbounded above (see
``attacks/calibrate_epsilon.py`` for how to pick a data-grounded ``epsilon``).

The objective maximised is untargeted dense flow error -- by default the
repo's EPE (``eval.vector_loss_functions.mod_loss_function``), with
``angular``/``cosine`` available as alternates -- not a classification
fooling objective.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from spikingjelly.clock_driven import functional

from eval.vector_loss_functions import (
    mod_loss_function,
    angular_loss_function,
    cosine_loss_function,
)

from ..base import EventThreat, register_threat


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
    """Temporarily set every model parameter's ``requires_grad`` to False.

    Context manager which temporarily sets every parameter's ``requires_grad`` to False, 
    then restores the original values on exit. This is used to ensure that the model's 
    parameters are not updated during the attack perturbation process, 
    which is important for white-box attacks like FGSM and PGD where we want to compute gradients
    with respect to the input, not the model parameters.
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
    """Shared ``verify_constraint`` body for FGSMAttack/PGDAttack.

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


@register_threat("fgsm")
class FGSMAttack(EventThreat):
    """Single-step white-box FGSM on the raw event-count tensor.

    Parameters
    ----------
    epsilon : float
        L-infinity perturbation budget, in event-count units.
    loss : {"epe", "angular", "cosine"}
        Objective to maximise (default: EPE -- the repo's flow endpoint error).
    clip_min : float
        Lower clamp applied after perturbing (event counts can't be negative).
    record_history : bool
        If True, ``self.history`` is populated with ``[(loss, grad_linf)]``
        (a single entry for FGSM) for diagnostics.
    """

    name = "fgsm"

    def __init__(self, epsilon: float = 1.0, loss: str = "epe",
                 clip_min: float = 0.0, record_history: bool = False, **kw):
        super().__init__(epsilon=epsilon, loss=loss, clip_min=clip_min,
                         record_history=record_history, **kw)
        self.epsilon = float(epsilon)
        self.loss_name = loss
        self.clip_min = float(clip_min)

    def perturb(self, chunk, *, model=None, label=None, mask=None):
        if model is None or label is None:
            raise ValueError("FGSMAttack requires `model` and `label`.")
        if mask is None:
            mask = torch.ones_like(label[:, :1])

        loss_fn = _loss_fn(self.loss_name)
        self.history = []

        with _FreezeParams(model):
            grad, loss_val = _input_grad(model, chunk, label, mask, loss_fn)
            self._record(loss_val, grad.abs().max().item())
            adv = chunk.detach() + self.epsilon * grad.sign()
            adv = adv.clamp_(min=self.clip_min)

        return adv.detach()

    def verify_constraint(self, chunk, adv):
        return _epsilon_ball_report(chunk, adv, self.epsilon, self.clip_min)


@register_threat("pgd")
class PGDAttack(EventThreat):
    """Iterative, L-infinity-projected white-box PGD on the raw event-count tensor.

    Parameters
    ----------
    epsilon : float
        L-infinity perturbation budget, in event-count units.
    alpha : float, optional
        Per-step size. Defaults to ``epsilon / 4`` if not given.
    iters : int
        Number of PGD steps.
    rand_init : bool
        If True, start from a uniform random point inside the epsilon-ball
        (standard PGD with random restart) instead of the clean chunk.
    loss : {"epe", "angular", "cosine"}
        Objective to maximise (default: EPE).
    clip_min : float
        Lower clamp applied after every step (event counts can't be negative).
    record_history : bool
        If True, ``self.history`` collects ``(loss, grad_linf)`` per step, so
        callers can check the objective is actually rising across iterations.
    """

    name = "pgd"

    def __init__(self, epsilon: float = 1.0, alpha: Optional[float] = None,
                 iters: int = 7, rand_init: bool = True, loss: str = "epe",
                 clip_min: float = 0.0, record_history: bool = False, **kw):
        super().__init__(epsilon=epsilon, alpha=alpha, iters=iters,
                         rand_init=rand_init, loss=loss, clip_min=clip_min,
                         record_history=record_history, **kw)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha) if alpha is not None else self.epsilon / 4.0
        self.iters = int(iters)
        self.rand_init = rand_init
        self.loss_name = loss
        self.clip_min = float(clip_min)

    def perturb(self, chunk, *, model=None, label=None, mask=None):
        if model is None or label is None:
            raise ValueError("PGDAttack requires `model` and `label`.")
        if mask is None:
            mask = torch.ones_like(label[:, :1])

        loss_fn = _loss_fn(self.loss_name)
        self.history = []
        x0 = chunk.detach()

        if self.rand_init:
            noise = (torch.rand_like(x0) * 2 - 1) * self.epsilon
            x = (x0 + noise).clamp_(min=self.clip_min)
        else:
            x = x0.clone()

        with _FreezeParams(model):
            for _ in range(self.iters):
                grad, loss_val = _input_grad(model, x, label, mask, loss_fn)
                self._record(loss_val, grad.abs().max().item())
                x = x.detach() + self.alpha * grad.sign()
                x = torch.max(torch.min(x, x0 + self.epsilon), x0 - self.epsilon)
                x = x.clamp_(min=self.clip_min)

        return x.detach()

    def verify_constraint(self, chunk, adv):
        return _epsilon_ball_report(chunk, adv, self.epsilon, self.clip_min)
