"""PGD attack for event-based optical flow SNNs.

Iterative, L-infinity-projected variant of the FGSM attack in
``fgsm.py``, also from (Sharmin et al., arXiv:2003.10399). Adapted to
dense optical-flow regression on DSEC event tensors. 

Uses shared functions from _common.py.
"""

from typing import Optional

import torch

from ..base import EventThreat, register_threat
from ._common import _loss_fn, _FreezeParams, _input_grad, _epsilon_ball_report


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
        Lower clamp applied after every step (as event counts can't be negative).
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
        """Pertubations applied to the model input ``chunk`` - an event-count
        tensor of shape [B, C, T, H, W]. The pertubation is computed in the 
        same way as FGSM, but instead, iteratively take the sign of the input
        gradient and step in that direction, then project back into the L-infinity
        epsilon-ball and clamp to ``clip_min`` so event counts stay non-negative."""

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
