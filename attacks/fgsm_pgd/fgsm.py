"""FGSM attack for event-based optical flow SNNs.

Reproduces the attack technique of Sharmin et al., "Inherent Adversarial
Robustness of Deep Spiking Neural Networks" (arXiv:2003.10399). The original was 
designed for image classification tasks; here it is adapated for optical flow
regression tasks.

Uses shared functions from _common.py.
"""

import torch

from ..base import EventThreat, register_threat
from ._common import _loss_fn, _FreezeParams, _input_grad, _epsilon_ball_report


@register_threat("fgsm")
class FGSMAttack(EventThreat):
    """Single-step white-box FGSM on the raw event-count tensor.

    Parameters
    ----------
    epsilon : float
        L-infinity perturbation budget, in event-count units.
    loss : {"epe", "angular", "cosine"}
        Objective to maximise (default: EPE).
    clip_min : float
        Lower clamp applied after perturbing (as event counts can't be negative).
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
        """Pertubations applied to the model input ``chunk`` - an event-count
        tensor of shape [B, C, T, H, W]. The perturbation is computed by taking
        the sign of the input gradient and stepping in that direction, then
        clamping to ``clip_min`` so event counts stay non-negative."""

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
