"""Random-sign control for the additive L-infinity attacks.

A validation technique for FGSM/PGD algorithms. This is used to answer the question of
whether the *gradient* sign does better than an arbitrary sign of the *same* L-infinity.

This threat draws a per-voxel sign ``S ~ Uniform{-1, +1}`` and applies the
identical signed step

    E_rand = clamp(E + epsilon * S, 0, inf)

Averaging over ``R`` independent draws is an *evaluation* concern: each
``perturb`` call is one draw (the internal generator advances), so the caller
loops ``R`` times. See ``sweep_epsilon.py``.
"""

from typing import Optional

import torch

from ..base import EventThreat, register_threat
from ._common import _epsilon_ball_report


@register_threat("random_sign", "rand_sign", "control")
class RandomSignAttack(EventThreat):
    """Magnitude-matched random-sign control on the raw event-count tensor.

    Parameters
    ----------
    epsilon : float
        L-infinity step magnitude, in event-count units (matched to FGSM/PGD).
    clip_min : float
        Lower clamp applied after perturbing (event counts can't be negative).
    seed : int, optional
        Seed for the internal (CPU) generator; makes the sequence of draws
        reproducible. ``None`` leaves the generator unseeded.
    """

    name = "random_sign"

    def __init__(self, epsilon: float = 1.0, clip_min: float = 0.0,
                 seed: Optional[int] = None, **kw):
        super().__init__(epsilon=epsilon, clip_min=clip_min, seed=seed, **kw)
        self.epsilon = float(epsilon)
        self.clip_min = float(clip_min)
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def perturb(self, E, *, model=None, label=None, M=None):
        """Draw one sign tensor ``S ~ Uniform{-1, +1}`` and step by ``epsilon * S``.

        ``model``/``label``/``M`` are ignored (this control is model-agnostic);
        they are accepted only so it plugs into the same harness as FGSM/PGD.
        """
        E = E.detach()
        # Sample on CPU (Generator is CPU-bound) then move to E's device.
        signs = torch.randint(0, 2, E.shape, generator=self._gen).mul_(2).sub_(1)
        S = signs.to(device=E.device, dtype=E.dtype)
        E_rand = E + self.epsilon * S
        E_rand = E_rand.clamp_(min=self.clip_min)
        return E_rand

    def verify_constraint(self, E, E_rand):
        return _epsilon_ball_report(E, E_rand, self.epsilon, self.clip_min)
