"""FGSM/PGD attack family (additive L-infinity perturbations on event counts).

See ``attack.py`` for the attack classes and ``calibrate_epsilon.py`` for
picking a data-grounded epsilon budget.
"""

from . import attack  # noqa: F401
from .attack import FGSMAttack, PGDAttack

__all__ = ["FGSMAttack", "PGDAttack"]
