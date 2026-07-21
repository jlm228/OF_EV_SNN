"""Modular cybersecurity-threat suite for the event-based optical-flow SNN.

Public API::

    from attacks import build_attack, EventThreat, THREAT_REGISTRY

    attack = build_attack("retiming_blackbox", budget=2)
    adv_chunk = attack(chunk, model=net, label=label, mask=mask)

Importing this package registers the built-in threats (see ``THREAT_REGISTRY``).
Each attack family lives in its own subpackage:

- ``attacks.retiming`` -- timing-only, rate-preserving spike-retiming attacks.
- ``attacks.fgsm_pgd`` -- additive L-infinity FGSM/PGD attacks on event counts.
"""

from .base import EventThreat, THREAT_REGISTRY, build_attack, register_threat

# Import for the side effect of registering the threats in THREAT_REGISTRY.
from . import retiming  # noqa: F401
from .retiming import BlackBoxRetiming, PILRetimingAttack
from . import fgsm_pgd  # noqa: F401
from .fgsm_pgd import FGSMAttack, PGDAttack

__all__ = [
    "EventThreat",
    "THREAT_REGISTRY",
    "build_attack",
    "register_threat",
    "BlackBoxRetiming",
    "PILRetimingAttack",
    "FGSMAttack",
    "PGDAttack",
]
