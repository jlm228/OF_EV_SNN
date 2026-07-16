"""Modular cybersecurity-threat suite for the event-based optical-flow SNN.

Public API::

    from attacks import build_attack, EventThreat, THREAT_REGISTRY

    attack = build_attack("retiming_blackbox", budget=2)
    adv_chunk = attack(chunk, model=net, label=label, mask=mask)

Importing this package registers the built-in threats (see ``THREAT_REGISTRY``).
"""

from .base import EventThreat, THREAT_REGISTRY, build_attack, register_threat

# Import for the side effect of registering the threats in THREAT_REGISTRY.
from . import spike_retiming  # noqa: F401
from .spike_retiming import BlackBoxRetiming, PILRetimingAttack

__all__ = [
    "EventThreat",
    "THREAT_REGISTRY",
    "build_attack",
    "register_threat",
    "BlackBoxRetiming",
    "PILRetimingAttack",
]
