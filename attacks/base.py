"""Modular cybersecurity-threat framework for the event-based optical-flow SNN.

A *threat* perturbs the model-input event tensor ``chunk`` of shape
``[B, C, T, H, W]`` (i.e. after the ``torch.transpose(chunk, 1, 2)`` performed by
the eval/train loops, so the **time axis is dim 2**, ``C = 2`` ON/OFF polarity
channels, ``T = 21`` temporal bins).

The design mirrors the existing data-augmentation pipeline
(``data/data_augmentation_2d.py``): each threat is a small callable class.  It is
kept as a dedicated base class rather than a ``torchvision`` transform because
white-box threats additionally need access to the ``model`` and a ``label`` to
optimise against.

Adding a new threat is intentionally a one-file change::

    from attacks.base import EventThreat, register_threat

    @register_threat("my_threat")
    class MyThreat(EventThreat):
        def perturb(self, chunk, *, model=None, label=None, mask=None):
            ...
            return adv_chunk

...then reference it by name via ``build_attack("my_threat", **cfg)``.  No change
to the network, dataset, or evaluation loop is required.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from torch import Tensor


# Registry mapping a threat name -> EventThreat subclass.
THREAT_REGISTRY: Dict[str, Type["EventThreat"]] = {}


def register_threat(*names: str):
    """Class decorator registering an :class:`EventThreat` under one or more names."""

    def _decorator(cls: Type["EventThreat"]) -> Type["EventThreat"]:
        for name in names:
            key = name.lower()
            existing = THREAT_REGISTRY.get(key)
            # Allow re-registration by the same class (e.g. a module re-imported
            # under both its package name and __main__ via ``python -m``); only a
            # genuinely different threat class reusing the name is an error.
            if existing is not None and existing.__name__ != cls.__name__:
                raise KeyError(
                    f"Threat name '{name}' is already registered to "
                    f"{existing.__name__}."
                )
            THREAT_REGISTRY[key] = cls
        return cls

    return _decorator


class EventThreat(ABC):
    """Base class for a threat acting on the ``[B, C, T, H, W]`` event tensor."""

    #: Human-readable identifier (overridden by subclasses).
    name: str = "base"

    def __init__(self, **config):
        # Store the config verbatim so threats are self-describing / reproducible.
        self.config = config

    @abstractmethod
    def perturb(
        self,
        chunk: Tensor,
        *,
        model=None,
        label: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return an adversarial copy of ``chunk`` (same shape/layout).

        Parameters
        ----------
        chunk : Tensor
            Event counts, shape ``[B, C, T, H, W]`` (time = dim 2).
        model : nn.Module, optional
            Only needed by white-box / model-aware threats.
        label, mask : Tensor, optional
            Ground-truth flow ``[B, 2, H, W]`` and validity mask; needed by
            optimisation-based threats.
        """

    # Callable so a threat can be dropped into a pipeline like a transform.
    def __call__(self, chunk, *, model=None, label=None, mask=None) -> Tensor:
        return self.perturb(chunk, model=model, label=label, mask=mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"


@register_threat("none", "clean", "identity")
class IdentityThreat(EventThreat):
    """No-op threat, so the evaluation harness can treat 'clean' uniformly."""

    name = "identity"

    def perturb(self, chunk, *, model=None, label=None, mask=None):
        return chunk


def build_attack(name: Optional[str], **cfg) -> EventThreat:
    """Instantiate a registered threat by name.

    ``name`` of ``None`` / ``"none"`` / ``"clean"`` yields the no-op
    :class:`IdentityThreat`.  Unknown config keys are simply stored on the
    threat, so callers may pass a superset of options without error.
    """
    key = "none" if name is None else str(name).lower()
    if key not in THREAT_REGISTRY:
        available = ", ".join(sorted(THREAT_REGISTRY))
        raise KeyError(f"Unknown threat '{name}'. Available: {available}")
    return THREAT_REGISTRY[key](**cfg)
