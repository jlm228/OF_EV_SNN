"""Modular threat framework for an event-based optical flow SNN.

A threat pertubs the model's input event tensor ``chunk`` of shape ``[B, C, T, H, W]``.
Where:
- ``B`` is the batch size,
- ``C`` is the number of channels (2 for ON/OFF polarity),
- ``T`` is the number of temporal bins (21),
- ``H`` is the height of the input,
- ``W`` is the width of the input.

The design mirrors the existing data-augmentation pipeline (``data/data_augmentation_2d.py``)
from Cuadrado et al.: "Optical flow estimation from event-based cameras and spiking neural 
networks" (arXiv:2302.06492).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

from torch import Tensor


# Registry mapping a threat name -> EventThreat subclass.
THREAT_REGISTRY: Dict[str, Type["EventThreat"]] = {}


def register_threat(*names: str):
    """Class decorator registering an :class:`EventThreat` under one or more names."""

    def _decorator(cls: Type["EventThreat"]) -> Type["EventThreat"]:
        for name in names:
            key = name.lower()
            existing = THREAT_REGISTRY.get(key)
            # Re-registration by the same class is fine (e.g. re-imported via
            # ``python -m``); a different class reusing the name is an error.
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
        # Kept verbatim so threats are self-describing / reproducible.
        self.config = config
        # Per-iteration (loss, grad_metric) trace, filled by `_record` when
        # `record_history=True`. Empty on non-iterative threats, so callers can
        # use `len(attack.history)` instead of `hasattr` guards.
        self.history: List[Tuple[float, Optional[float]]] = []

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

    def verify_constraint(self, chunk: Tensor, adv: Tensor) -> dict:
        """Self-report whether ``adv`` respects this threat's own invariant.

        Default: no invariant, returns ``{}``. Threats with a formal constraint
        (e.g. an L-infinity budget) should override this to return at least
        ``{"passed": bool, "description": str}`` plus any family-specific
        diagnostics.
        """
        return {}

    def _record(self, loss: float, grad_metric: Optional[float] = None) -> None:
        """Append one ``(loss, grad_metric)`` entry to ``self.history``.

        No-op unless built with ``record_history=True``. ``grad_metric`` is
        threat-specific in meaning and scale, so don't compare it across threat
        classes; only loss trajectories are portable.
        """
        if self.config.get("record_history"):
            self.history.append((loss, grad_metric))

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

    ``None`` / ``"none"`` / ``"clean"`` yields the no-op :class:`IdentityThreat`.
    Unknown config keys are just stored, so callers may pass a superset.
    """
    key = "none" if name is None else str(name).lower()
    if key not in THREAT_REGISTRY:
        available = ", ".join(sorted(THREAT_REGISTRY))
        raise KeyError(f"Unknown threat '{name}'. Available: {available}")
    return THREAT_REGISTRY[key](**cfg)
