# `attacks/` — modular cybersecurity threat suite

Threats that perturb the event input to the optical-flow SNN, for robustness /
security analysis. The first threat implemented is a **spike-retiming attack**
(timing-only, rate-preserving), but the framework is built so other threats plug
in without touching the network, dataset, or evaluation loop.

## Concepts

A **threat** subclasses `EventThreat` and implements `perturb(...)`, which takes
the model-input event tensor and returns an adversarial copy of the same shape.

- **Tensor layout:** `chunk` has shape `[B, C, T, H, W]` — this is the tensor
  *after* the `torch.transpose(chunk, 1, 2)` that the eval/train loops perform, so
  the **time axis is dim 2**, `C = 2` (ON/OFF polarity), `T = 21` bins.
- **Black-box vs white-box:** `perturb` receives optional `model`, `label`, `mask`.
  Model-agnostic threats ignore them; optimisation-based threats use them.

## Built-in threats

| name (`build_attack`)   | class                 | notes |
|-------------------------|-----------------------|-------|
| `none` / `clean`        | `IdentityThreat`      | no-op baseline |
| `retiming_blackbox`     | `BlackBoxRetiming`    | model-agnostic sampled temporal shift (rate-preserving) |
| `retiming_pil`          | `PILRetimingAttack`   | white-box projected-in-the-loop optimisation |

## Usage

```python
from attacks import build_attack

attack = build_attack("retiming_blackbox", budget=2, granularity="pixel")
adv_chunk = attack(chunk, model=net, label=label, mask=mask)
```

Or from the command line via the evaluation harness:

```bash
python evaluate_attack.py --attack retiming_blackbox --budget 2 --visualize
```

## Adding a new threat

1. Create a class in a new module (or in an existing one), subclassing
   `EventThreat` and decorating it with `@register_threat("my_threat")`.
2. Implement `perturb(self, chunk, *, model=None, label=None, mask=None)` and
   return a perturbed `[B, C, T, H, W]` tensor.
3. If it lives in a new module, import that module in `attacks/__init__.py` so the
   registration runs.
4. Select it anywhere with `build_attack("my_threat", **cfg)` — including
   `python evaluate_attack.py --attack my_threat`.

```python
from attacks.base import EventThreat, register_threat

@register_threat("event_injection")
class EventInjection(EventThreat):
    def perturb(self, chunk, *, model=None, label=None, mask=None):
        ...  # e.g. add spurious counts
        return adv_chunk
```

## Reference

Spike-retiming method re-implemented from Yu et al., *"Time Is All It Takes:
Spike-Retiming Attacks on Event-Driven Spiking Neural Networks"* (ICLR 2026). Only
the method (threat model + projected-in-the-loop optimisation) is reused; no code
from the reference repository is copied.
