# `attacks/` — modular cybersecurity threat suite

Threats that perturb the event input to the optical-flow SNN. Each attack *family* lives in its own subpackage, but all of them plug into the same `EventThreat` registry without touching the network, dataset, or evaluation loop.

## Layout

```
attacks/
  base.py               EventThreat ABC, @register_threat, build_attack registry
  cli_common.py          shared CLI flags + model/data loading for the scripts below
  retiming/              timing-only, rate-preserving spike-retiming attacks
    spike_retiming.py    BlackBoxRetiming, PILRetimingAttack
  fgsm_pgd/               additive L-infinity attacks on the raw event-count tensor
    _common.py            common functions for attacks in this module
    calibrate_epsilon.py  script for picking an epsilon budgetm from the input data
    fgsm.py               FGSMAttack
    pdg.py                PDGAttack
  compare_easy_hard.py   generic clean/attacked comparison across two conditions
```

## Concepts

A **threat** subclasses `EventThreat` and implements `perturb(...)`, which takes
the model-input event tensor and returns an adversarial copy of the same shape.

- **Tensor layout:** `chunk` has shape `[B, C, T, H, W]` — this is the tensor
  *after* the `torch.transpose(chunk, 1, 2)` that the eval/train loops perform, so
  the **time axis is dim 2**, `C = 2` (ON/OFF polarity), `T = 21` bins.
- **Black-box vs white-box:** `perturb` receives optional `model`, `label`, `mask`.
  Model-agnostic threats ignore them; optimisation-based threats use them.

## Built-in threats

| name (`build_attack`)   | class                 | family      | notes |
|-------------------------|-----------------------|-------------|-------|
| `none` / `clean`        | `IdentityThreat`      | `base`      | no-op baseline |
| `retiming_blackbox`     | `BlackBoxRetiming`    | `spike_retiming`  | model-agnostic sampled temporal shift (rate-preserving) |
| `retiming_pil`          | `PILRetimingAttack`   | `spike_retiming`  | white-box projected-in-the-loop optimisation |
| `fgsm`                  | `FGSMAttack`          | `fgsm_pgd`  | white-box single-step L-infinity attack, maximises EPE |
| `pgd`                   | `PGDAttack`           | `fgsm_pgd`  | white-box iterative, projected L-infinity attack |

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

1. Either add to an existing family (e.g. a new module inside `retiming/` or
   `fgsm_pgd/`) or create a new subpackage `attacks/<family>/` for a genuinely
   new attack family, with its own `__init__.py` re-exporting its classes
   (mirror `attacks/fgsm_pgd/__init__.py`).
2. Subclass `EventThreat` (import it via `from ..base import EventThreat,
   register_threat` if nested one level deep) and decorate with
   `@register_threat("my_threat")`.
3. Implement `perturb(self, chunk, *, model=None, label=None, mask=None)` and
   return a perturbed `[B, C, T, H, W]` tensor.
4. Import the new subpackage/module in `attacks/__init__.py` so the
   registration runs (mirror the existing `retiming`/`fgsm_pgd` imports).
5. Select it anywhere with `build_attack("my_threat", **cfg)` — including
   `python evaluate_attack.py --attack my_threat`.
6. Optional self-reporting hooks: if the threat is gradient-based/iterative,
   accept a `record_history: bool = False` constructor kwarg (pass it to
   `super().__init__(...)` like the other kwargs) and call
   `self._record(loss, grad_metric)` once per optimisation step -- record the
   *true* loss being maximised, not a sign-flipped/regularised objective. If
   it has a formal invariant (a budget, a preserved quantity, ...), override
   `verify_constraint(self, chunk, adv) -> dict` and return at least
   `{"passed": bool, "description": str}`. Neither is required.

```python
from attacks.base import EventThreat, register_threat

@register_threat("event_injection")
class EventInjection(EventThreat):
    def perturb(self, chunk, *, model=None, label=None, mask=None):
        ...  # e.g. add spurious counts
        return adv_chunk
```

## Reference

- Spike-retiming method re-implemented from Yu et al., *"Time Is All It Takes:
Spike-Retiming Attacks on Event-Driven Spiking Neural Networks"* (ICLR 2026). Only
the method (threat model + projected-in-the-loop optimisation) is reused; no code
from the reference repository is copied.
- FGSM/PGD attack method re-implemented from Sharmin et al., *"Inherent Adversarial Robustness of Deep Spiking Neural Networks: Effects of Discrete Input Encoding and Non-Linear Activations"* (ECCV 2020). The concept and Algorithm 1 for FGSM is retargeted from static-image classification to
the dense flow regression on event tensors used with this SNN. No source code from https://github.com/ssharmin/spikingNN-adversarial-attack was re-used.

