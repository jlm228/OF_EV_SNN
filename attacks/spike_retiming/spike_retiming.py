"""Spike-retiming attacks on the event-based optical-flow SNN.

A spike-retiming attack perturbs *only the timing* of events: it moves event
counts between temporal bins while leaving each pixel's total count and spatial
location untouched.  Because optical flow is (displacement / dt), retiming
directly corrupts the estimated velocity field -- yet the perturbation is
**rate-preserving** (per-pixel counts unchanged) and **spatially faithful**, so
it is invisible to count/rate-based anomaly monitors and to DVS noise filters.

Method re-implemented from Yu et al., "Time Is All It Takes: Spike-Retiming
Attacks on Event-Driven Spiking Neural Networks" (ICLR 2026).  Only the *method*
(threat model + projected-in-the-loop optimisation) is reused; no code from the
reference repository is copied.

Two threats are provided:

* :class:`BlackBoxRetiming` -- model-agnostic.  Samples a discrete per-pixel /
  per-block temporal shift within a budget and applies it.  Fast, no gradients.
* :class:`PILRetimingAttack` -- white-box "projected-in-the-loop".  Optimises a
  differentiable soft-retiming (softmax over candidate shifts) to maximise the
  flow error, then projects to a feasible discrete schedule.

Both operate on the model-input tensor ``[B, C, T, H, W]`` (time = dim 2).

Run ``python -m attacks.retiming.spike_retiming`` for a self-test.
"""

import math
from typing import List, Optional

import torch
from torch import Tensor

from spikingjelly.clock_driven import functional

from eval.vector_loss_functions import angular_loss_function, cosine_loss_function

from ..base import EventThreat, register_threat


# ---------------------------------------------------------------------------
# Core retiming primitives (both are count-preserving = rate-preserving)
# ---------------------------------------------------------------------------

def retime_counts(chunk: Tensor, delta_map: Tensor) -> Tensor:
    """Shift event counts along time by a per-pixel integer schedule.

    ``delta_map`` has shape ``[B, C, H, W]`` (integer bins; positive = later in
    time).  Counts that would leave the ``[0, T-1]`` window are clamped into the
    boundary bin, so the total count at every pixel is exactly preserved.

    Implemented as a scatter-add along the time axis, which supports an arbitrary
    (per-pixel, per-polarity) shift in a single vectorised op.
    """
    B, C, T, H, W = chunk.shape
    t_idx = torch.arange(T, device=chunk.device).view(1, 1, T, 1, 1)
    dest = (t_idx + delta_map.long().unsqueeze(2)).clamp_(0, T - 1)  # [B,C,T,H,W]
    out = torch.zeros_like(chunk)
    out.scatter_add_(2, dest, chunk)
    return out


def global_soft_shift(chunk: Tensor, delta: int, preserve_counts: bool = True) -> Tensor:
    """Differentiable rigid temporal shift of the whole tensor by ``delta`` bins.

    Positive ``delta`` shifts events later in time.  Overflowing counts are
    accumulated into the boundary bin (so the operation is count-preserving), and
    every op is autograd-friendly -- this is the building block for the soft
    retiming used by the white-box attack.
    """
    if delta == 0:
        return chunk

    T = chunk.shape[2]
    if abs(delta) >= T:
        summed = chunk.sum(dim=2, keepdim=True)
        out = torch.zeros_like(chunk)
        out[:, :, -1:] = summed if delta > 0 else out[:, :, -1:]
        out[:, :, :1] = summed if delta < 0 else out[:, :, :1]
        return out

    rolled = torch.roll(chunk, shifts=delta, dims=2)
    out = rolled.clone()
    if delta > 0:
        overflow = rolled[:, :, :delta].sum(dim=2, keepdim=True)
        out[:, :, :delta] = 0.0
        if preserve_counts:
            out[:, :, -1:] = out[:, :, -1:] + overflow
    else:
        d = -delta
        overflow = rolled[:, :, -d:].sum(dim=2, keepdim=True)
        out[:, :, -d:] = 0.0
        if preserve_counts:
            out[:, :, :1] = out[:, :, :1] + overflow
    return out


def _grid_shape(H: int, W: int, granularity: str, block_size: int):
    """Spatial resolution of the shift schedule for a given granularity."""
    if granularity == "global":
        return 1, 1
    if granularity == "block":
        return math.ceil(H / block_size), math.ceil(W / block_size)
    if granularity == "pixel":
        return H, W
    raise ValueError(f"Unknown granularity '{granularity}'.")


def _upsample_grid(t: Tensor, H: int, W: int, granularity: str, block_size: int) -> Tensor:
    """Upsample a ``[..., gh, gw]`` schedule to ``[..., H, W]`` (nearest)."""
    if granularity == "pixel":
        return t
    if granularity == "global":
        return t.expand(*t.shape[:-2], H, W)
    up = t.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
    return up[..., :H, :W]


# ---------------------------------------------------------------------------
# Black-box retiming (model-agnostic)
# ---------------------------------------------------------------------------

@register_threat("retiming_blackbox", "retiming")
class BlackBoxRetiming(EventThreat):
    """Sample a discrete temporal-shift schedule within a budget and apply it.

    Parameters
    ----------
    budget : int
        Maximum absolute shift in bins (the per-spike jitter budget B_inf).
    granularity : {"pixel", "block", "global"}
        Spatial resolution at which shifts vary.  ``"pixel"`` gives the strongest
        jitter; ``"block"`` / ``"global"`` model coarser transport-style delays.
    block_size : int
        Block edge (pixels) when ``granularity == "block"``.
    per_polarity : bool
        If True, ON and OFF channels are shifted independently (tests the
        polarity-asymmetry hypothesis); otherwise both share one schedule.
    mode : {"random", "worst_of_n"}
        ``"random"`` applies a single sampled schedule.  ``"worst_of_n"`` samples
        ``n_samples`` schedules and keeps the one that maximises the angular flow
        error (a light, gradient-free black-box optimisation; needs ``model``).
    n_samples : int
        Number of candidates for ``worst_of_n``.
    seed : int, optional
        Seed for reproducible sampling.
    """

    name = "retiming_blackbox"

    def __init__(self, budget: int = 2, granularity: str = "pixel", block_size: int = 16,
                 per_polarity: bool = True, mode: str = "random", n_samples: int = 8,
                 seed: Optional[int] = None, preserve_counts: bool = True, **kw):
        super().__init__(budget=budget, granularity=granularity, block_size=block_size,
                         per_polarity=per_polarity, mode=mode, n_samples=n_samples,
                         seed=seed, preserve_counts=preserve_counts, **kw)
        self.budget = int(budget)
        self.granularity = granularity
        self.block_size = int(block_size)
        self.per_polarity = per_polarity
        self.mode = mode
        self.n_samples = int(n_samples)
        self.preserve_counts = preserve_counts
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def _sample_delta_map(self, chunk: Tensor) -> Tensor:
        B, C, T, H, W = chunk.shape
        D = self.budget
        gh, gw = _grid_shape(H, W, self.granularity, self.block_size)
        n_chan = C if self.per_polarity else 1
        # Sample on CPU (Generator is CPU-bound) then move to the chunk's device.
        raw = torch.randint(-D, D + 1, (B, n_chan, gh, gw), generator=self._gen)
        if n_chan == 1 and C > 1:
            raw = raw.expand(B, C, gh, gw)
        raw = raw.to(chunk.device).contiguous()
        delta_map = _upsample_grid(raw, H, W, self.granularity, self.block_size)
        return delta_map.contiguous()

    def perturb(self, chunk, *, model=None, label=None, mask=None):
        if self.mode == "random":
            return retime_counts(chunk, self._sample_delta_map(chunk))

        if self.mode == "worst_of_n":
            if model is None or label is None:
                raise ValueError("mode='worst_of_n' requires `model` and `label`.")
            if mask is None:
                mask = torch.ones_like(label[:, :1])
            best_adv, best_loss = chunk, -float("inf")
            for _ in range(self.n_samples):
                adv = retime_counts(chunk, self._sample_delta_map(chunk))
                functional.reset_net(model)
                with torch.no_grad():
                    pred = model(adv)[-1]
                loss = angular_loss_function(pred, label, mask).item()
                if loss > best_loss:
                    best_adv, best_loss = adv, loss
            return best_adv

        raise ValueError(f"Unknown mode '{self.mode}'.")


# ---------------------------------------------------------------------------
# White-box projected-in-the-loop (PIL) retiming
# ---------------------------------------------------------------------------

@register_threat("retiming_pil", "pil")
class PILRetimingAttack(EventThreat):
    """White-box spike-retiming via projected-in-the-loop optimisation.

    A differentiable soft retiming ``soft = sum_d softmax(pi)_d * shift(x, d)`` is
    optimised (per-channel / per-block shift-probability logits ``pi``) to
    *maximise* the flow error, then projected to a feasible discrete schedule by
    taking the arg-max shift per block.

    Parameters
    ----------
    budget : int
        Candidate shifts span ``{-budget, ..., +budget}`` bins.
    granularity, block_size, per_polarity :
        As in :class:`BlackBoxRetiming` (schedule resolution).  ``"global"`` /
        ``"block"`` keep the logit tensor small and CPU-tractable.
    iters : int
        Number of optimisation steps.
    lr : float
        Adam learning rate on the logits.
    loss : {"angular", "cosine"}
        Flow-error objective to maximise (both already exist in the repo).
    budget_weight : float
        Optional penalty on the expected absolute shift (encourages small,
        stealthier delays); 0 maximises raw damage.
    """

    name = "retiming_pil"

    def __init__(self, budget: int = 2, granularity: str = "block", block_size: int = 32,
                 per_polarity: bool = True, iters: int = 10, lr: float = 0.5,
                 loss: str = "angular", budget_weight: float = 0.0, **kw):
        super().__init__(budget=budget, granularity=granularity, block_size=block_size,
                         per_polarity=per_polarity, iters=iters, lr=lr, loss=loss,
                         budget_weight=budget_weight, **kw)
        self.budget = int(budget)
        self.granularity = granularity
        self.block_size = int(block_size)
        self.per_polarity = per_polarity
        self.iters = int(iters)
        self.lr = float(lr)
        self.loss_name = loss
        self.budget_weight = float(budget_weight)

    def _loss_fn(self):
        return angular_loss_function if self.loss_name == "angular" else cosine_loss_function

    def perturb(self, chunk, *, model=None, label=None, mask=None):
        if model is None or label is None:
            raise ValueError("PILRetimingAttack requires `model` and `label`.")
        if mask is None:
            mask = torch.ones_like(label[:, :1])

        B, C, T, H, W = chunk.shape
        D = self.budget
        deltas: List[int] = list(range(-D, D + 1))
        K = len(deltas)
        loss_fn = self._loss_fn()

        # Freeze the model: only the retiming logits carry gradients.
        for p in model.parameters():
            p.requires_grad_(False)

        # Candidate rigid shifts are constants w.r.t. the optimisation.
        with torch.no_grad():
            shifted = torch.stack([global_soft_shift(chunk, d) for d in deltas], dim=0)
        abs_delta = torch.tensor([abs(d) for d in deltas], dtype=chunk.dtype,
                                 device=chunk.device).view(K, 1, 1, 1, 1)

        gh, gw = _grid_shape(H, W, self.granularity, self.block_size)
        n_chan = C if self.per_polarity else 1
        logits = torch.zeros(K, B, n_chan, gh, gw, device=chunk.device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=self.lr)

        def probs_upsampled(raw_logits):
            probs = torch.softmax(raw_logits, dim=0)          # [K,B,n_chan,gh,gw]
            if n_chan == 1 and C > 1:
                probs = probs.expand(K, B, C, gh, gw)
            return _upsample_grid(probs, H, W, self.granularity, self.block_size)

        for _ in range(self.iters):
            optimizer.zero_grad()
            functional.reset_net(model)
            p_up = probs_upsampled(logits)                    # [K,B,C,H,W]
            soft = (p_up.unsqueeze(3) * shifted).sum(dim=0)   # [B,C,T,H,W]
            pred = model(soft)[-1]
            loss = loss_fn(pred, label, mask)
            objective = -loss
            if self.budget_weight > 0:
                objective = objective + self.budget_weight * (p_up * abs_delta).sum(0).mean()
            objective.backward()
            optimizer.step()

        # Hard projection: arg-max shift per (channel, block) -> discrete schedule.
        with torch.no_grad():
            idx = torch.softmax(logits, dim=0).argmax(dim=0)  # [B,n_chan,gh,gw]
            delta_vals = torch.tensor(deltas, device=chunk.device)[idx]
            if n_chan == 1 and C > 1:
                delta_vals = delta_vals.expand(B, C, gh, gw)
            delta_map = _upsample_grid(delta_vals.contiguous(), H, W,
                                       self.granularity, self.block_size)
            adv = retime_counts(chunk, delta_map.contiguous())
        return adv.detach()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    torch.manual_seed(0)
    B, C, T, H, W = 2, 2, 21, 16, 20
    chunk = torch.randint(0, 4, (B, C, T, H, W)).float()

    for gran in ("global", "block", "pixel"):
        atk = BlackBoxRetiming(budget=3, granularity=gran, block_size=8, seed=1)
        adv = atk.perturb(chunk)
        assert adv.shape == chunk.shape, (gran, adv.shape)
        # Rate preservation: per-pixel total count over time is unchanged.
        assert torch.allclose(adv.sum(dim=2), chunk.sum(dim=2)), \
            f"count not preserved for granularity={gran}"
        # And it actually moved something (with very high probability).
        assert not torch.allclose(adv, chunk), f"no perturbation for granularity={gran}"

    # Global rigid soft-shift is also count-preserving.
    for d in (-2, 0, 3):
        assert torch.allclose(global_soft_shift(chunk, d).sum(dim=2), chunk.sum(dim=2))

    print("attacks.spike_retiming self-test passed "
          "(shape + rate-preservation verified for all granularities).")


if __name__ == "__main__":
    _self_test()
