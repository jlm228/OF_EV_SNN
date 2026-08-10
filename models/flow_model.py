"""Common interface for optical-flow models.

See SNN_AV_experiment_plan.md Stage 1a. OfEvSnnAdapter wraps the existing
network_3d.poolingNet_cat_1res.NeuronPool_Separable_Pool3d unmodified.
"""
from typing import Protocol

import numpy as np
import torch
from spikingjelly.clock_driven import functional

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d


class FlowModel(Protocol):
    window_ms: float

    def input_from_events(self, ev, t0: int, t1: int) -> torch.Tensor:
        """Build one model-ready input tensor from raw (x, y, t, pol) events spanning
        [t0, t1) microseconds. t1 - t0 must equal whatever temporal context the model needs
        (for OfEvSnnAdapter: 2 * window_ms, i.e. the current window plus its predecessor)."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x -> predicted flow (2, H, W) in pixels over one `window_ms` window."""
        ...

    def reset_state(self) -> None:
        """Clear any state (e.g. SNN membrane potentials) that must not leak across samples."""
        ...


def _bin_window(x, y, t, pol, t_start_us, t_end_us, num_bins, height, width):
    """Matches CARLA-hpc-scripts/inspect_capture.py::bin_window_events and
    dsec_dataset_lite/data/event2frame.py::cumulate_spikes_into_frames exactly (verified by
    test_carla_event_binning.py). Duplicated in full rather than imported so this module
    doesn't depend on a sibling repo existing on disk -- keep the two in sync if either
    changes; channel 0 = ON (pol truthy), channel 1 = OFF, native unrectified (x, y)."""
    frame = np.zeros((num_bins, 2, height, width), dtype=np.float32)
    dt = (t_end_us - t_start_us) / num_bins
    on = np.asarray(pol).astype(bool)
    x, y, t = np.asarray(x), np.asarray(y), np.asarray(t)
    for b in range(num_bins):
        b_start, b_end = t_start_us + b * dt, t_start_us + (b + 1) * dt
        sel = (t >= b_start) & (t < b_end)
        xb, yb, onb = x[sel], y[sel], on[sel]
        np.add.at(frame[b, 0], (yb[onb], xb[onb]), 1)
        np.add.at(frame[b, 1], (yb[~onb], xb[~onb]), 1)
    return frame


class OfEvSnnAdapter:
    """Wraps NeuronPool_Separable_Pool3d to satisfy FlowModel.

    Stateful (SNN membrane potentials) but reset per-sample, not persisted across a
    sequence -- confirmed from test_network_metrics.py, which calls reset_state() before
    every forward(). Each 21-time-bin sample is self-contained; there is no cross-window
    state to carry.
    """

    window_ms: float = 100.0
    num_bins: int = 11
    height: int = 480
    width: int = 640

    def __init__(self, checkpoint_path: str = None, multiply_factor: float = 35.0,
                 device: str = None):
        self.device = torch.device(device) if device else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.net = NeuronPool_Separable_Pool3d(multiply_factor=multiply_factor).to(self.device)
        if checkpoint_path:
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.net.eval()

    def input_from_events(self, ev, t0: int, t1: int) -> torch.Tensor:
        window_us = int(round(self.window_ms * 1000))
        assert t1 - t0 == 2 * window_us, (
            "OfEvSnnAdapter needs exactly two consecutive %dms windows of context "
            "(got t1-t0=%d us)" % (self.window_ms, t1 - t0))
        t_mid = t0 + window_us

        frame1 = _bin_window(ev["x"], ev["y"], ev["t"], ev["pol"], t0, t_mid,
                              self.num_bins, self.height, self.width)
        frame2 = _bin_window(ev["x"], ev["y"], ev["t"], ev["pol"], t_mid, t1,
                              self.num_bins, self.height, self.width)
        chunk = np.concatenate([frame1, frame2], axis=0)[-21:]  # matches DSECDatasetLite
        chunk = torch.from_numpy(chunk).unsqueeze(0)            # (1, 21, 2, H, W)
        return torch.transpose(chunk, 1, 2)                     # (1, 2, 21, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            _, _, _, pred = self.net(x)
        return pred

    def reset_state(self) -> None:
        functional.reset_net(self.net)
