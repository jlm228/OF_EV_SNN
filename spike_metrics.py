"""Record NeuroBench-style spike metrics for OF_EV_SNN over a few windows.

    # synthetic input, no capture needed -- validates the hooks against the real architecture
    python spike_metrics.py --synthetic --windows 1 --out results/spike_metrics

    # real voxel tensors from a capture
    python spike_metrics.py --tensors <capture>/tensors --id carla_<job> --windows 8 \
        --out results/spike_metrics

The input is (B, 2, 21, 480, 640). 21 frames is not arbitrary: the five Conv3d stages each
consume 4 frames with no temporal padding, 21 -> 17 -> 13 -> 9 -> 5 -> 1, which is what makes
`squeeze(out_conv4, 2)` and the `[:, :, -1]` skip slices line up.
"""
import argparse
import json
import os
import sys
import time

import torch

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d

CARLA_SCRIPTS_ROOT = os.environ.get("CARLA_SCRIPTS_ROOT", "../CARLA-hpc-scripts")
sys.path.insert(0, os.path.abspath(CARLA_SCRIPTS_ROOT))

from snnmetrics.probe import SpikeProbe                      # noqa: E402
from snnmetrics.cost import (footprint_bytes, connection_sparsity,   # noqa: E402
                             write_csvs)

FRAMES = 21
RESOLUTION = (480, 640)


def load_net(checkpoint, device):
    net = NeuronPool_Separable_Pool3d().to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    return net


def synthetic_windows(n, density=0.03, seed=0, device="cpu"):
    """Random input in this model's real input format: two channels of ON/OFF event counts.

    For checking the code runs, not for reporting. How much the network spikes depends on the
    real input, so the numbers this produces mean nothing.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(n):
        x = torch.zeros(1, 2, FRAMES, *RESOLUTION)
        k = int(density * x.numel())
        idx = torch.randint(0, x.numel(), (k,), generator=g)
        x.view(-1)[idx] = torch.randint(1, 20, (k,), generator=g).float()
        yield x.to(device)


def real_windows(tensors_dir, capture_id, n, device="cpu"):
    """Voxel tensors as carla_to_voxel / inspect_capture --voxelise wrote them."""
    import numpy as np
    root = os.path.join(tensors_dir, "event_tensors", "%dframes" % (FRAMES // 2 + 1))
    if not os.path.isdir(root):
        root = os.path.join(tensors_dir, "event_tensors")
    files = sorted(f for f in os.listdir(root)
                   if f.endswith(".npy") and f.startswith(capture_id or ""))
    if not files:
        raise SystemExit("no .npy tensors under %s for id %r" % (root, capture_id))
    for fname in files[:n]:
        arr = np.load(os.path.join(root, fname)).astype("float32")
        t = torch.from_numpy(arr)
        while t.dim() < 5:
            t = t.unsqueeze(0)
        yield t.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="examples/dsec_fullset.pth")
    ap.add_argument("--tensors", default=None, help="<capture>/tensors")
    ap.add_argument("--id", default=None, help="capture id prefix")
    ap.add_argument("--synthetic", action="store_true",
                    help="random input at realistic occupancy; for validating the hooks")
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--out", default=os.path.join("results", "spike_metrics"))
    ap.add_argument("--name", default="of_ev_snn")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--channel-dim", type=int, default=None,
                    help="override the inferred channel axis for the per-channel counts")
    args = ap.parse_args()

    if not args.synthetic and not args.tensors:
        raise SystemExit("pass --tensors <dir> or --synthetic")

    device = torch.device(args.device)
    print("device: %s | windows: %d | input: (1, 2, %d, %d, %d)"
          % (device, args.windows, FRAMES, *RESOLUTION), flush=True)

    net = load_net(args.checkpoint, device)
    static = {"footprint_bytes": footprint_bytes(net),
              "connection_sparsity": connection_sparsity(net),
              "device": str(device),
              "input_source": "synthetic" if args.synthetic else args.tensors}
    print("footprint: %.2f MB | connection sparsity: %.4g"
          % (static["footprint_bytes"] / 1e6, static["connection_sparsity"]), flush=True)

    source = (synthetic_windows(args.windows, device=device) if args.synthetic
              else real_windows(args.tensors, args.id, args.windows, device=device))

    # spikingjelly's clock_driven reset lives here rather than in snnmetrics, which must stay
    # free of spikingjelly so one metric implementation serves both repos.
    from spikingjelly.clock_driven import functional

    wall = []
    with SpikeProbe(net, channel_dim=args.channel_dim) as probe, torch.no_grad():
        for i, chunk in enumerate(source):
            functional.reset_net(net)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(chunk)
            if device.type == "cuda":
                torch.cuda.synchronize()
            wall.append(time.perf_counter() - t0)
            probe.mark_window()
            print("  window %d/%d  %.2f s" % (i + 1, args.windows, wall[-1]), flush=True)

    static["wall_ms_per_window"] = 1000.0 * sum(wall) / len(wall)
    os.makedirs(args.out, exist_ok=True)
    records_path = probe.dump(os.path.join(args.out, "%s_spikes.json" % args.name), meta=static)

    overall = write_csvs(args.name, json.load(open(records_path)), args.out, extra=static)
    print("\n=== %s ===" % args.name)
    for k, v in overall.items():
        print("  %-28s %s" % (k, "%.6g" % v if isinstance(v, float) else v))
    print("\nrecords: %s" % records_path)


if __name__ == "__main__":
    main()
