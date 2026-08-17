"""Check that inspect_capture.py's event binning matches DSEC's own preprocessing.

    python test_carla_event_binning.py

Bins raw events straight out of DSEC's events.h5 with the CARLA converter's function and
compares the result against the precomputed event tensors on disk, which were produced by the
original preprocessing path. Any divergence in bin edges, polarity channel order or pixel
indexing shows up as a mismatch here rather than as an unexplained accuracy drop later.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CARLA-hpc-scripts"))
from inspect_capture import bin_window_events  # noqa: E402

from dsec_dataset_lite.data.event2frame import EventSlicer  # noqa: E402
import h5py  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "data", "dataset")

# (sequence, window indices to check -- 1-based, matching the _%04d.npy filenames)
CASES = [
    ("thun_00_a", [1, 2, 20]),
    ("zurich_city_02_a", [1, 15]),
]


def check_window(sequence, timestamps, window_idx, slicer):
    row = timestamps[window_idx - 1]
    t_beg, t_end = int(row[0]), int(row[1])

    ev = slicer.get_events(t_beg, t_end)
    got = bin_window_events(ev["x"], ev["y"], ev["t"], ev["p"], t_beg, t_end, num_bins=11)

    expected_path = os.path.join(
        ROOT, "saved_flow_data", "event_tensors", "11frames",
        "%s_%04d.npy" % (sequence, window_idx))
    expected = np.load(expected_path)

    ok = got.shape == expected.shape and np.array_equal(got, expected)
    print("  window %4d: t=[%d,%d) got_sum=%.0f expected_sum=%.0f -> %s"
          % (window_idx, t_beg, t_end, got.sum(), expected.sum(), "PASS" if ok else "FAIL"))
    return ok


def main():
    all_ok = True
    for sequence, window_idxs in CASES:
        print("=== %s ===" % sequence)
        timestamps = np.loadtxt(
            os.path.join(ROOT, "train", sequence, "flow", "forward_timestamps.txt"),
            delimiter=",", dtype="int64", skiprows=1)
        events_h5 = os.path.join(ROOT, "train", sequence, "events", "left", "events.h5")
        with h5py.File(events_h5, "r") as f:
            slicer = EventSlicer(f)
            for window_idx in window_idxs:
                all_ok &= check_window(sequence, timestamps, window_idx, slicer)

    assert all_ok, ("bin_window_events disagrees with the precomputed tensors for at least "
                    "one window")
    print("\nPASS: bin_window_events reproduces DSEC's own preprocessing exactly.")


if __name__ == "__main__":
    main()
