"""Pick an epsilon range for FGSM on DSEC event tensors.

As event tensors are used, one unit of epsilon means:
'add one event to every pixel, in every polarity channel, in every time bin'.

This is a very large pertubation for sparse data, so the epsilon range in this case
is orders of magnitude smaller than the typical FGSM range for image recognition tasks.

Run once. Copy the printed epsilon values into the attack sweep.

Use:
    python calibrate_epsilon.py
"""

import numpy as np
import torch

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite

ROOT = "data/dataset/saved_flow_data"
SPLIT = "train_split_doubleseq.csv" # use training split to calibrate epsilon
N_SAMPLES = 200 # num. of samples to randomly take from the dataset
MASS_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25] # target percentages of clean events to inject as noise

# Load dataset and pick samples
dataset = DSECDatasetLite(root=ROOT, file_list=SPLIT, num_frames_per_ts=11,
                          stereo=False, transform=None)

rng = np.random.default_rng(999) # keep seed the same for reproducibility
indices = rng.choice(len(dataset), size=min(N_SAMPLES, len(dataset)), replace=False)

# Initialise counters
n_voxels, sum_of_event_counts, count_of_nonzero_voxels = 0, 0.0, 0
counts = np.zeros(64, dtype=np.int64)          # histogram of occupied voxel counts

# Scan the dataset and update histogram and counters
for idx in indices:
    E, _, _ = dataset[int(idx)]
    arr = torch.as_tensor(E).float().numpy()     # [T, 2, H, W]

    n_voxels = arr.size
    sum_of_event_counts += arr.sum()
    count_of_nonzero_voxels += (arr > 0).sum()

    occupied = np.rint(arr[arr > 0]).astype(int)
    counts += np.bincount(np.clip(occupied, 0, 63), minlength=64)

# Compute statistics
n = len(indices)
avg_events_per_sample = sum_of_event_counts / n
sparsity = count_of_nonzero_voxels / (n * n_voxels) # fraction of voxels occupied by at least one event
frac_single = counts[1] / counts[1:].sum() # fraction of occupied voxels that hold exactly one event

# Print epsilon table and dataset statistics
print(f"Scanned {n} chunks\n")
print(f"  voxels per sample     {n_voxels:,}")
print(f"  occupied              {sparsity * 100:.2f}%")
print(f"  events per sample     {avg_events_per_sample:,.0f}")
print(f"  occupied voxels = 1   {frac_single * 100:.1f}%")
print(f"  largest count seen    {np.flatnonzero(counts).max()}\n")

# As FGSM's signed step pushes every voxel by +/- epsilon, any empty voxels that get a negative sign
# will be clamped to 0 and add nothing. Only the +epsilon half survive.
# Therefore, the attack injects roughly 0.5 * epsilon * n_voxels events.
# This can be inverted to find the epsilon that injects a target fraction of clean events.
print("  eps        injects   as % of clean events")
for rho in MASS_FRACTIONS:
    eps = rho * avg_events_per_sample / (0.5 * n_voxels)
    print(f"  {eps:<9.5f} {rho * avg_events_per_sample:>9,.0f}   {rho * 100:>5.1f}%")

print(f"\n  eps = 1.0 erases any occupied voxel holding 1 event "
      f"({frac_single * 100:.0f}% of them)")
