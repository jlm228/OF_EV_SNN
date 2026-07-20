import argparse
import csv
import os

import numpy as np


def build_sequence_list(root: str, sequence: str):
    """Build a 3-column sequence-list CSV (frame1.npy, frame2.npy, camera_image.png)
    for an already-preprocessed GT sequence (see dsec_dataset_lite/main.py), matching
    each ground-truth sample's timestamp to the nearest real camera image.

    Written to <root>/saved_flow_data/sequence_lists/test_instances/<sequence>.csv.
    """
    gt_dir = os.path.join(root, 'saved_flow_data', 'gt_tensors')
    seq_lists_dir = os.path.join(root, 'saved_flow_data', 'sequence_lists', 'test_instances')
    os.makedirs(seq_lists_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(gt_dir) if f.startswith(sequence + '_'))

    timestamps = np.loadtxt(os.path.join(root, 'train', sequence, 'flow', 'forward_timestamps.txt'),
                             delimiter=',', dtype='int64', skiprows=1)
    t_ends = timestamps[:, 1]
    assert len(files) == len(t_ends), (
        f'{len(files)} gt_tensors files but {len(t_ends)} forward_timestamps.txt rows for {sequence}')

    image_timestamps = np.loadtxt(os.path.join(root, 'train', sequence, 'image_timestamps.txt'), dtype='int64')

    rows = []
    for i in range(1, len(files)):
        frame1, frame2 = files[i - 1], files[i]
        t_end = t_ends[i]  # timestamp of the "current" (frame2) label
        image_idx = int(np.argmin(np.abs(image_timestamps - t_end)))
        image_name = f'{image_idx:06d}.png'
        rows.append((frame1, frame2, image_name))

    csv_path = os.path.join(seq_lists_dir, f'{sequence}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(rows)

    print(f'Wrote {len(rows)} rows -> {csv_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequences', nargs='+')
    args = p.parse_args()

    for sequence in args.sequences:
        build_sequence_list(root='../data/dataset', sequence=sequence)
