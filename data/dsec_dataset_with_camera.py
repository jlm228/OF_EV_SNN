import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DSECDatasetWithCamera(Dataset):
    """Like DSECDatasetLite, but each sample also carries a real camera image.

    Sequence-list CSV format (3 columns, no header): frame1.npy, frame2.npy,
    camera_image.png -- the camera image is the one whose capture timestamp is
    closest to frame2's ground-truth timestamp (see
    dsec_dataset_lite/build_sequence_list.py, which generates these CSVs).
    """

    def __init__(self, root: str, file_list: str, images_root: str, sequence: str,
                 num_frames_per_ts: int = 11):
        self.events_path = os.path.join(root, 'event_tensors', '{}frames'.format(str(num_frames_per_ts).zfill(2)))
        self.flow_path = os.path.join(root, 'gt_tensors')
        self.mask_path = os.path.join(root, 'mask_tensors')
        self.images_path = os.path.join(images_root, sequence, 'images', 'left', 'rectified')

        sequence_file = os.path.join(root, 'sequence_lists', file_list)
        self.files = pd.read_csv(sequence_file, header=None)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        target_file_1 = self.files.iloc[idx, 0]
        target_file_2 = self.files.iloc[idx, 1]
        camera_file = self.files.iloc[idx, 2]

        eventsL1 = torch.from_numpy(np.load(os.path.join(self.events_path, target_file_1)))
        eventsL2 = torch.from_numpy(np.load(os.path.join(self.events_path, target_file_2)))
        eventsL = torch.cat((eventsL1, eventsL2), axis=0)

        mask = torch.from_numpy(np.load(os.path.join(self.mask_path, target_file_2)))
        label = torch.from_numpy(np.load(os.path.join(self.flow_path, target_file_2)))

        image_bgr = cv2.imread(os.path.join(self.images_path, camera_file), cv2.IMREAD_COLOR)
        image = torch.from_numpy(image_bgr)  # [H, W, 3] uint8 BGR, native camera resolution

        return eventsL[-21:], mask, label, image
