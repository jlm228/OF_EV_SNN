import random

import torch

from spikingjelly.clock_driven import functional

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d

from tqdm import tqdm

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite 

from eval.vector_loss_functions import *

import math
import os
import re
import csv

import pandas as pd

# Enable GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

RAD2DEG = 180.0 / math.pi

################################
## DATASET LOADING/GENERATION ##
################################

# Define desired temporal resolution (50ms between consecutive layers)
num_frames_per_ts = 11
forward_labels = 1

# Create validation dataset
ROOT = 'data/dataset/saved_flow_data'
SPLIT = 'valid_split_doubleseq.csv'
OUTDIR = 'results'

print("Creating Validation Dataset ...")
valid_dataset = DSECDatasetLite(root = ROOT, file_list = SPLIT, num_frames_per_ts = num_frames_per_ts, stereo = False, transform = None)

# Per-sample identity, reconstructed from the split CSV. The loader iterates rows in order
# (shuffle=False) and takes each sample's label/mask from the *second* file in the row, so
# that column is the canonical per-sample name. This lets us break results down by sequence.
_split_rows = pd.read_csv(os.path.join(ROOT, 'sequence_lists', SPLIT), header = None)
sample_files = _split_rows.iloc[:, 1].tolist()
sample_seqs = [re.sub(r'_\d+\.npy$', '', nm) for nm in sample_files]

# Define validation dataloader
batch_size = 1
valid_dataloader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False, drop_last = False, pin_memory = True)


########################
## TRAINING FRAMEWORK ##
########################

# Create the network

net = NeuronPool_Separable_Pool3d(multiply_factor = 35.).to(device)
net.load_state_dict(torch.load('examples/checkpoint_epoch34.pth'))

mod_fcn = mod_loss_function
lambda_mod = 1.

ang_fcn = angular_loss_function
lambda_ang = 1.

###############################
## COMPUTE MODEL PERFORMANCE ##
###############################

n_chunks_valid = len(valid_dataloader)

net.eval()

epoch_mod_loss_test = 0.
epoch_ang_loss_test = 0.

# One row per sample: [index, sequence, filename, EPE, angular_rad, angular_deg].
per_sample = []

print('Validating... (test sequence)')

for sample_idx, (chunk, mask, label) in enumerate(tqdm(valid_dataloader)):

    functional.reset_net(net)

    chunk = torch.transpose(chunk, 1, 2)

    mask = torch.unsqueeze(mask, dim = 1)

    chunk = chunk.to(device = device, dtype = torch.float32)
    label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]
    mask = mask.to(device = device)

    with torch.no_grad():
        _, _, _, pred = net(chunk)


    mod_loss = mod_fcn(pred, label, mask)
    ang_loss = ang_fcn(pred, label, mask)

    epoch_mod_loss_test += mod_loss.item() * batch_size
    epoch_ang_loss_test += ang_loss.item() * batch_size

    ang_rad = ang_loss.item()
    per_sample.append([
        sample_idx,
        sample_seqs[sample_idx] if sample_idx < len(sample_seqs) else '',
        sample_files[sample_idx] if sample_idx < len(sample_files) else '',
        mod_loss.item(),
        ang_rad,
        ang_rad * RAD2DEG,
    ])


epoch_mod_loss_test /= n_chunks_valid
epoch_ang_loss_test /= n_chunks_valid

epoch_loss_valid = epoch_mod_loss_test + epoch_ang_loss_test
print('Epoch loss (Validation): {} \n'.format(epoch_loss_valid))
    
print({
    'TOT_mod_loss': epoch_mod_loss_test,
    'TOT_ang_loss': epoch_ang_loss_test * 180 / math.pi,
    'TOT_total_loss': epoch_loss_valid,
})

###############################
## SAVE RESULTS AS CSV        ##
###############################

os.makedirs(OUTDIR, exist_ok = True)

# 1. Overall metrics (the paper's Table 1/2 numbers).
overall_path = os.path.join(OUTDIR, 'baseline_metrics.csv')
with open(overall_path, 'w', newline = '') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    w.writerow(['EPE', epoch_mod_loss_test])                       # = AEE, px/s
    w.writerow(['angular_rad', epoch_ang_loss_test])               # paper's units
    w.writerow(['angular_deg', epoch_ang_loss_test * RAD2DEG])     # printed units
    w.writerow(['n_samples', len(per_sample)])

# 2. Per-sample metrics.
per_sample_path = os.path.join(OUTDIR, 'baseline_per_sample.csv')
with open(per_sample_path, 'w', newline = '') as f:
    w = csv.writer(f)
    w.writerow(['index', 'sequence', 'filename', 'EPE', 'angular_rad', 'angular_deg'])
    w.writerows(per_sample)

# 3. Per-sequence means (grouped in first-appearance / split order). These break the
#    validation set down per DSEC sequence -- unpublished, but the reusable asset for
#    later analysis of which scenes are hardest.
seq_groups = {}
seq_order = []
for row in per_sample:
    seq = row[1]
    if seq not in seq_groups:
        seq_groups[seq] = []
        seq_order.append(seq)
    seq_groups[seq].append(row)

per_seq_path = os.path.join(OUTDIR, 'baseline_per_sequence.csv')
with open(per_seq_path, 'w', newline = '') as f:
    w = csv.writer(f)
    w.writerow(['sequence', 'n_samples', 'EPE', 'angular_rad', 'angular_deg'])
    for seq in seq_order:
        rows = seq_groups[seq]
        cnt = len(rows)
        epe = sum(r[3] for r in rows) / cnt
        rad = sum(r[4] for r in rows) / cnt
        deg = sum(r[5] for r in rows) / cnt
        w.writerow([seq, cnt, epe, rad, deg])

print('\nResults written to:')
print('  ' + overall_path)
print('  ' + per_sample_path)
print('  ' + per_seq_path)

print('SO FAR, EVERYTHING IS WORKING!!!')
