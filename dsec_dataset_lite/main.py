import sys

import torch

from data.file_generator import generate_files

# The 5 sequences of the paper's DSEC validation split (valid_split_doubleseq.csv).
DEFAULT_SEQUENCES = ['thun_00_a', 'zurich_city_02_d', 'zurich_city_03_a', 'zurich_city_08_a', 'zurich_city_11_b']

# Sequences may be named on the command line, e.g. `python main.py thun_00_a`, so a single
# sequence can be validated on its own or the work split across parallel jobs.
flow_sequences = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SEQUENCES

if __name__=='__main__':

        for sequence in flow_sequences:

            generate_files(root = '../data/dataset', sequence = sequence, num_frames_per_ts = 11)
