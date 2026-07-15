import torch

from data.file_generator import generate_files

flow_sequences = ['thun_00_a']


if __name__=='__main__':

        for sequence in flow_sequences:

            generate_files(root = '../data/dataset', sequence = sequence, num_frames_per_ts = 11)
