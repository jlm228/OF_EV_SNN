import torch

from data.file_generator import generate_files

flow_sequences = ['thun_00_a', 'zurich_city_02_d', 'zurich_city_03_a', 'zurich_city_08_a', 'zurich_city_11_b']


if __name__=='__main__':

        for sequence in flow_sequences:

            generate_files(root = '../data/dataset', sequence = sequence, num_frames_per_ts = 11)
