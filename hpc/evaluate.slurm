#!/bin/bash
# Reproduce the paper's DSEC validation metrics (Table 1/2) on the full 2152-sample split.
#
#   sbatch hpc/evaluate.slurm
#
# Expect TOT_mod_loss ~= 1.10 (AEE, px/s) and TOT_ang_loss ~= 5.4 (degrees; the paper
# reports 0.094 rad and the script converts). Within ~+-0.02 on mod loss is a faithful
# pipeline; a large gap points at preprocessing or multiply_factor, not the weights.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2:0:0
#SBATCH --mem=32G
#SBATCH --job-name=of_ev_snn_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --qos=bbgpu

set -e

source "${SLURM_SUBMIT_DIR}/hpc/env.sh"
cd "${SLURM_SUBMIT_DIR}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

python test_network_metrics.py
