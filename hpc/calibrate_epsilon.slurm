#!/bin/bash
# Calibrate the FGSM/PGD epsilon range from the data (CPU, quick).
#
# Writes the epsilon list to $EPS_OUT (default results/calibrated_epsilons.txt);
# the sweep jobs read that file straight into `--epsilons`.
#
#   sbatch hpc/calibrate_epsilon.slurm
#
#SBATCH --job-name=calib_eps
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=hpc/logs/calibrate_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"          # jobs are submitted from the repo root
source hpc/env.sh                    # modules + venv (see hpc/env.sh)

export EPS_OUT="${EPS_OUT:-results/calibrated_epsilons.txt}"
mkdir -p "$(dirname "${EPS_OUT}")" hpc/logs

# Run as a module from the repo root so its absolute imports (data.*, eval.*) resolve.
python -m attacks.fgsm_pgd.calibrate_epsilon

echo
echo "Calibrated epsilons -> ${EPS_OUT}: $(cat "${EPS_OUT}")"
