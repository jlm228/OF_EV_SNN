#!/bin/bash
# Sweep one additive attack (fgsm|pgd) over epsilon x objective across a folder of
# per-sequence test splits, with the random-sign control and gradient attribution.
# GPU job. Reads the calibrated epsilon list and prepends 0.0 (clean baseline).
#
# Driven by environment variables (set them before sbatch, pass with --export=ALL):
#   ATTACK         fgsm | pgd                         (required)
#   SPLIT_DIR      split folder under sequence_lists/ (default: test_instances)
#   LOSSES         attack objectives                  (default: "epe angular cosine")
#   RAND_RESTARTS  random-sign control draws R        (default: 5)
#   EPS_FILE       calibrated epsilon list            (default: results/calibrated_epsilons.txt)
#   OUTDIR         output directory                   (default: results/sweep_<ATTACK>)
#   EXTRA_ARGS     extra sweep flags, e.g. "--iters 7" (pgd) or "--max-chunks 20" (smoke test)
#
#   ATTACK=fgsm sbatch --export=ALL hpc/sweep_attack.slurm
#
#SBATCH --job-name=sweep_attack
#SBATCH --time=24:00:00
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

ATTACK="${ATTACK:?set ATTACK=fgsm or ATTACK=pgd}"
SPLIT_DIR="${SPLIT_DIR:-test_instances}"
LOSSES="${LOSSES:-epe angular cosine}"
RAND_RESTARTS="${RAND_RESTARTS:-5}"
EPS_FILE="${EPS_FILE:-results/calibrated_epsilons.txt}"
OUTDIR="${OUTDIR:-results/sweep_${ATTACK}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[ -f "${EPS_FILE}" ] || { echo "ERROR: ${EPS_FILE} not found -- run calibrate_epsilon.slurm first."; exit 1; }
EPSILONS="0.0 $(cat "${EPS_FILE}")"     # clean baseline + calibrated points

echo "attack=${ATTACK} | split=${SPLIT_DIR} | losses=[${LOSSES}] | R=${RAND_RESTARTS}"
echo "epsilons=[${EPSILONS}] | outdir=${OUTDIR} | extra=[${EXTRA_ARGS}]"

# Unquoted list vars are intentional (nargs='+' flags). Model/data flags use the
# defaults from attacks/cli_common (checkpoint_epoch34.pth, multiply_factor=35,
# root data/dataset/saved_flow_data); override via EXTRA_ARGS if needed.
python sweep_epsilon.py \
    --attack "${ATTACK}" \
    --split "${SPLIT_DIR}" \
    --losses ${LOSSES} \
    --epsilons ${EPSILONS} \
    --rand-restarts "${RAND_RESTARTS}" \
    --attribution \
    --outdir "${OUTDIR}" \
    ${EXTRA_ARGS}

echo "Done. Outputs in ${OUTDIR}/"
