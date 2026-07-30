#!/bin/bash
# Submit the attack pipeline on BlueBEAR: calibrate epsilon (CPU) once, then the
# FGSM sweep (GPU) which waits for calibration via a Slurm dependency.
# (PGD is dropped for now; re-enable via the commented block below.)
#
# Run from anywhere; it cd's to the repo root itself:
#   bash hpc/submit_attacks.sh
#
# Smoke-test first (a few samples per sequence) before the full run:
#   SMOKE=1 bash hpc/submit_attacks.sh
#
# Prerequisite:
#   * The split named by SPLIT (default valid_split_doubleseq.csv -- the paper's 5
#     validation/test sequences, already preprocessed) exists under
#     data/dataset/saved_flow_data/sequence_lists/. Output is split per-sequence
#     automatically. SPLIT may also be a folder to sweep every *.csv in it.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"     # repo root
mkdir -p hpc/logs results

SPLIT="${SPLIT:-valid_split_doubleseq.csv}"
# Smoke test: cap samples per sequence so the whole pipeline finishes in minutes.
EXTRA_SMOKE=""
if [ "${SMOKE:-0}" != "0" ]; then
    EXTRA_SMOKE="--max-chunks 5"
    echo "[SMOKE] capping to 5 samples per sequence"
fi

CAL_ID=$(sbatch --parsable hpc/calibrate_epsilon.slurm)
echo "calibrate      : job ${CAL_ID}"

FGSM_ID=$(ATTACK=fgsm SPLIT="${SPLIT}" EXTRA_ARGS="${EXTRA_SMOKE}" \
    sbatch --parsable --job-name=sweep_fgsm --dependency=afterok:${CAL_ID} \
           --export=ALL hpc/sweep_attack.slurm)
echo "fgsm sweep     : job ${FGSM_ID}  (after ${CAL_ID})"

# To also run PGD, uncomment:
# PGD_ID=$(ATTACK=pgd SPLIT="${SPLIT}" EXTRA_ARGS="--iters 7 ${EXTRA_SMOKE}" \
#     sbatch --parsable --job-name=sweep_pgd --dependency=afterok:${CAL_ID} \
#            --export=ALL hpc/sweep_attack.slurm)
# echo "pgd sweep      : job ${PGD_ID}  (after ${CAL_ID})"

echo
echo "Submitted. Watch with:  squeue --me"
echo "Outputs land in results/sweep_fgsm/"
