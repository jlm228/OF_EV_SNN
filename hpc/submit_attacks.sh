#!/bin/bash
# Submit the full attack pipeline on BlueBEAR: calibrate epsilon (CPU) once, then
# the FGSM and PGD sweeps (GPU) which wait for calibration via a Slurm dependency.
#
# Run from anywhere; it cd's to the repo root itself:
#   bash hpc/submit_attacks.sh
#
# Smoke-test first (a few samples per sequence) before the full run:
#   SMOKE=1 bash hpc/submit_attacks.sh
#
# Prerequisites:
#   * The split folder (default test_instances) under
#     data/dataset/saved_flow_data/sequence_lists/ holds ONE CSV per test
#     sequence (the paper's 5: thun_00_a, zurich_city_02_d, zurich_city_03_a,
#     zurich_city_08_a, zurich_city_11_b). The sweep processes every *.csv there.
#   * The preprocessed tensors for those sequences exist (see hpc/setup_data.sh).

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"     # repo root
mkdir -p hpc/logs results

SPLIT_DIR="${SPLIT_DIR:-test_instances}"
# Smoke test: cap samples per sequence so the whole pipeline finishes in minutes.
EXTRA_SMOKE=""
if [ "${SMOKE:-0}" != "0" ]; then
    EXTRA_SMOKE="--max-chunks 5"
    echo "[SMOKE] capping to 5 samples per sequence"
fi

CAL_ID=$(sbatch --parsable hpc/calibrate_epsilon.slurm)
echo "calibrate      : job ${CAL_ID}"

FGSM_ID=$(ATTACK=fgsm SPLIT_DIR="${SPLIT_DIR}" EXTRA_ARGS="${EXTRA_SMOKE}" \
    sbatch --parsable --job-name=sweep_fgsm --dependency=afterok:${CAL_ID} \
           --export=ALL hpc/sweep_attack.slurm)
echo "fgsm sweep     : job ${FGSM_ID}  (after ${CAL_ID})"

PGD_ID=$(ATTACK=pgd SPLIT_DIR="${SPLIT_DIR}" EXTRA_ARGS="--iters 7 ${EXTRA_SMOKE}" \
    sbatch --parsable --job-name=sweep_pgd --dependency=afterok:${CAL_ID} \
           --export=ALL hpc/sweep_attack.slurm)
echo "pgd sweep      : job ${PGD_ID}  (after ${CAL_ID})"

echo
echo "Submitted. Watch with:  squeue --me"
echo "Outputs land in results/sweep_fgsm/ and results/sweep_pgd/"
