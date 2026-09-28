#!/bin/bash
# Pre-flight measurement on one window: ATTACK_PLAN.md 2.3 step 1.
#
#   sbatch hpc/preflight.slurm <capture_dir> [window]
#
# Reports, for this model:
#   * swap coverage -- which spiking modules a surrogate swap reaches, and that `pool`
#     (the V_th = inf integrator) is skipped;
#   * mean and max |u| per spiking layer, which set the sigmoid omega search bracket for the
#     A calibration (step 5). OF_EV_SNN's u = 35a - 1, so its |u| is large and the bracket
#     cannot be carried over from the Atan one;
#   * seconds per forward and per forward+backward, which set EPS_CHUNK and, with SDA's
#     candidate counts, its largest feasible rho;
#   * peak GPU memory, which decides the finite-difference batch size.

#SBATCH --job-name=of_ev_snn_preflight
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch hpc/preflight.slurm <capture_dir> [window]"
CAPTURE="${1:?${USAGE}}"
WINDOW="${2:-}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT

[ -d "${CARLA_SCRIPTS_ROOT}/attack_core" ] || {
  echo "no attack_core under ${CARLA_SCRIPTS_ROOT} -- set CARLA_SCRIPTS_ROOT" >&2; exit 1; }
[ -f "${CAPTURE}/events.npy" ] || {
  echo "no ${CAPTURE}/events.npy -- the pre-flight encodes its window from the raw events" >&2
  exit 1; }
[ -f "${CAPTURE}/windows.csv" ] || {
  echo "no ${CAPTURE}/windows.csv -- needed for the window's start timestamp" >&2; exit 1; }

WINDOW_ARG=""
[ -n "${WINDOW}" ] && WINDOW_ARG="--preflight-window ${WINDOW}"

echo "=== pre-flight: of_ev_snn on ${CAPTURE} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --surrogate stays native: the measurement swaps in its own probe whichever surrogate is
# configured, so there is no A to supply and nothing to tune yet.
# shellcheck disable=SC2086
python attack_carla.py --preflight ${WINDOW_ARG} \
    --capture "${CAPTURE}" \
    --objective div \
    --epsilons 0

echo
echo "Next: put mean |u| into the step-5 omega bracket, and the forward+backward time into"
echo "EPS_CHUNK (one SA-PGD task is ~101 passes per epsilon against PGD's 10)."
