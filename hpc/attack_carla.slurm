#!/bin/bash
# One array task = one attack objective, swept over every epsilon.
#
#   sbatch --array=1-5 hpc/attack_carla.slurm <capture_dir> <grid_manifest>
#
# Submitted by hpc/submit_attack_sweep.sh, which writes the manifest. Run that, not this.
#
# WHY A MANIFEST AND NOT ENVIRONMENT VARIABLES: this cluster's sbatch does not honour an
# in-script `#SBATCH --export=ALL`, so variables set at submission time silently do not arrive.
# The array index selects a line of a file instead, which needs neither exported variables nor
# a growing list of positional arguments.
#
# WHY ONE TASK PER OBJECTIVE, NOT PER CELL: the checkpoint loads once per task rather than once
# per epsilon, and the clean forward pass is objective- and epsilon-independent, so attack_core
# computes it once per window and reuses it across the whole ramp.
#
# Manifest line format (whitespace separated):
#   <objective> <sign> <attack> <iters> <epsilon> [<epsilon> ...]

#SBATCH --job-name=of_ev_snn_attack
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:0:0
#SBATCH --output=hpc/logs/%x_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch --array=1-N hpc/attack_carla.slurm <capture_dir> <grid_manifest>"
CAPTURE="${1:?${USAGE}}"
MANIFEST="${2:?${USAGE}}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT

[ -d "${CAPTURE}/tensors" ] || {
  echo "no ${CAPTURE}/tensors -- run inspect_capture.py --voxelise first" >&2; exit 1; }
[ -f "${MANIFEST}" ] || { echo "no manifest at ${MANIFEST}" >&2; exit 1; }
[ -d "${CARLA_SCRIPTS_ROOT}/attack_core" ] || {
  echo "no attack_core under ${CARLA_SCRIPTS_ROOT} -- set CARLA_SCRIPTS_ROOT" >&2; exit 1; }

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")"
[ -n "${LINE}" ] || { echo "manifest has no line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
read -r OBJECTIVE SIGN ATTACK ITERS EPSILONS <<< "${LINE}"

CLEAN="results/carla_eval/pred/of_ev_snn"
OUT="results/attack/of_ev_snn"
LABEL="${OBJECTIVE}"
[ "${OBJECTIVE}" = "div" ] && LABEL="div_${SIGN}"

[ -d "${CLEAN}" ] || {
  echo "no clean predictions at ${CLEAN} -- run hpc/carla_eval.slurm first; every attacked" >&2
  echo "output directory is SEEDED from it, so a band-limited attack has no gaps" >&2
  exit 1; }

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${LABEL} ==="
echo "capture   ${CAPTURE}"
echo "attack    ${ATTACK} (${ITERS} iters)"
echo "epsilons  ${EPSILONS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Unquoted on purpose: --epsilons takes nargs='+'.
# shellcheck disable=SC2086
python attack_carla.py \
    --capture "${CAPTURE}" \
    --objective "${OBJECTIVE}" \
    --sign "${SIGN}" \
    --attack "${ATTACK}" \
    --iters "${ITERS}" \
    --epsilons ${EPSILONS} \
    --clean-pred "${CLEAN}" \
    --out "${OUT}" \
    --report "${OUT}/reports"

echo
echo "=== round trip: does the dumped .npy carry the div the objective reported? ==="
for REPORT in "${OUT}/reports/of_ev_snn_${LABEL}"_eps*.json; do
    python attack_carla.py --capture "${CAPTURE}" --round-trip "${REPORT}" \
        --objective "${OBJECTIVE}" --epsilons 0 --clean-pred "${CLEAN}" --out "${OUT}"
done

echo
echo "Done: ${LABEL}. Score with avoidance/eval/run_case, then avoidance.eval.sweep."
