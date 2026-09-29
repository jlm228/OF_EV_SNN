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

# Build the tensors if they are not there, exactly as carla_eval.slurm does. Captures ship
# without them: they are the bulk of a capture's size and regenerable from its raw events. Any
# sequence list means a previous conversion finished, whatever prefix it used.
CAP_ABS="$(cd "${CAPTURE}" && pwd)"
mkdir -p "${CAP_ABS}/tensors"
(
  flock 9
  if [ -n "$(find "${CAP_ABS}/tensors/sequence_lists" -name '*.csv' 2>/dev/null | head -1)" ]; then
    echo "tensors already present at ${CAP_ABS}/tensors -- skipping conversion"
  else
    echo "converting ${CAP_ABS} to the OF_EV_SNN layout"
    ( cd "${CARLA_SCRIPTS_ROOT}" && python groundtruth/inspect_capture.py "${CAP_ABS}" \
        --voxelise --out "${CAP_ABS}/tensors" --capture-id "carla_$(basename "${CAP_ABS}")" )
  fi
) 9>"${CAP_ABS}/tensors.lock"

[ -n "$(find "${CAP_ABS}/tensors/sequence_lists" -name '*.csv' 2>/dev/null | head -1)" ] || {
  echo "conversion did not produce a sequence list under ${CAP_ABS}/tensors" >&2; exit 1; }
[ -f "${MANIFEST}" ] || { echo "no manifest at ${MANIFEST}" >&2; exit 1; }
[ -d "${CARLA_SCRIPTS_ROOT}/attack_core" ] || {
  echo "no attack_core under ${CARLA_SCRIPTS_ROOT} -- set CARLA_SCRIPTS_ROOT" >&2; exit 1; }

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")"
[ -n "${LINE}" ] || { echo "manifest has no line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
read -r OBJECTIVE SIGN ATTACK ITERS EPSILONS <<< "${LINE}"

# The manifest's <attack> is "<optimiser>-<surrogate>" for everything new (sapgd-assgs,
# sapgd-native), and the bare "pgd"/"fgsm" for the runs already on disk. Split it here so the
# manifest stays one column and the labels keep matching the dump directories.
OPTIMISER="${ATTACK%%-*}"
SURROGATE="${ATTACK#*-}"
[ "${SURROGATE}" = "${ATTACK}" ] && SURROGATE="native"

# The paper's setting for SA-PGD is N = 100; a manifest that leaves ITERS at the PGD default
# would quietly run a tenth of the attack.
if [ "${OPTIMISER}" = "sapgd" ] && [ "${ITERS}" -lt 100 ]; then
  echo "warning: ${ATTACK} with --iters ${ITERS}; the published setting is 100" >&2
fi

# ASSG's A is tuned per model and frozen. Passed through the environment so the manifest
# format does not change.
ASSG_ARGS=""
case "${SURROGATE}" in
  assg|assgs) ASSG_ARGS="--assg-A ${ASSG_A:?set ASSG_A for surrogate ${SURROGATE}}" ;;
esac

CLEAN="results/carla_eval/pred/of_ev_snn"
OUT="results/attack/of_ev_snn"
LABEL="${OBJECTIVE}"
[ "${OBJECTIVE}" = "div" ] && LABEL="div_${SIGN}"
# The attack is part of the label: one manifest runs the same objective under both PGD
# and FGSM, which would otherwise write the same output files.
LABEL="${LABEL}_${ATTACK}"

# RAND_INIT=0 starts PGD at the clean input. Separates a bad random start from a bad
# gradient when PGD underperforms FGSM.
NRI=""
if [ "${RAND_INIT:-1}" = "0" ]; then
  NRI="--no-rand-init"
  LABEL="${LABEL}_nri"
fi

# SDA has no epsilon: the manifest's last column carries tau factors (target mode, the core
# matrix) or event budgets from `calibrate --events` (budget mode). SDA_MODE picks which, and
# ITERS is SDA's N rather than a step count.
SDA_ARGS=""
if [ "${OPTIMISER}" = "sda" ]; then
  case "${SDA_MODE:-target}" in
    target) SDA_ARGS="--sda-tau-factors ${EPSILONS}" ;;
    budget) SDA_ARGS="--sda-budgets ${EPSILONS}" ;;
    *) echo "SDA_MODE must be target or budget, got '${SDA_MODE}'" >&2; exit 1 ;;
  esac
  SDA_ARGS="${SDA_ARGS} --sda-iters ${ITERS}"
  SDA_ARGS="${SDA_ARGS} --sda-directions ${SDA_DIRECTIONS:-inject_only}"
  SDA_ARGS="${SDA_ARGS} --sda-rank ${SDA_RANK:-grad}"
  [ -n "${SDA_K_INIT:-}" ] && SDA_ARGS="${SDA_ARGS} --sda-k-init ${SDA_K_INIT}"
  [ -n "${SDA_FD_BATCH:-}" ] && SDA_ARGS="${SDA_ARGS} --sda-fd-batch ${SDA_FD_BATCH}"
  [ -n "${SDA_EPE_MARGIN:-}" ] && SDA_ARGS="${SDA_ARGS} --sda-epe-margin ${SDA_EPE_MARGIN}"
  LEVEL_ARGS="${SDA_ARGS}"
else
  LEVEL_ARGS="--iters ${ITERS} ${NRI} --epsilons ${EPSILONS}"
fi

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
    --attack "${OPTIMISER}" \
    --surrogate "${SURROGATE}" ${ASSG_ARGS} \
    ${LEVEL_ARGS} \
    --clean-pred "${CLEAN}" \
    --out "${OUT}" \
    --report "${OUT}/reports"

echo
echo "=== round trip: does the dumped .npy carry the div the objective reported? ==="
RT_FAIL=0
for REPORT in "${OUT}/reports/of_ev_snn_${LABEL}"_eps*.json; do
    python attack_carla.py --capture "${CAPTURE}" --round-trip "${REPORT}" \
        --objective "${OBJECTIVE}" --epsilons 0 --clean-pred "${CLEAN}" --out "${OUT}" || RT_FAIL=1
done
[ "${RT_FAIL}" = "0" ] || { echo "round trip FAILED on at least one epsilon" >&2; exit 1; }

echo
echo "Done: ${LABEL}. Score with avoidance/eval/run_case, then avoidance.eval.sweep."
