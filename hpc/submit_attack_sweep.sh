#!/bin/bash
# The whole OF_EV_SNN attack sweep, from one command.
#
#   bash hpc/submit_attack_sweep.sh <capture_dir>
#   SMOKE=1 bash hpc/submit_attack_sweep.sh <capture_dir>     # 5 windows, 2 epsilons
#
# Submits: a GPU array (one task per objective, all epsilons inside each task), then a CPU job
# chained on --dependency=afterok that scores every cell through the frozen planner, aggregates
# to sweep.csv, and renders the figures.
#
# TWO SUBMISSIONS COVER THE STUDY, not one. OF_EV_SNN needs spikingjelly.clock_driven and
# SDformerFlow needs spikingjelly.activation_based, in separate venvs -- the same reason
# score_flow.py scores every model from dumped predictions rather than in one process. Run
# SDformerFlow/hpc/submit_attack_sweep.sh for the other two models.
#
# Prerequisites, both fatal if missing:
#   * the clean run: sbatch hpc/carla_eval.slurm <capture_dir>
#   * the band:      python -m attack_core.band --capture <capture_dir>   (in CARLA-hpc-scripts)

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p hpc/logs results/attack/of_ev_snn

USAGE="usage: bash hpc/submit_attack_sweep.sh <capture_dir>"
CAPTURE="${1:?${USAGE}}"
CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"

CLEAN="results/carla_eval/pred/of_ev_snn"
[ -d "${CLEAN}" ] || {
  echo "ERROR: no clean predictions at ${CLEAN}."
  echo "       sbatch hpc/carla_eval.slurm ${CAPTURE}"; exit 1; }
[ -f "${CAPTURE}/attack_band.json" ] || {
  echo "ERROR: no ${CAPTURE}/attack_band.json."
  echo "       Compute it once, in an environment with avoidance's dependencies:"
  echo "         cd ${CARLA_SCRIPTS_ROOT} && python -m attack_core.band --capture ${CAPTURE}"
  exit 1; }

# Calibrated per model, because epsilon does not mean the same thing across representations:
# OF_EV_SNN's ON/OFF counts run [0, 19] at 3.09% occupancy against the swin voxel's
# [-14.20, 10.89] at 10.24%.
EPS_FILE="${EPS_FILE:-results/calibrated_epsilons.txt}"
if [ -f "${EPS_FILE}" ]; then
    EPSILONS="0.0 $(cat "${EPS_FILE}")"
else
    EPSILONS="${EPSILONS:-0.0 0.01 0.02 0.05 0.1 0.2}"
    echo "note: no ${EPS_FILE}; using ${EPSILONS}"
    echo "      calibrate with: python -m attacks.fgsm_pgd.calibrate_epsilon"
fi


# A sweep owns its output directory. Budgets are keyed by value, so a later sweep with a
# different ramp leaves the previous one's cells behind and everything downstream scores both.
# KEEP_PREVIOUS=1 to append instead.
if [ "${KEEP_PREVIOUS:-0}" = "0" ]; then
  for D in results/attack/of_ev_snn; do
    [ -d "${D}" ] && { echo "clearing ${D}"; rm -rf "${D}"; }
  done
fi

RAND_INIT="${RAND_INIT:-1}"
ITERS="${ITERS:-10}"
ATTACK="${ATTACK:-pgd}"
# FGSM is one step by construction; the report records whatever the manifest says.
[ "${ATTACK}" = "fgsm" ] && ITERS=1
if [ "${SMOKE:-0}" != "0" ]; then
    EPSILONS="0.0 0.05"
    ITERS=4
    echo "[SMOKE] epsilons ${EPSILONS}, ${ITERS} iters"
fi



# Stage 6 check 3: one epsilon past the calibrated top, to confirm a larger budget does
# more damage rather than less. Multiplier is deliberately small: far outside the band the
# attack stops steering the flow and simply destroys it, which tests nothing. Override
# with EPS_HIGH_MULT.
EPS_HUGE="$(python -c "import sys; print('%g' % (${EPS_HIGH_MULT:-2} * max(float(v) for v in sys.argv[1:])))" ${EPSILONS})"
SWEEP_EPS="${EPSILONS} ${EPS_HUGE}"

# One line per array task: objective, sign, attack, iters, then the whole epsilon ramp.
# epsilon 0 appears once per line and is the clean case; it is objective-independent, so
# sweep.py collapses the duplicates when it builds the table.
# Per-submission override of the slurm header's --time.
TIME="${TIME:-}"
SB_TIME=""
if [ -n "${TIME}" ]; then SB_TIME="--time=${TIME}"; fi

MANIFEST="hpc/logs/attack_grid_$(basename "${CAPTURE}").txt"
{
  echo "random_sign  none      ${ATTACK} ${ITERS} ${SWEEP_EPS}"
  echo "epe_global   none      ${ATTACK} ${ITERS} ${SWEEP_EPS}"
  echo "epe_masked   none      ${ATTACK} ${ITERS} ${SWEEP_EPS}"
  echo "div          suppress  ${ATTACK} ${ITERS} ${SWEEP_EPS}"
  echo "div          inflate   ${ATTACK} ${ITERS} ${SWEEP_EPS}"
  # FGSM rows for the one-step-vs-iterative comparison, both signs. Skipped when
  # ATTACK is already fgsm, which would duplicate the rows above.
  if [ "${ATTACK}" != "fgsm" ]; then
    echo "div          suppress  fgsm 1 ${SWEEP_EPS}"
    echo "div          inflate   fgsm 1 ${SWEEP_EPS}"
  fi
} > "${MANIFEST}"
N=$(wc -l < "${MANIFEST}")

echo "manifest  ${MANIFEST} (${N} tasks)"
cat "${MANIFEST}" | sed 's/^/    /'
echo

ARRAY_ID=$(sbatch --parsable --array=1-"${N}" ${SB_TIME} \
    --export=ALL,RAND_INIT="${RAND_INIT}" \
    hpc/attack_carla.slurm "${CAPTURE}" "${MANIFEST}")
echo "attack array : job ${ARRAY_ID} (1-${N})"

# afterany, not afterok: one failed cell must not block scoring and figures for the
# rest. sweep.py reports what is missing.
SCORE_ID=$(sbatch --parsable --dependency=afterany:"${ARRAY_ID}" \
    hpc/score_attack.slurm "${CAPTURE}")
echo "score+figures: job ${SCORE_ID} (after ${ARRAY_ID})"
echo
echo "Watch with: squeue --me"
echo "A failed objective reruns alone:  sbatch --array=<index> hpc/attack_carla.slurm ${CAPTURE} ${MANIFEST}"
