#!/bin/bash
# Submit one clean-eval GPU job per capture.
#
#   bash hpc/submit_clean_sweep.sh <capture_dir> [<capture_dir> ...]
#
# The eval job builds the capture's tensors if they are missing, so raw captures are fine.
# Predictions land in results/carla_eval/pred/of_ev_snn as <capture_id>_<window>.npy.
# Captures with no events.npy, or already predicted, are skipped; RERUN=1 submits them anyway.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p hpc/logs

[ "$#" -ge 1 ] || { echo "usage: bash hpc/submit_clean_sweep.sh <capture_dir> [...]" >&2; exit 1; }

PRED="results/carla_eval/pred/of_ev_snn"
# DEPEND=<jobid> holds every job until that one finishes, so a clean
# sweep can be queued behind an attack array without competing for GPUs.
SB_DEP=""
if [ -n "${DEPEND:-}" ]; then SB_DEP="--dependency=afterany:${DEPEND}"; fi

SUBMITTED=0
SKIPPED=0

for CAPTURE in "$@"; do
  SCEN="$(basename "${CAPTURE}")"
  if [ ! -f "${CAPTURE}/events.npy" ]; then
    echo "SKIP ${SCEN}: no events.npy"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  # The capture id is the tensor filename prefix, not always carla_<basename>.
  PROBE="$(find "${CAPTURE}/tensors/event_tensors" -name "*_[0-9][0-9][0-9][0-9].npy" \
           2>/dev/null | head -1)"
  CAPTURE_ID="carla_${SCEN}"
  [ -n "${PROBE}" ] && CAPTURE_ID="$(basename "${PROBE}" .npy | sed 's/_[0-9]\{4\}$//')"

  if [ "${RERUN:-0}" = "0" ] && [ -n "$(find "${PRED}" -name "${CAPTURE_ID}_*.npy" \
                                        2>/dev/null | head -1)" ]; then
    echo "SKIP ${SCEN}: predictions already present for ${CAPTURE_ID}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  JOB=$(sbatch --parsable ${SB_DEP} hpc/carla_eval.slurm "${CAPTURE}")
  echo "submitted ${SCEN}  (${CAPTURE_ID})  job ${JOB}"
  SUBMITTED=$((SUBMITTED + 1))
done

echo
echo "${SUBMITTED} submitted, ${SKIPPED} skipped"
echo "Watch with: squeue --me"
