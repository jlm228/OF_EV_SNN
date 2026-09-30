#!/bin/bash
# The whole sweep over every capture, from one command.
#
#   bash hpc/submit_all_captures.sh <captures_dir>
#   ATTACK=sda-pdsg SDA_MODE=target TAU_PROBE=<record.json> \
#       bash hpc/submit_all_captures.sh ../CARLA-hpc-scripts/captures/135_scenarios
#   LIMIT=5 DRY_RUN=1 bash hpc/submit_all_captures.sh <captures_dir>    # see what it would do
#
# submit_attack_sweep.sh handles ONE capture and CLEARS results/attack/of_ev_snn on the way in,
# so calling it in a loop directly would leave only the last capture's cells. This clears once,
# then passes KEEP_PREVIOUS=1 to every call.
#
# SDA needs a per-capture target: the tau factor at which that capture's outcome tier changes,
# from a tau_probe record. A capture the probe never moved has no meaningful target, so it is
# SKIPPED rather than attacked against a threshold the planner does not care about. Set
# SDA_FALLBACK to attack those at a fixed factor instead.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p hpc/logs

USAGE="usage: bash hpc/submit_all_captures.sh <captures_dir>"
CAPTURES="${1:?${USAGE}}"
CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"

ATTACK="${ATTACK:-sapgd-assgs}"
OPTIMISER="${ATTACK%%-*}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAGGER="${STAGGER:-2}"

CLEAN="results/carla_eval/pred/of_ev_snn"
[ -d "${CLEAN}" ] || {
  echo "ERROR: no clean predictions at ${CLEAN}; run hpc/carla_eval.slurm first" >&2; exit 1; }

mapfile -t DIRS < <(ls -d "${CAPTURES}"/[0-9][0-9][0-9] 2>/dev/null)
[ "${#DIRS[@]}" -gt 0 ] || { echo "no captures under ${CAPTURES}" >&2; exit 1; }
[ "${LIMIT}" -gt 0 ] && DIRS=("${DIRS[@]:0:${LIMIT}}")

# SDA's targets come from the probe, so the record is required up front rather than discovered
# missing 60 captures in.
TAU_ABS=""
if [ "${OPTIMISER}" = "sda" ]; then
  TAU_PROBE="${TAU_PROBE:?set TAU_PROBE to a tau_probe JSON for SDA_MODE=target}"
  [ -f "${TAU_PROBE}" ] || { echo "no tau probe record at ${TAU_PROBE}" >&2; exit 1; }
  # Absolute, because tau_levels is resolved from the scripts repo, not from here.
  TAU_ABS="$(cd "$(dirname "${TAU_PROBE}")" && pwd)/$(basename "${TAU_PROBE}")"
fi

echo "captures   ${CAPTURES} (${#DIRS[@]} to submit)"
echo "attack     ${ATTACK}"
[ "${OPTIMISER}" = "sda" ] && echo "targets    ${TAU_PROBE}"
echo

# Clear ONCE, before the loop, for the same reason submit_attack_sweep.sh clears: a previous
# sweep with a different ramp leaves cells behind that everything downstream would score too.
if [ "${KEEP_PREVIOUS:-0}" = "0" ] && [ "${DRY_RUN}" = "0" ]; then
  echo "clearing results/attack/of_ev_snn (KEEP_PREVIOUS=1 to append)"
  rm -rf results/attack/of_ev_snn
fi
mkdir -p results/attack/of_ev_snn

SUBMITTED=0
SKIPPED=""
LOG="hpc/logs/submit_all_$(date +%Y%m%d_%H%M%S).txt"

for CAP in "${DIRS[@]}"; do
  ID="$(basename "${CAP}")"

  if [ ! -f "${CAP}/attack_band.json" ]; then
    SKIPPED="${SKIPPED} ${ID}:no-band"
    continue
  fi

  LEVELS=""
  if [ "${OPTIMISER}" = "sda" ]; then
    # tau_levels exits 1 when this capture never reached a tier change.
    if ! LEVELS=$( cd "${CARLA_SCRIPTS_ROOT}" && python -m attack_core.tau_levels \
                     "${TAU_ABS}" "${ID}" \
                     ${SDA_FALLBACK:+--fallback "${SDA_FALLBACK}"} 2>/dev/null ); then
      SKIPPED="${SKIPPED} ${ID}:no-zeta"
      continue
    fi
  fi

  if [ "${DRY_RUN}" != "0" ]; then
    echo "would submit ${ID}${LEVELS:+ at tau ${LEVELS}}"
    SUBMITTED=$((SUBMITTED + 1))
    continue
  fi

  echo "=== ${ID}${LEVELS:+ (tau ${LEVELS})} ===" | tee -a "${LOG}"
  # export, not an env-var prefix: a prefix would word-split a multi-value level list.
  [ -n "${LEVELS}" ] && export SDA_LEVELS="${LEVELS}"
  KEEP_PREVIOUS=1 ATTACK="${ATTACK}" \
    bash hpc/submit_attack_sweep.sh "${CAP}" 2>&1 | tee -a "${LOG}"
  SUBMITTED=$((SUBMITTED + 1))

  # The scheduler accepts these faster than it can process them; a short pause keeps the
  # submission log readable and avoids tripping a rate limit partway through.
  sleep "${STAGGER}"
done

echo
echo "submitted ${SUBMITTED} of ${#DIRS[@]} captures"
[ -n "${SKIPPED}" ] && {
  echo "skipped:${SKIPPED}"
  echo "  no-band: run 'python -m attack_core.band --capture <dir>' in ${CARLA_SCRIPTS_ROOT}"
  echo "  no-zeta: the tau probe never moved this capture's outcome tier. Raise the probe's"
  echo "           ceiling, or set SDA_FALLBACK to attack it at a fixed factor anyway"; }
[ "${DRY_RUN}" = "0" ] && echo "log: ${LOG}"
echo "watch with: squeue --me"
exit 0
