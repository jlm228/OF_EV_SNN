#!/bin/bash
# Score every attacked cell through the frozen planner, aggregate, and render the figures.
#
#   sbatch hpc/score_attack.slurm <capture_dir> [model]
#
# Chained behind the attack array by submit_attack_sweep.sh. CPU only -- scoring is seconds per
# case and needs no GPU, so this must not sit in the GPU queue.
#
# The avoidance package is model-free (numpy/pandas/cv2, no torch), so it runs in this repo's
# environment and scores every model through one implementation.

#SBATCH --job-name=score_attack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:0:0
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p hpc/logs

USAGE="usage: sbatch hpc/score_attack.slurm <capture_dir> [model]"
CAPTURE="${1:?${USAGE}}"
MODEL="${2:-of_ev_snn}"

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
CAP_ABS="$(cd "${CAPTURE}" && pwd)"
SCEN="$(basename "${CAP_ABS}")"
REPO="$(pwd)"   # already cd'd to SLURM_SUBMIT_DIR; $0 is Slurm's spooled copy
ATTACK_ROOT="${REPO}/results/attack/${MODEL}"
CLEAN_PRED="${REPO}/results/carla_eval/pred/${MODEL}"
# The prefix the capture was voxelised under; not always "carla_<basename>", which is what
# run_case would otherwise assume.
_PROBE="$(find "${CLEAN_PRED}" -name "*_[0-9][0-9][0-9][0-9].npy" 2>/dev/null | head -1)"
CAPTURE_ID="carla_${SCEN}"
[ -n "${_PROBE}" ] && CAPTURE_ID="$(basename "${_PROBE}" .npy | sed "s/_[0-9]\{4\}$//")"

# Results land in the CARLA-hpc-scripts checkout, in the same directory SDformerFlow's scoring
# job writes to, so the grid figure spans all three models rather than one per repo.
cd "${CARLA_SCRIPTS_ROOT}"
RES="results/attack-${SCEN}"
mkdir -p "${RES}/reports"

# The controls go in the SAME directory as the attacked cells. viz/bev.py refuses to run
# without no_action (it supplies the recorded path), sweep.py refuses without gt_oracle (it is
# what deviation is measured against), and without the floor a non-collision means nothing.
for COND in no_action gt_oracle; do
    if [ ! -f "${RES}/${SCEN}_none_${COND}.json" ]; then
        echo "=== control: ${COND} ==="
        python -m avoidance.eval.run_case --capture "${CAP_ABS}" --model none \
            --condition "${COND}" --out "${RES}"
    fi
done

# The model's own clean run: the epsilon-0 anchor every pair figure is read against.
if [ ! -d "${CLEAN_PRED}" ]; then
    echo "WARNING: no clean predictions at ${CLEAN_PRED}."
    echo "         Without them there is no epsilon-0 anchor, so every pair figure and the"
    echo "         whole grid will be empty. Run hpc/carla_eval.slurm for this capture."
fi
if [ -d "${CLEAN_PRED}" ] && [ ! -f "${RES}/${SCEN}_${MODEL}_clean.json" ]; then
    echo "=== clean anchor: ${MODEL} ==="
    python -m avoidance.eval.run_case --capture "${CAP_ABS}" --model "${MODEL}" \
        --condition clean --pred "${CLEAN_PRED}" --capture-id "${CAPTURE_ID}" --out "${RES}"
fi

# One case per (objective, epsilon). --model is a free string and becomes the result filename,
# so encoding the cell into it needs no change to run_case.
echo
echo "=== attacked cells ==="
for EPS_DIR in "${ATTACK_ROOT}"/*/eps*/; do
    [ -d "${EPS_DIR}" ] || continue
    EPS_TAG="$(basename "${EPS_DIR}")"
    OBJECTIVE="$(basename "$(dirname "${EPS_DIR}")")"
    EPS="${EPS_TAG#eps}"
    # eps 0 is the clean case and is objective-independent; scoring it once per objective would
    # put five identical rows in sweep.csv.
    if [ "$(python -c "print(float('${EPS}') == 0)")" = "True" ]; then continue; fi
    NAME="${MODEL}__${OBJECTIVE}__eps${EPS}"
    echo "  ${NAME}"
    python -m avoidance.eval.run_case --capture "${CAP_ABS}" --model "${NAME}" \
        --condition clean --pred "${EPS_DIR}" --capture-id "${CAPTURE_ID}" --out "${RES}"
done

# Reports live beside the dumps; sweep.py searches recursively and joins on
# (model, objective, epsilon, window). Their filenames carry the model, so gathering all three
# models' reports here cannot lose one to a name collision.
cp -f "${ATTACK_ROOT}/reports/"*.json "${RES}/reports/" 2>/dev/null || true

echo
# The prediction root is the PARENT of the per-model directories: make_figures.sh walks
# <root>/<model>/<objective>/eps<value>/ and takes each objective's eps 0 as its clean
# reference.
bash avoidance/viz/make_figures.sh "${RES}" "${CAP_ABS}" "${REPO}/results/attack"
