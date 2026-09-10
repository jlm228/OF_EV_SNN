#!/bin/bash
# Run OF_EV_SNN over a CARLA capture and dump the raw predicted flow fields.
#
#   sbatch hpc/carla_eval.slurm <capture_dir>
#
# The capture must already carry <capture_dir>/tensors, which capture.slurm writes with
# inspect_capture.py --voxelise. Predictions land in results/carla_eval/pred/of_ev_snn as
# <capture_id>_<window>.npy, which is what CARLA-hpc-scripts/avoidance and score_flow.py read.
#
# The capture id is the tensor filenames' prefix, set when the capture was voxelised: the
# capture directory's basename unless --capture-id overrode it.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2:0:0
#SBATCH --mem=64G
#SBATCH --job-name=of_ev_snn_carla
#SBATCH --output=logs/carla_eval_%j.out
#SBATCH --qos=bbgpu

set -euo pipefail

USAGE="usage: sbatch hpc/carla_eval.slurm <capture_dir>"
CAPTURE="${1:?${USAGE}}"

source "${SLURM_SUBMIT_DIR}/hpc/env.sh"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
CAP_ABS="$(cd "${CAPTURE}" && pwd)"

# Build the tensors if they are not there. Captures ship without them; they are the bulk of a
# capture's size and are regenerable from its raw events. Any sequence list means a previous
# conversion finished, whatever prefix it used.
mkdir -p "${CAP_ABS}/tensors"
(
  flock 9
  if [ -n "$(find "${CAP_ABS}/tensors/sequence_lists" -name '*.csv' 2>/dev/null              | head -1)" ]; then
    echo "tensors already present at ${CAP_ABS}/tensors -- skipping conversion"
  else
    echo "converting ${CAP_ABS} to the OF_EV_SNN layout"
    ( cd "${CARLA_SCRIPTS_ROOT}" && python inspect_capture.py "${CAP_ABS}"         --voxelise --out "${CAP_ABS}/tensors" --capture-id "carla_$(basename "${CAP_ABS}")" )
  fi
) 9>"${CAP_ABS}/tensors.lock"

[ -n "$(find "${CAP_ABS}/tensors/sequence_lists" -name '*.csv' 2>/dev/null | head -1)" ] || {
  echo "conversion did not produce a sequence list under ${CAP_ABS}/tensors" >&2; exit 1; }

PRED="results/carla_eval/pred/of_ev_snn"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

# Counted without a pipe into head: under `set -o pipefail` head exits first, ls dies of
# SIGPIPE, and the non-zero pipeline status trips `set -e` on a progress message.
echo "=== capture ${CAPTURE} ==="
echo "gt tensors: $(find "${CAPTURE}/tensors/gt_tensors" -name '*.npy' | wc -l)"

python carla_epe_eval.py --carla "${CAPTURE}" --dump-pred "${PRED}"

echo "=== dumped ==="
echo "predictions: $(find "${PRED}" -name '*.npy' | wc -l) in ${PRED}"
