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
#SBATCH --mem=32G
#SBATCH --job-name=of_ev_snn_carla
#SBATCH --output=logs/carla_eval_%j.out
#SBATCH --qos=bbgpu

set -euo pipefail

USAGE="usage: sbatch hpc/carla_eval.slurm <capture_dir>"
CAPTURE="${1:?${USAGE}}"

source "${SLURM_SUBMIT_DIR}/hpc/env.sh"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ ! -d "${CAPTURE}/tensors" ]; then
    echo "no ${CAPTURE}/tensors -- run inspect_capture.py --voxelise --out ${CAPTURE}/tensors" >&2
    exit 1
fi

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
