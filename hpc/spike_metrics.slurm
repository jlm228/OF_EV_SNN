#!/bin/bash
# Record NeuroBench-style spike metrics for OF_EV_SNN on a GPU.
#
#   sbatch hpc/spike_metrics.slurm                      # synthetic smoke test, 2 windows
#   sbatch hpc/spike_metrics.slurm <capture_dir> 8      # 8 real windows from a capture
#
# WINDOWS is POSITIONAL for the same reason carla_eval.slurm's arguments are: this cluster's
# sbatch does not honour an in-script `#SBATCH --export=ALL`, so variables set at submission
# time silently do not arrive.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:0
#SBATCH --mem=64G
#SBATCH --job-name=of_ev_snn_spikes
#SBATCH --output=logs/spike_metrics_%j.out
#SBATCH --qos=bbgpu

set -euo pipefail

CAPTURE="${1:-}"
WINDOWS="${2:-2}"

source "${SLURM_SUBMIT_DIR}/hpc/env.sh"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

CARLA_SCRIPTS_ROOT="${CARLA_SCRIPTS_ROOT:-../CARLA-hpc-scripts}"
export CARLA_SCRIPTS_ROOT
[ -f "${CARLA_SCRIPTS_ROOT}/snnmetrics/probe.py" ] || {
  echo "ERROR: ${CARLA_SCRIPTS_ROOT}/snnmetrics/probe.py not found."
  echo "       Set CARLA_SCRIPTS_ROOT to your CARLA-hpc-scripts checkout."; exit 1; }

OUT="results/spike_metrics"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"
echo

# The metric maths is environment-independent, so its tests run here too.
( cd "${CARLA_SCRIPTS_ROOT}" && python -m snnmetrics.tests.run_all )
echo

if [ -z "${CAPTURE}" ]; then
    echo "--- synthetic smoke test, ${WINDOWS} window(s) ---"
    python -u spike_metrics.py --synthetic --windows "${WINDOWS}" \
        --out "${OUT}" --name of_ev_snn_synthetic
else
    CAP_ABS="$(cd "${CAPTURE}" && pwd)"
    # Build the tensors if they are not there, under the same lock carla_eval.slurm uses so a
    # concurrent job cannot read a half-written conversion. This model reads the ON/OFF count
    # layout from inspect_capture.py --voxelise, NOT the voxels carla_to_voxel.py writes.
    (
      flock 9
      if [ -n "$(find "${CAP_ABS}/tensors/sequence_lists" -name '*.csv' 2>/dev/null | head -1)" ]; then
        echo "tensors already present at ${CAP_ABS}/tensors"
      else
        echo "converting ${CAP_ABS} to the OF_EV_SNN layout"
        ( cd "${CARLA_SCRIPTS_ROOT}" && python groundtruth/inspect_capture.py "${CAP_ABS}" \
              --voxelise --out "${CAP_ABS}/tensors" --capture-id "carla_$(basename "${CAP_ABS}")" )
      fi
    ) 9>"${CAP_ABS}/tensors.lock"
    [ -n "$(find "${CAP_ABS}/tensors/sequence_lists" -name '*.csv' 2>/dev/null | head -1)" ] || {
      echo "conversion produced no sequence list under ${CAP_ABS}/tensors" >&2; exit 1; }
    echo "--- ${CAP_ABS}, ${WINDOWS} window(s) ---"
    python -u spike_metrics.py --tensors "${CAP_ABS}/tensors" \
        --id "carla_$(basename "${CAP_ABS}")" --windows "${WINDOWS}" \
        --out "${OUT}" --name "of_ev_snn_$(basename "${CAP_ABS}")"
fi

echo
echo "Done. Metrics in ${OUT}:"
ls -la "${OUT}"
