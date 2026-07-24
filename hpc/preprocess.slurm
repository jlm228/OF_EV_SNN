#!/bin/bash
# Convert raw DSEC sequences into the .npy tensors the dataloader reads.
#
# One array task per sequence, because generate_files() handles sequences independently
# and zurich_city_11_b alone is ~45% of the validation split.
#
#   sbatch --array=0-4 hpc/preprocess.slurm      # all 5 sequences in parallel
#   sbatch --array=0   hpc/preprocess.slurm      # thun_00_a only (env smoke test)
#
# This is a pure-Python event-binning loop: single-threaded, and it never touches a GPU.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:0:0
#SBATCH --mem=16G
#SBATCH --job-name=of_ev_snn_prep
#SBATCH --output=logs/prep_%A_%a.out
#SBATCH --account=_projectaccount_          # TODO: your BEAR project account
#SBATCH --qos=bbdefault                     # TODO: bbshort only allows very short jobs

set -e

module purge; module load bluebear
module load Python/3.11.3-GCCcore-12.3.0    # TODO: confirm with `module avail Python`

source "${HOME}/venvs/of_ev_snn/bin/activate"

# imageio fetches the FreeImage library on first use and compute nodes have no internet,
# so point it at a cache populated on the login node (see hpc/README.md).
export IMAGEIO_USERDIR="${HOME}/.imageio"

SEQUENCES=(thun_00_a zurich_city_02_d zurich_city_03_a zurich_city_08_a zurich_city_11_b)
SEQUENCE="${SEQUENCES[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "Preprocessing ${SEQUENCE} on $(hostname)"

# main.py resolves both its imports and root='../data/dataset' relative to this directory.
cd "${SLURM_SUBMIT_DIR}/dsec_dataset_lite"
python main.py "${SEQUENCE}"

echo "Finished ${SEQUENCE}"
