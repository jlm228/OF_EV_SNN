# Load the modules and virtualenv this project needs. Source it, don't execute it:
#
#   source hpc/env.sh
#
# Every new shell (login, srun session, batch job) starts without these, so this runs
# at the top of an interactive session and of each Slurm script.

# Paths are derived from THIS file's own location, so moving the whole project
# (repo + sibling venvs/) needs no edits here. Layout assumed:
#     <parent>/OF_EV_SNN/     <- the repo (this file is at OF_EV_SNN/hpc/env.sh)
#     <parent>/venvs/of_ev_snn <- the virtualenv
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <repo>/hpc
REPO_DIR="$(dirname "${_ENV_DIR}")"                        # <repo>
PARENT_DIR="$(dirname "${REPO_DIR}")"                      # <parent> (holds repo + venvs)

# venv sits beside the repo; override with `export OF_EV_SNN_VENV=/path/to/venv`.
VENV_DIR="${OF_EV_SNN_VENV:-${PARENT_DIR}/venvs/of_ev_snn}"

module purge
module load bluebear
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0

source "${VENV_DIR}/bin/activate"

# imageio downloads the FreeImage library on first use, which compute nodes cannot do,
# so it must point at a cache populated on the login node. HOME is deliberate here: it is
# a tiny (~5 MB) shared-filesystem cache, unaffected by where the bulky project data lives.
export IMAGEIO_USERDIR="${HOME}/.imageio"
