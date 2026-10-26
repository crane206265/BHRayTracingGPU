#!/bin/bash -l
#SBATCH --job-name=kerr-trace
#SBATCH --partition=defaults
#SBATCH --nodelist=reber
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
test -f src/examples/trace_Kerr.py

if ! command -v conda >/dev/null 2>&1; then
    module load miniforge/2026.09
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grrt
set -u

export JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1

hostname
date
nvidia-smi
python -c "import sys, jax; print(sys.executable); print('JAX:', jax.__version__); print('Devices:', jax.devices())"
python -u -m src.examples.trace_Kerr
date
