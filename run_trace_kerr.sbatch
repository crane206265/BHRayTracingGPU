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

MASS="${MASS:-1.0}"
SPIN="${SPIN:-0.0}"
RADIUS="${RADIUS:-50.0}"
INC="${INC:-60.0}"
FREQ="${FREQ:-1.0}"
FOV="${FOV:-20.0}"
PPD="${PPD:-0.25}"
ESCAPE_FACTOR="${ESCAPE_FACTOR:-2.0}"
EPSILON="${EPSILON:-0.01}"
H="${H:--0.01}"
CHUNK_STEPS="${CHUNK_STEPS:-256}"
MAX_CHUNKS="${MAX_CHUNKS:-100}"
OUTPUT="${OUTPUT:-rays_${SLURM_JOB_ID}.h5}"

hostname
date
nvidia-smi
python -c "import sys, jax; print(sys.executable); print('JAX:', jax.__version__); print('Devices:', jax.devices())"
python -u -m src.examples.trace_Kerr \
    --mass="$MASS" \
    --spin="$SPIN" \
    --radius="$RADIUS" \
    --inc="$INC" \
    --freq="$FREQ" \
    --fov="$FOV" \
    --ppd="$PPD" \
    --escape-factor="$ESCAPE_FACTOR" \
    --epsilon="$EPSILON" \
    --h="$H" \
    --chunk-steps="$CHUNK_STEPS" \
    --max-chunks="$MAX_CHUNKS" \
    --output="$OUTPUT"
date
