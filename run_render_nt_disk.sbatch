#!/bin/bash -l
#SBATCH --job-name=nt-render
#SBATCH --partition=defaults
#SBATCH --nodelist=reber
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-render-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
test -f src/examples/render_NT_disk.py

CACHE_INPUT="${1:-${CACHE_ID:-}}"
if [[ -z "$CACHE_INPUT" ]]; then
    printf 'Usage: sbatch run_render_nt_disk.sbatch CACHE_ID_OR_OUTPUT_FILENAME\n' >&2
    exit 2
fi
if [[ "$CACHE_INPUT" =~ ^[0-9]+$ ]]; then
    CACHE_INPUT="rays_${CACHE_INPUT}.h5"
fi
case "$CACHE_INPUT" in
    /*) CACHE_PATH="$CACHE_INPUT" ;;
    "~/"*) CACHE_PATH="${HOME}/${CACHE_INPUT:2}" ;;
    *) CACHE_PATH="cache/$CACHE_INPUT" ;;
esac
CACHE_LABEL="${CACHE_PATH##*/}"
CACHE_LABEL="${CACHE_LABEL%.*}"
CACHE_LABEL="${CACHE_LABEL#rays_}"
OUTPUT="${RENDER_OUTPUT:-${OUTPUT:-cache/render_${CACHE_LABEL}_${SLURM_JOB_ID}.npz}}"
if [[ ! -f "$CACHE_PATH" ]]; then
    printf 'Cache file not found: %s\n' "$CACHE_PATH" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    module load miniforge/2026.09
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grrt
set -u

export JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

hostname
date
nvidia-smi
printf 'Cache input: %s\nCache: %s\nOutput: %s\n' "$CACHE_INPUT" "$CACHE_PATH" "$OUTPUT"
python -c "import sys, jax; print(sys.executable); print('JAX:', jax.__version__); print('Devices:', jax.devices())"
python -u -m src.examples.render_NT_disk \
    --cache="$CACHE_PATH" \
    --output="$OUTPUT"
date
