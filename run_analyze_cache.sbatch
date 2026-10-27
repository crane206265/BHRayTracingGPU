#!/bin/bash -l
#SBATCH --job-name=cache-analysis
#SBATCH --partition=defaults
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-analysis-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
test -f src/examples/analyze_cache.py

CACHE_INPUT="${1:-${CACHE_JOB_ID:-}}"
if [[ -z "$CACHE_INPUT" ]]; then
    printf 'Usage: sbatch run_analyze_cache.sbatch TRACE_JOB_ID_OR_OUTPUT_FILENAME\n' >&2
    exit 2
fi
if [[ "$CACHE_INPUT" =~ ^[0-9]+$ ]]; then
    CACHE_INPUT="rays_${CACHE_INPUT}.h5"
fi

if ! command -v conda >/dev/null 2>&1; then
    module load miniforge/2026.09
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grrt
set -u

export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CACHE_LABEL="${CACHE_INPUT##*/}"
CACHE_LABEL="${CACHE_LABEL%.*}"
REPORT_PREFIX="${REPORT_PREFIX:-cache/reports/${CACHE_LABEL}_analysis_${SLURM_JOB_ID}}"
ANALYSIS_CHUNK_SAMPLES="${ANALYSIS_CHUNK_SAMPLES:-65536}"

hostname
date
python -u -m src.examples.analyze_cache \
    --cache-output="$CACHE_INPUT" \
    --report-prefix="$REPORT_PREFIX" \
    --chunk-samples="$ANALYSIS_CHUNK_SAMPLES"
date
