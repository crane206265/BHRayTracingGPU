#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

trace_submission=$(sbatch --parsable run_trace_kerr.sbatch)
trace_job="${trace_submission%%;*}"
if [[ ! "$trace_job" =~ ^[0-9]+$ ]]; then
    printf 'Unexpected trace submission result: %s\n' "$trace_submission" >&2
    exit 1
fi
printf 'Trace job: %s\n' "$trace_job"
cache_output="${OUTPUT:-rays_${trace_job}.h5}"
analysis_submission=$(sbatch --parsable --dependency="afterok:${trace_job}" --kill-on-invalid-dep=yes run_analyze_cache.sbatch "$cache_output")
analysis_job="${analysis_submission%%;*}"
printf 'Analysis job: %s (after trace %s succeeds)\n' "$analysis_job" "$trace_job"
printf 'Cache output: %s\n' "$cache_output"
printf 'Analysis log: slurm-analysis-%s.out\n' "$analysis_job"
