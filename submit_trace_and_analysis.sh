#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
test -f run_trace_kerr.sbatch
test -f run_analyze_cache.sbatch
test -f run_render_nt_disk.sbatch

trace_submission=$(sbatch --parsable run_trace_kerr.sbatch)
trace_job="${trace_submission%%;*}"
if [[ ! "$trace_job" =~ ^[0-9]+$ ]]; then
    printf 'Unexpected trace submission result: %s\n' "$trace_submission" >&2
    exit 1
fi
printf 'Trace job: %s\n' "$trace_job"
cache_output="${OUTPUT:-rays_${trace_job}.h5}"

render_submission=$(env -u OUTPUT sbatch --parsable --dependency="afterok:${trace_job}" --kill-on-invalid-dep=yes run_render_nt_disk.sbatch "$cache_output")
render_job="${render_submission%%;*}"
if [[ ! "$render_job" =~ ^[0-9]+$ ]]; then
    printf 'Unexpected render submission result: %s\n' "$render_submission" >&2
    exit 1
fi
printf 'Render job: %s (after trace %s succeeds)\n' "$render_job" "$trace_job"

analysis_submission=$(sbatch --parsable --dependency="afterok:${render_job}" --kill-on-invalid-dep=yes run_analyze_cache.sbatch "$cache_output")
analysis_job="${analysis_submission%%;*}"
if [[ ! "$analysis_job" =~ ^[0-9]+$ ]]; then
    printf 'Unexpected analysis submission result: %s\n' "$analysis_submission" >&2
    exit 1
fi
printf 'Analysis job: %s (after render %s succeeds)\n' "$analysis_job" "$render_job"
printf 'Cache output: %s\n' "$cache_output"
printf 'Analysis log: slurm-analysis-%s.out\n' "$analysis_job"
printf 'Render log: slurm-render-%s.out\n' "$render_job"
