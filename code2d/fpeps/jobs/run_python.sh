#!/bin/bash
#SBATCH --job-name=clean-fpeps-python
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm, never on a login node}"
[[ ${SLURM_JOB_PARTITION:-} == preempt ]] || exit 2
: "${CLEAN_ROOT:?Set CLEAN_ROOT at submission}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Reuse the installed Python dependency cache, as Julia reuses read-only depots.
# Scientific source and data do not live in this cache.
export PYTHONPATH="${FPEPS_PYTHON_DEPS:-/ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps/benchmark/.plot_deps}${PYTHONPATH:+:$PYTHONPATH}"
cd "$CLEAN_ROOT/code2d/fpeps"
exec python "$@"
