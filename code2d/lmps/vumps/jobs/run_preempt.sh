#!/bin/bash
#SBATCH --job-name=clean-vumps
#SBATCH --cluster=smp
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch --partition=preempt; do not run on login nodes}"
if [[ ${SLURM_JOB_PARTITION:-} != preempt ]]; then
    echo 'Refusing to run outside preempt' >&2
    exit 2
fi
: "${CLEAN_ROOT:?Set CLEAN_ROOT to the absolute clean directory when submitting}"
test -d "$CLEAN_ROOT/code2d/lmps/vumps/src"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_CPU_TARGET=generic
PRIVATE_DEPOT=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/clean-vumps-${SLURM_JOB_ID}.XXXXXX")
export JULIA_DEPOT_PATH="$PRIVATE_DEPOT:${JULIA_DEPOT_PATH:-/ix/zdai/kangw/PEPS3EE/.julia_depot_vumps_cpu:/ix/zdai/kangw/PEPS3EE/.julia_depot}"
export JULIA_PKG_PRECOMPILE_AUTO=0
JULIA_EXE="${JULIA_EXE:-/ihome/zdai/kaw593/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
test -x "$JULIA_EXE"
cd "$CLEAN_ROOT/code2d/lmps/vumps"
if [[ ${CLEAN_SMOKE:-0} == 1 ]]; then
    exec "$JULIA_EXE" --compiled-modules=no --pkgimages=no --compile=min -O0 --startup-file=no --project=. "$@"
fi
exec "$JULIA_EXE" --startup-file=no --project=. "$@"
