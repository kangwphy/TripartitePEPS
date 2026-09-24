#!/bin/bash
#SBATCH --job-name=clean-fpeps
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
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_CPU_TARGET=generic
PRIVATE_DEPOT=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/clean-fpeps-${SLURM_JOB_ID}.XXXXXX")
export JULIA_DEPOT_PATH="$PRIVATE_DEPOT:${JULIA_DEPOT_PATH:-/ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps/.julia_depot:/ix/zdai/kangw/PEPS3EE/.julia_depot:/ix/zdai/kangw/PEPS3EE/.julia_depot_vumps_cpu}"
export JULIA_PKG_PRECOMPILE_AUTO=0
JULIA_EXE="${JULIA_EXE:-/ihome/zdai/kaw593/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
cd "$CLEAN_ROOT/code2d/fpeps"
if [[ ${CLEAN_SMOKE:-0} == 1 ]]; then
    exec "$JULIA_EXE" --compiled-modules=no --pkgimages=no --compile=min -O0 --startup-file=no --project=environments/default "$@"
fi
exec "$JULIA_EXE" --startup-file=no --project=environments/default "$@"
