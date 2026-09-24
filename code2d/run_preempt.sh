#!/bin/bash
#SBATCH --job-name=clean-mc-ctm
#SBATCH --cluster=smp
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
PRIVATE_DEPOT=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/clean-mcctm-${SLURM_JOB_ID}.XXXXXX")
export JULIA_DEPOT_PATH="$PRIVATE_DEPOT:${JULIA_DEPOT_PATH:-/ix/zdai/kangw/PEPS3EE/.julia_depot}"
export JULIA_PKG_PRECOMPILE_AUTO=0
JULIA_EXE="${JULIA_EXE:-/ihome/zdai/kaw593/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
cd "$CLEAN_ROOT/code2d"
if [[ ${1:-} == --python ]]; then
    shift
    exec python "$@"
fi
exec "$JULIA_EXE" --startup-file=no --project=. "$@"
