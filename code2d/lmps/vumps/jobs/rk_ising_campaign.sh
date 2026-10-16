#!/bin/bash
#SBATCH --job-name=rk-ising
#SBATCH --cluster=smp
#SBATCH --partition=preempt
#SBATCH --qos=smp-preempt-ll
#SBATCH --time=21-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch --partition=preempt}"
: "${CLEAN_ROOT:?Set the clean checkout root}"
: "${RK_CAMPAIGN_ROOT:?Set a separate campaign root}"
[[ ${SLURM_JOB_PARTITION:-} == preempt ]] || { echo 'preempt partition required' >&2; exit 2; }
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic}" JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/ix/zdai/kangw/PEPS3EE/.julia_depot_vumps_cpu:/ix/zdai/kangw/PEPS3EE/.julia_depot}"
JULIA_EXE="${JULIA_EXE:-/ihome/zdai/kaw593/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
mode="${1:-point}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ -z "${RK_PYTHON_EXE:-}" ]]; then
  module load anaconda3/2023.09-0-python_3.11.5
  RK_PYTHON_EXE=python3
fi
package="$RK_CAMPAIGN_ROOT/code_snapshot/code2d/lmps/vumps"
case "$mode" in
  init)
    exec "$RK_PYTHON_EXE" "$CLEAN_ROOT/code2d/lmps/vumps/production/rk_ising/rk_ising_campaign.py" \
      --root "$RK_CAMPAIGN_ROOT" init --clean "$CLEAN_ROOT" "$@"
    ;;
  verify|verify-code|collect)
    exec "$RK_PYTHON_EXE" "$package/production/rk_ising/rk_ising_campaign.py" \
      --root "$RK_CAMPAIGN_ROOT" "$mode" "$@"
    ;;
  point)
    [[ -f "$package/Project.toml" ]] || { echo 'Initialize campaign before submission' >&2; exit 2; }
    "$RK_PYTHON_EXE" "$package/production/rk_ising/rk_ising_campaign.py" \
      --root "$RK_CAMPAIGN_ROOT" verify-code
    cd "$package"
    exec "$JULIA_EXE" --startup-file=no --project=. -O3 production/rk_ising/rk_ising_point.jl "$@"
    ;;
  test)
    package="$CLEAN_ROOT/code2d/lmps/vumps"
    cd "$package"
    "$RK_PYTHON_EXE" test/test_rk_ising_campaign.py
    "$JULIA_EXE" --startup-file=no --project=. -O2 test/sixr_cap_selected.jl
    "$JULIA_EXE" --startup-file=no --project=. -O2 test/runtests.jl
    "$JULIA_EXE" --startup-file=no --project=. -O2 test/rk_ising_z2_smoke.jl
    "$JULIA_EXE" --startup-file=no --project=. -O2 test/rk_ising_mode_smoke.jl
    ;;
  *) echo 'Usage: rk_ising_campaign.sh {init|verify|verify-code|collect|point|test} [args]' >&2; exit 2 ;;
esac
