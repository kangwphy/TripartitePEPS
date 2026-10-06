#!/bin/bash
# Invoked only inside a real allocation by either host-specific launcher.
set -euo pipefail
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${CHAIN_PACKAGE:?Set package path}"
: "${CHAIN_CAMPAIGN_ROOT:?Set new per-D campaign root}"
: "${JULIA_EXE:?Set Julia executable}"
DIRECTION=${1:?Pass increasing_h or decreasing_h}
case "$DIRECTION" in increasing_h|decreasing_h) ;; *) exit 2 ;; esac
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export JULIA_PKG_PRECOMPILE_AUTO=0
export TFIM_BASE_GIT_COMMIT=${TFIM_BASE_GIT_COMMIT:-1e7cfd36531b670db8d3cc3b975cc417963e3d50}
export TFIM_CODE_REVISION=${TFIM_CODE_REVISION:?Export the actual committed driver revision}
PLAN=${CHAIN_PLAN:-$CHAIN_CAMPAIGN_ROOT/campaign_plan.json}
SEED=${CHAIN_SEED:-$CHAIN_CAMPAIGN_ROOT/seeds/$DIRECTION.jls}
cd "$CHAIN_PACKAGE"
python3 "$CHAIN_PACKAGE/production/tfim/run_bidirectional_chain.py" \
  --package "$CHAIN_PACKAGE" --plan "$PLAN" --root "$CHAIN_CAMPAIGN_ROOT" \
  --direction "$DIRECTION" --seed "$SEED" --julia "$JULIA_EXE" \
  --wall-seconds "${CHAIN_WALL_SECONDS:-85800}" \
  --pilot-steps "${CHAIN_PILOT_STEPS:-2}" --pilot-segment 1 &
controller_pid=$!
trap 'kill -USR1 "$controller_pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$controller_pid" 2>/dev/null || true' TERM
trap 'kill -INT "$controller_pid" 2>/dev/null || true' INT
set +e
while true; do
  wait "$controller_pid"
  result=$?
  kill -0 "$controller_pid" 2>/dev/null || break
done
set -e
if [[ "$result" -eq 42 && "${CHAIN_AUTO_REQUEUE:-1}" == 1 ]]; then
  # Intentional wall-limit continuation only. TERM/cancellation is not requeued here.
  if [[ -n "${CHAIN_SLURM_CLUSTER:-}" ]]; then
    scontrol -M "$CHAIN_SLURM_CLUSTER" requeue "$SLURM_JOB_ID"
  else
    scontrol requeue "$SLURM_JOB_ID"
  fi
  exit 0
fi
exit "$result"
