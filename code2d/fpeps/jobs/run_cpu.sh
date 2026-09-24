#!/bin/bash
# Compatibility name used by byte-preserved bare Python adapters.
#SBATCH --job-name=clean-fpeps
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${CLEAN_ROOT:?Set CLEAN_ROOT at submission}"
exec bash "$CLEAN_ROOT/code2d/fpeps/jobs/run_preempt.sh" "$@"
