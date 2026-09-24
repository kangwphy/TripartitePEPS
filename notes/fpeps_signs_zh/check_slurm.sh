#!/bin/bash
#SBATCH --job-name=fpeps-sign-note-check
#SBATCH --cluster=htc --partition=preempt --no-requeue
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00
#SBATCH --output=/ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_signs_zh/check_%j.out
#SBATCH --error=/ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_signs_zh/check_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_signs_zh
python3 check_signs.py
