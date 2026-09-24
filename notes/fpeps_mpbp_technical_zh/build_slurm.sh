#!/bin/bash
#SBATCH --job-name=fpeps-mpbp-note
#SBATCH --cluster=htc --partition=preempt --no-requeue
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:15:00
#SBATCH --output=/ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_mpbp_technical_zh/build_%j.out
#SBATCH --error=/ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_mpbp_technical_zh/build_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_PARTITION:-}" = preempt
LMOD_CMD="${LMOD_CMD:-/software/rhel9/spack/install/linux-rhel9-x86_64/gcc-11.4.1/lmod-8.7.24-7qktfjy26elph6anogbie43yasm4uo3u/lmod/lmod/libexec/lmod}"
eval "$("$LMOD_CMD" bash load texlive/2021)"
cd /ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_mpbp_technical_zh
mkdir -p build previews
# The rewritten tutorial uses vector TikZ diagrams. Historical numerical
# assets and their producer remain available; no plotting rerun is needed.
for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
done
if rg 'Undefined control sequence|LaTeX Error|undefined references|Citation .* undefined|Missing character' build/main.log; then
  exit 2
fi
cp build/main.pdf main.pdf
pdftotext -layout main.pdf build/main.txt
pdfinfo main.pdf
pdftoppm -r 72 -png main.pdf previews/page
# Keep an immutable, job-specific manifest and atomically replace the latest
# name. Rewriting the same inode left one NFS client reading the old manifest
# while a separate compute-node verification saw the current correct bytes.
manifest="build/SHA256SUMS_${SLURM_JOB_ID}"
sha256sum main.tex eigctm_details.tex build_slurm.sh main.pdf appendix_experiments_v18.tex appendix_experiments_v18.pdf > "$manifest.tmp"
mv "$manifest.tmp" "$manifest"
cp "$manifest" "build/SHA256SUMS.latest_${SLURM_JOB_ID}.tmp"
mv "build/SHA256SUMS.latest_${SLURM_JOB_ID}.tmp" build/SHA256SUMS
sha256sum -c "$manifest"
