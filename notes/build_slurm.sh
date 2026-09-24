#!/bin/bash
#SBATCH --job-name=lmps-two-english-notes
#SBATCH --cluster=htc --partition=preempt --no-requeue
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:20:00
#SBATCH --output=/ix/zdai/kangw/PEPS3EE/PEPS/notes/build/job_logs/build_%j.out
#SBATCH --error=/ix/zdai/kangw/PEPS3EE/PEPS/notes/build/job_logs/build_%j.err
set -euo pipefail
LMOD_CMD="${LMOD_CMD:-/software/rhel9/spack/install/linux-rhel9-x86_64/gcc-11.4.1/lmod-8.7.24-7qktfjy26elph6anogbie43yasm4uo3u/lmod/lmod/libexec/lmod}"
eval "$("$LMOD_CMD" bash load texlive/2021)"
cd /ix/zdai/kangw/PEPS3EE/PEPS/notes
mkdir -p build previews
for note in detailed concise; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$note.tex"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$note.tex"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$note.tex"
  cp "build/$note.pdf" "$note.pdf"
  pdftotext -layout "$note.pdf" "build/$note.txt"
  pdfinfo "$note.pdf"
  pdftoppm -f 1 -l 1 -scale-to 1400 -png -singlefile "$note.pdf" "previews/$note"
done
pdftoppm -f 4 -l 8 -scale-to 1400 -png detailed.pdf previews/methods
if rg 'Undefined control sequence|LaTeX Error|undefined references|Citation .* undefined' build/detailed.log build/concise.log; then
  exit 2
fi
sha256sum detailed.tex concise.tex methods.tex detailed.pdf concise.pdf > build/SHA256SUMS
