#!/bin/bash
#SBATCH --job-name=fpeps-sign-note-pdf
#SBATCH --cluster=htc --partition=preempt --no-requeue
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00
#SBATCH --output=/ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_signs_zh/build_%j.out
#SBATCH --error=/ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_signs_zh/build_%j.err
set -euo pipefail
LMOD_CMD="${LMOD_CMD:-/software/rhel9/spack/install/linux-rhel9-x86_64/gcc-11.4.1/lmod-8.7.24-7qktfjy26elph6anogbie43yasm4uo3u/lmod/lmod/libexec/lmod}"
eval "$("$LMOD_CMD" bash load texlive/2021)"
cd /ix/zdai/kangw/PEPS3EE/PEPS/notes/fpeps_signs_zh
mkdir -p build previews
for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
done
cp build/main.pdf main.pdf
pdftotext -layout main.pdf build/main.txt
pdfinfo main.pdf
pdftoppm -f 1 -l 1 -scale-to 1300 -singlefile -png main.pdf previews/first
for page in 5 8 9 10 11 12; do
  pdftoppm -f "$page" -l "$page" -scale-to 1400 -singlefile -png main.pdf "previews/page_$page"
done
if rg 'Undefined control sequence|LaTeX Error|undefined references|Citation .* undefined|Missing character' build/main.log; then
  exit 2
fi
for pass in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build independent_bivumps_note.tex
done
cp build/independent_bivumps_note.pdf independent_bivumps.pdf
pdftotext -layout independent_bivumps.pdf build/independent_bivumps.txt
pdftoppm -f 2 -l 2 -scale-to 1400 -singlefile -png independent_bivumps.pdf previews/independent_bivumps_page2
if rg 'Undefined control sequence|LaTeX Error|undefined references|Citation .* undefined|Missing character' build/independent_bivumps_note.log; then
  exit 2
fi
sha256sum main.tex two_sided_solver.tex two_sided_solver.md independent_bivumps.tex independent_bivumps_note.tex main.pdf independent_bivumps.pdf check_signs.py checks.json ../../code2d/fpeps/data/direct_chi_curve/independent_bivumps_stilde_chi.pdf > build/SHA256SUMS
