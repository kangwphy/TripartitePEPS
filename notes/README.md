# Current English methods notes

Only two documents are maintained as current methods notes:

- [Detailed methods](detailed.pdf): tensor diagrams, index conventions, three distinct LMPS constructions, six-R wiring, numerical checks, limitations and references.
- [Concise guide](concise.pdf): the working method and the most important qualifications.

The source is `detailed.tex` + `methods.tex`, and `concise.tex`. Build with
`sbatch build_slurm.sh`; compilation, PDF checks and previews run on Slurm.
`build/` and `previews/` are generated products, not additional notes.

Important correction: TFIM GS optimization currently uses AD/CTMRG. VUMPS
solves the saved PEPS boundary. The historical production `direct` label
refers to **grown-tail** contraction, not to separate bare `LR` and
three-column `LTR` fixed points. The new cardinal bare diagnostic implements
the latter. Do not infer the method from an old CSV label alone.

The detailed note's 17 September revision adds explicit canonical-chain and
self-overlap diagrams (Section 3), literal bare recursion and center-placement
diagrams (Section 4), and the complete A/B bent boundaries with their individual
rail/index definitions (Section 5). It distinguishes spatial `M_W/M_E` from
the canonical `A_L/A_R` of one solve. Grown-tail and polished endmap use the
same mixed-map iteration under the stated matching conditions; their agreement
does not prove equivalence to the original bare network. Previous source and
PDF are backed up in `archive/20260917_gauge_geometry_before/`.

Section 4.1 now also spells out the actual exports
`M_W[p] = transpose(A_L,W[p])`, `M_E[p] = A_L,E[p]`: the bare driver
does not explicitly insert the saved center matrices. A side-by-side diagram
shows the different cut-centered convention, including `C_W^T` and `C_E`,
and finite-chain identities show where the centers move. Equivalence after
changing caps or open-cut coordinates is not assumed. The preceding version
is saved in `archive/20260917_explicit_centers_before/`.

Historical documents, including abandoned derivations, are preserved under
`archive/`. Archiving is not scientific endorsement of their claims. Do not
merge their formulas into the current method without checking the network,
normalization and metric definitions. Recently edited seam-phase documents
are protected from movement until their editing status is confirmed.

Data access: [organized results](../results/README.md). Run-specific README
files are provenance records, not additional current methods manuscripts.

Separate RK-Ising scaling discussion requested on 18 September:
[Chinese interpretation note](rk_ising_scaling_zh.pdf)
([source](rk_ising_scaling_zh.tex)). This reproduces the three displayed slope
fits, audits window/chi dependence, and distinguishes finite-scale crossover
from an asymptotic universal logarithmic coefficient. It corrects stronger
claims in archived seam-phase notes; it does not replace the two methods
manuscripts. Reproducible data and figures:
[scaling analysis](../results/RK_Ising/scaling_interpretation/README.md).

Separate research proposal: [strict fermionic PEPS and six-R signs (Chinese)](fpeps_signs_zh/main.pdf).
This develops the local tensor, ordering conventions and validation plan.
The separate [implementation supplement](fpeps_signs_zh/implementation.md) and
[fermionic backend](../code2d/fpeps/README.md) now cover finite-chi signed LMPS
measurements, physical sewing checks and seam-length convergence. They do not
replace the two RK/TFIM methods notes or establish an infinite-chi extrapolation.
