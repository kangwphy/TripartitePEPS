#!/usr/bin/env python3
# plot_ipeps_chi.py -- iPEPS junction numbers vs chi, against the external
# finite-L exact-route benchmark (the refutation, made visible).
# Data provenance (hardcoded from job logs):
#   iPEPS tS(chi), gapped beta=0.30 : yvertex13, job 23690063
#   iPEPS -ln(gamma_A)(chi)         : yvertex12, job 23689859
#   truth: finite-L exact route     : jbench_driver, job 23692675
#          S_T(L): 0.010327/0.010755/0.010894/0.011039 (L=6/8/10/12, chi384)
#          corner_A(L): -0.0031689/-0.0033040/-0.0033450/-0.0033572 -> -0.00336(2)
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/ix/zdai/kangw/PEPS3EE/PEPS/clean/figs2d"; os.makedirs(FIG, exist_ok=True)

# (chi, keff, tS, corner-gate score, good-arm?)
IPEPS_TS = [
    (2, 2, 1.0926e-4, 3.5e-2, True),
    (3, 3, 6.372e-2,  9.2e-2, False),   # bad arm: resL=0.27 >> floor
    (4, 4, 1.1256e-4, 4.0e-2, True),
    (5, 5, 7.257e-5,  9.6e-2, False),   # bad arm: resL=0.44
    (6, 6, 1.0532e-4, 3.0e-3, True),
    (8, 6, 1.0532e-4, 3.0e-3, True),    # keff trims back to 6
    (10, 6, 1.0532e-4, 3.0e-3, True),
    (12, 6, 1.0532e-4, 3.0e-3, True),
]
TRUTH_TS = (0.0110, 0.0104, 0.0118)     # central, band lo, band hi (L->inf est.)

IPEPS_GA = [(4, 1.1655e-4), (6, 1.3488e-4)]      # |-ln gamma_A|
TRUTH_GA = (3.357e-3, 3.30e-3, 3.42e-3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=160)

# ---- (a) tS vs chi ----
ax1.axhspan(TRUTH_TS[1], TRUTH_TS[2], color="#BBBBBB", alpha=0.5, zorder=1)
ax1.axhline(TRUTH_TS[0], color="#555555", lw=1.2, ls="--", zorder=2)
ax1.annotate("truth: finite-$L$ exact route\n$\\tilde S_T(L\\to\\infty)\\approx0.011$",
             (2.1, 0.0122), fontsize=9, color="#444444")
good = [(c, v) for c, k, v, s, g in IPEPS_TS if g]
bad  = [(c, v) for c, k, v, s, g in IPEPS_TS if not g]
ax1.plot([c for c, v in good], [v for c, v in good], "o-", color="#2E5FA3",
         ms=7, lw=1.2, label="iPEPS (good arm)", zorder=4)
ax1.plot([c for c, v in bad], [v for c, v in bad], "x", color="#C4552D",
         ms=9, mew=2, label="iPEPS (bad arm, resL$\\gg$floor)", zorder=4)
ax1.annotate("iPEPS plateau $1.05\\times10^{-4}$\n(flat for ALL reachable $\\chi$,\n"
             "$\\chi_{\\rm eff}$ caps at 6)", (6.2, 6e-5), fontsize=9, color="#2E5FA3")
ax1.annotate("$\\times$100 gap", (2.6, 1.4e-3), fontsize=11, color="#8C2F12")
ax1.annotate("", xy=(2.4, 0.0104), xytext=(2.4, 1.3e-4),
             arrowprops=dict(arrowstyle="<->", color="#8C2F12", lw=1.4))
ax1.set_yscale("log")
ax1.set_xlabel(r"$\chi$ (requested)"); ax1.set_ylabel(r"$\tilde S$  (log)")
ax1.set_title(r"(a) gapped $\beta=0.30$: iPEPS $\tilde S(\chi)$ vs the benchmark", fontsize=10)
ax1.set_xticks([2, 3, 4, 5, 6, 8, 10, 12])
ax1.grid(alpha=0.22, lw=0.5); ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(fontsize=8, loc="center right", framealpha=0.9)

# ---- (b) corner constant vs chi ----
ax2.axhspan(TRUTH_GA[1], TRUTH_GA[2], color="#BBBBBB", alpha=0.5, zorder=1)
ax2.axhline(TRUTH_GA[0], color="#555555", lw=1.2, ls="--", zorder=2)
ax2.annotate("truth: $|{-}\\ln\\gamma_A| = 3.36(2)\\times10^{-3}$\n(finite-$L$ quadrant$-$line)",
             (4.1, 2.4e-3), fontsize=9, color="#444444")
ax2.plot([c for c, v in IPEPS_GA], [v for c, v in IPEPS_GA], "s-", color="#2E5FA3",
         ms=7, lw=1.2, label="iPEPS $|{-}\\ln\\gamma_A|$", zorder=4)
ax2.annotate("iPEPS: $1.2$–$1.3\\times10^{-4}$\n($\\times$25 too small)",
             (4.6, 8e-5), fontsize=9, color="#2E5FA3")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\chi$"); ax2.set_ylabel(r"$|{-}\ln\gamma_A|$  (log)")
ax2.set_title(r"(b) the 2-replica corner constant: same story at 25$\times$", fontsize=10)
ax2.set_xticks([4, 6]); ax2.set_xlim(3.4, 8.2)
ax2.grid(alpha=0.22, lw=0.5); ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)

fig.suptitle("iPEPS junction constants vs $\\chi$: internally stable, externally refuted "
             "(jobs 23690063 / 23689859 vs jbench 23692675)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, "ipeps_vs_chi_refutation.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print("DONE_IPEPSCHI")
