#!/usr/bin/env python3
"""Exploded tensor-network picture of the four-replica CTMRG window contraction.

(A) AB quarter, (B) CA quarter, (C) the join on w_1, (D) the peak against CB.
Every quarter is drawn as its real 2x2 corner of the window (C, E*, T, script-A)
once per replica layer, the two layers welded by thick seam links at the E* and
script-A positions only.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch

OUT = ("/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/figures/"
       "beta_chi_scan_split/quarter_join_v3.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

COL  = {1: "#1f5fbf", 2: "#e0701a", 3: "#1a8f4c", 4: "#7b3fa0"}
TINT = {1: "#d9e6fa", 2: "#fbe1c9", 3: "#d0ecdc", 4: "#e9dbf6"}
INK, GREY, NEU, NEUT = "#1a1a1a", "#5a5a5a", "#4a4a4a", "#e8e8e8"

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})

GA = dict(dx=4.2, dy=3.4, sC=1.35, rT=0.62, wE=1.05, hE=0.72, sA=1.75,
          L=1.9, Ld=0.85, lwc=3.8, lwd=1.7, fst=16, fsl=14.5)
GC = dict(dx=3.4, dy=2.8, sC=1.15, rT=0.55, wE=0.90, hE=0.62, sA=1.50,
          L=1.6, Ld=0.60, lwc=3.2, lwd=1.5, fst=13, fsl=13)
GL = dict(GC, fsl=12.0)

# ------------------------------------------------------------------ strokes
def chi(ax, p0, p1, c, g, z=3, lw=None, halo=False):
    kw = dict(color=c, lw=lw or g["lwc"], solid_capstyle="round", zorder=z)
    if halo:
        kw["path_effects"] = [pe.withStroke(linewidth=(lw or g["lwc"]) + 5.0,
                                            foreground="white")]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **kw)

def dbond(ax, p0, p1, c, g, z=3):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=g["lwd"],
            ls=(0, (4.5, 2.2)), solid_capstyle="butt", zorder=z)

def seam(ax, p0, p1, ca, cb, lw=8.5, z=5):
    m = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=lw + 4.5,
            solid_capstyle="round", zorder=z - 0.2)
    ax.plot([p0[0], m[0]], [p0[1], m[1]], color=ca, lw=lw,
            solid_capstyle="butt", zorder=z, alpha=0.92)
    ax.plot([m[0], p1[0]], [m[1], p1[1]], color=cb, lw=lw,
            solid_capstyle="butt", zorder=z, alpha=0.92)

# ------------------------------------------------------------------- shapes
def sq(ax, x, y, s, c, t, lab, fs, z=6):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=t, ec=c, lw=2.5, zorder=z))
    ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1)

def ci(ax, x, y, r, c, t, lab, fs, z=6):
    ax.add_patch(Circle((x, y), r, fc=t, ec=c, lw=2.5, zorder=z))
    ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1)

def hexa(ax, x, y, w, h, c, t, lab, fs, z=6):
    pts = [(x - w, y), (x - 0.44 * w, y + h), (x + 0.44 * w, y + h),
           (x + w, y), (x + 0.44 * w, y - h), (x - 0.44 * w, y - h)]
    ax.add_patch(Polygon(pts, closed=True, fc=t, ec=c, lw=2.5, zorder=z))
    ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1)

# -------------------------------------------------- one replica layer (2x2)
def layer(ax, ox, oy, rep, g, lab_e, lab_t, sx=1, sy=1, dlab=False):
    """C top-left, E* top-right, T bottom-left, script-A bottom-right."""
    c, t = COL[rep], TINT[rep]
    dx, dy = g["dx"], g["dy"]
    sC, rT, wE, hE, sA = g["sC"], g["rT"], g["wE"], g["hE"], g["sA"]
    L, Ld = g["L"], g["Ld"]
    C = (ox, oy)
    E = (ox + sx * dx, oy)
    T = (ox, oy - sy * dy)
    A = (ox + sx * dx, oy - sy * dy)

    chi(ax, (C[0] + sx * sC / 2, C[1]), (E[0] - sx * wE, E[1]), c, g)
    chi(ax, (C[0], C[1] - sy * sC / 2), (T[0], T[1] + sy * rT), c, g)
    dbond(ax, (E[0], E[1] - sy * hE), (A[0], A[1] + sy * sA / 2), c, g)
    dbond(ax, (T[0] + sx * rT, T[1]), (A[0] - sx * sA / 2, A[1]), c, g)

    e1 = (E[0] + sx * (wE + L), E[1])
    t1 = (T[0], T[1] - sy * (rT + L))
    chi(ax, (E[0] + sx * wE, E[1]), e1, c, g)
    chi(ax, (T[0], T[1] - sy * rT), t1, c, g)
    ad1 = (A[0] + sx * (sA / 2 + Ld), A[1])
    ad2 = (A[0], A[1] - sy * (sA / 2 + Ld))
    dbond(ax, (A[0] + sx * sA / 2, A[1]), ad1, c, g)
    dbond(ax, (A[0], A[1] - sy * sA / 2), ad2, c, g)

    sq(ax, C[0], C[1], sC, c, t, r"$C$", g["fst"])
    ci(ax, T[0], T[1], rT, c, t, r"$T$", g["fst"])
    hexa(ax, E[0], E[1], wE, hE, c, t, r"$E^{*}$", g["fst"] - 1)
    sq(ax, A[0], A[1], sA, c, t, r"$\mathcal{A}$", g["fst"] + 3)

    if lab_e:
        ax.text(e1[0], e1[1] + 0.52, lab_e, ha="center", va="bottom",
                fontsize=g["fsl"], color=c, fontweight="bold", zorder=10)
    if lab_t:
        ax.text(t1[0], t1[1] - sy * 0.5, lab_t, ha="center",
                va="top" if sy > 0 else "bottom",
                fontsize=g["fsl"], color=c, fontweight="bold", zorder=10)
    if dlab:
        ax.text(ad1[0] + sx * 0.3, ad1[1], r"$D$", ha="left" if sx > 0 else "right",
                va="center", fontsize=12, color=GREY, zorder=10)
        ax.text(ad2[0], ad2[1] - sy * 0.3, r"$D$", ha="center",
                va="top" if sy > 0 else "bottom", fontsize=12, color=GREY, zorder=10)
    return dict(C=C, E=E, T=T, A=A, e=e1, t=t1)

def quarter(ax, b1, b2, r1, r2, g, labs1, labs2, sx=1, sy=1, dlab=False):
    """Two stacked replica layers welded by seam links at E* and script-A."""
    P1 = layer(ax, b1[0], b1[1], r1, g, labs1[0], labs1[1], sx=sx, sy=sy, dlab=dlab)
    P2 = layer(ax, b2[0], b2[1], r2, g, labs2[0], labs2[1], sx=sx, sy=sy)
    seam(ax, P1["E"], P2["E"], COL[r1], COL[r2])
    seam(ax, P1["A"], P2["A"], COL[r1], COL[r2])
    return P1, P2

def slab(ax, x, y, w, h, rep, txt, fs=13, z=6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.30",
                                fc=TINT[rep], ec=COL[rep], lw=2.6, zorder=z))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            color=COL[rep], fontweight="bold", zorder=z + 1)

def slab_leg(ax, x0, y0, x1, y1, rep, lab, g, side="left"):
    chi(ax, (x0, y0), (x1, y1), COL[rep], g)
    if lab:
        dx = -0.35 if side == "left" else 0.35
        ax.text(x1 + dx, y1, lab, ha="right" if side == "left" else "left",
                va="center", fontsize=g["fsl"], color=COL[rep],
                fontweight="bold", zorder=10)

def title(ax, xy, txt, sub=None):
    ax.text(xy[0], xy[1], txt, ha="left", va="top", fontsize=17.5,
            fontweight="bold", color=INK, zorder=12)
    if sub:
        ax.text(xy[0], xy[1] - 1.35, sub, ha="left", va="top", fontsize=13.5,
                color=GREY, zorder=12)

# ===========================================================  figure canvas
fig = plt.figure(figsize=(19.5, 17.8))
gs = GridSpec(3, 2, height_ratios=[0.30, 1.0, 1.0], hspace=0.05, wspace=0.04,
              left=0.014, right=0.986, top=0.952, bottom=0.018)

def panel(cell, xl, yl):
    ax = fig.add_subplot(cell)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal"); ax.set_axis_off()
    return ax

# ------------------------------------------------------------------ legend
kx = panel(gs[0, :], (0, 94.5), (0, 10))
GK = dict(dx=0, dy=0, sC=0, rT=0, wE=0, hE=0, sA=0, L=0, Ld=0,
          lwc=3.8, lwd=1.7, fst=13, fsl=13)
y1, y2 = 6.9, 2.3
sq(kx, 2.6, y1, 2.0, NEU, NEUT, "", 0)
kx.text(4.6, y1, r"$C$   corner, $\chi_1\!\times\!\chi_1$", va="center", fontsize=13.5)
ci(kx, 15.0, y1, 1.0, NEU, NEUT, "", 0)
kx.text(16.6, y1, r"$T$   edge, $\chi_1\!\times\!\chi_1\!\times\!D$", va="center", fontsize=13.5)
hexa(kx, 29.0, y1, 1.5, 1.0, NEU, NEUT, "", 0)
kx.text(31.0, y1, r"$E^{*}$   seam edge $-$ one tensor on 2 replicas",
        va="center", fontsize=13.5)
sq(kx, 55.0, y1, 2.4, NEU, NEUT, "", 0)
kx.text(57.0, y1, r"$\mathcal{A}$   dressed region $-$ one tensor on 2 replicas",
        va="center", fontsize=13.5)
seam(kx, (86.0, y1 + 1.3), (86.0, y1 - 1.3), COL[1], COL[2], lw=7.5)
kx.text(87.8, y1, "seam link", va="center", fontsize=13.5)

chi(kx, (2.0, y2), (5.4, y2), NEU, GK)
kx.text(6.0, y2, r"$\chi_1$ bond", va="center", fontsize=13.5)
dbond(kx, (13.5, y2), (16.9, y2), NEU, GK)
kx.text(17.5, y2, r"$D$ bond  ($D=2$)", va="center", fontsize=13.5)
kx.text(28.5, y2, "replicas:", va="center", fontsize=13.5)
for i, r in enumerate((1, 2, 3, 4)):
    x = 34.6 + 3.4 * i
    kx.add_patch(Rectangle((x - 0.85, y2 - 0.85), 1.7, 1.7, fc=COL[r],
                           ec=COL[r], lw=1.5, zorder=6))
    kx.text(x + 1.25, y2, str(r), va="center", fontsize=13, color=COL[r],
            fontweight="bold")
kx.text(50.5, y2, "seams:   AB = (1,2) (3,4)     CA = (1,3) (2,4)     CB = (1,4) (2,3)",
        va="center", fontsize=13.5, color=INK)

# =========================================================== (A)  AB quarter
ax = panel(gs[1, 0], (-2.2, 19.8), (-14.4, 3.6))
title(ax, (-2.0, 3.5), "(A)  AB quarter  $-$  north seam",
      "replicas 1 + 2 welded at $E^{*}$ and $\\mathcal{A}$")
P1, P2 = quarter(ax, (0.0, 0.0), (4.0, -6.6), 1, 2, GA,
                 (r"$n_1$", None), (r"$e_2$", r"$w_2$"))
ax.text(7.5, -2.4, "seam", fontsize=12.5, color=GREY, style="italic", zorder=11)
# w_1 : the leg that will be shared with (B)
ax.text(P1["t"][0], P1["t"][1] - 0.5, r"$w_1$", ha="center", va="top",
        fontsize=GA["fsl"], color=COL[1], fontweight="bold", zorder=11)
ax.plot([P1["t"][0]], [P1["t"][1]], "o", ms=12, mfc="white", mec=COL[1],
        mew=3.0, zorder=11)
ax.text(P1["t"][0] + 0.62, P1["t"][1], "joins (B)", ha="left", va="center",
        fontsize=12, color=COL[1], zorder=11)

AX, AY = 12.9, -0.4
ax.text(AX, AY, r"stores   $\chi_1^{4}D^{4}$", fontsize=19, color=INK, va="top")
ax.text(AX, AY - 2.0, r"cost   $\chi_1^{6}D^{4}$", fontsize=15, color=GREY, va="top")
ax.text(AX, AY - 4.3, r"4 dangling $\chi_1$ legs:", fontsize=13.5, va="top")
ax.text(AX, AY - 5.7, r"$n_1,\ w_1$   on replica 1", fontsize=14.5,
        color=COL[1], va="top", fontweight="bold")
ax.text(AX, AY - 7.0, r"$e_2,\ w_2$   on replica 2", fontsize=14.5,
        color=COL[2], va="top", fontweight="bold")
ax.text(AX, AY - 9.0, r"+ 4 dangling $D$ legs", fontsize=13.5, color=GREY, va="top")

# =========================================================== (B)  CA quarter
ax = panel(gs[1, 1], (-2.2, 19.8), (-14.4, 3.6))
title(ax, (-2.0, 3.5), "(B)  CA quarter  $-$  west seam",
      "replicas 1 + 3 welded at $E^{*}$ and $\\mathcal{A}$")
P1, P3 = quarter(ax, (0.0, 0.0), (4.0, -6.6), 1, 3, GA,
                 (None, r"$s_1$"), (r"$w_3$", r"$s_3$"))
ax.text(7.5, -2.4, "seam", fontsize=12.5, color=GREY, style="italic", zorder=11)
ax.text(P1["e"][0], P1["e"][1] + 0.52, r"$w_1$", ha="center", va="bottom",
        fontsize=GA["fsl"], color=COL[1], fontweight="bold", zorder=11)
ax.plot([P1["e"][0]], [P1["e"][1]], "o", ms=12, mfc="white", mec=COL[1],
        mew=3.0, zorder=11)
ax.text(P1["e"][0], P1["e"][1] - 0.75, "joins (A)", ha="center", va="top",
        fontsize=12, color=COL[1], zorder=11)

ax.text(AX, AY, r"stores   $\chi_1^{4}D^{4}$", fontsize=19, color=INK, va="top")
ax.text(AX, AY - 2.0, r"cost   $\chi_1^{6}D^{4}$", fontsize=15, color=GREY, va="top")
ax.text(AX, AY - 4.3, r"4 dangling $\chi_1$ legs:", fontsize=13.5, va="top")
ax.text(AX, AY - 5.7, r"$w_1,\ s_1$   on replica 1", fontsize=14.5,
        color=COL[1], va="top", fontweight="bold")
ax.text(AX, AY - 7.0, r"$w_3,\ s_3$   on replica 3", fontsize=14.5,
        color=COL[3], va="top", fontweight="bold")
ax.text(AX, AY - 9.0, r"+ 4 dangling $D$ legs", fontsize=13.5, color=GREY, va="top")

# ================================================================ (C)  join
ax = panel(gs[2, 0], (-1.4, 23.8), (-11.6, 11.6))
title(ax, (-1.2, 11.5),
      r"(C)  Join:  AB quarter $\bowtie$ CA quarter  on the one leg $w_1$")
CA1, CA3 = quarter(ax, (0.0, 0.0), (3.3, -5.4), 1, 3, GC,
                   (None, r"$s_1$"), (r"$w_3$", r"$s_3$"))
AB1, AB2 = quarter(ax, (13.0, 9.2), (16.3, 3.8), 1, 2, GC,
                   (r"$n_1$", None), (r"$e_2$", r"$w_2$"))
# the single shared bond
jx = [AB1["t"][0], AB1["t"][0], CA1["e"][0]]
jy = [AB1["t"][1], 0.0, 0.0]
ax.plot(jx, jy, color=COL[1], lw=GC["lwc"] + 2.4, solid_capstyle="round",
        solid_joinstyle="round", zorder=9,
        path_effects=[pe.withStroke(linewidth=GC["lwc"] + 8.0, foreground="white")])
for p in ((AB1["t"][0], AB1["t"][1]), (CA1["e"][0], CA1["e"][1])):
    ax.plot([p[0]], [p[1]], "o", ms=10, mfc="white", mec=COL[1], mew=2.8, zorder=10)
ax.text(9.6, 0.55, r"$w_1$", ha="center", va="bottom", fontsize=17,
        color=COL[1], fontweight="bold", zorder=11)
ax.text(9.6, -0.55, "the only shared leg", ha="center", va="top", fontsize=13,
        color=COL[1], zorder=11)

ax.text(0.3, 8.8, r"$\chi_1^{6}D^{6}$ = 91 GiB at $\chi_1=24$", fontsize=18,
        color=INK, va="center")
ax.text(0.3, 7.0, r"cost $\chi_1^{7}$  $-$  1% of the run", fontsize=15,
        color=GREY, va="center")
ax.text(0.3, 5.0, "6 dangling legs remain:", fontsize=13.5, va="center")
for dy_, txt, r in ((3.9, r"$n_1,\ s_1$   replica 1", 1),
                    (2.8, r"$e_2,\ w_2$   replica 2", 2),
                    (1.7, r"$w_3,\ s_3$   replica 3", 3)):
    ax.text(0.3, dy_, txt, fontsize=14, color=COL[r], va="center",
            fontweight="bold")

# result glyph: one tensor, three replica layers, six legs
ax.add_patch(FancyArrowPatch((11.4, -3.6), (14.9, -5.9), arrowstyle="-|>",
                             mutation_scale=26, lw=2.6, color="#333333",
                             connectionstyle="arc3,rad=-0.20", zorder=11))
for yy, r, tx in ((-4.2, 2, "replica 2"), (-6.0, 1, "replica 1"), (-7.8, 3, "replica 3")):
    slab(ax, 18.5, yy, 4.4, 1.4, r, tx, fs=12)
seam(ax, (17.2, -4.9), (17.2, -5.3), COL[2], COL[1], lw=7.0)
seam(ax, (19.8, -6.7), (19.8, -7.1), COL[1], COL[3], lw=7.0)
ax.text(16.2, -5.1, "AB", ha="right", va="center", fontsize=11, color=GREY)
ax.text(20.9, -6.9, "CA", ha="left", va="center", fontsize=11, color=GREY)
for yy, r, lab in ((-3.75, 2, r"$e_2$"), (-4.65, 2, r"$w_2$"),
                   (-7.35, 3, r"$w_3$"), (-8.25, 3, r"$s_3$")):
    slab_leg(ax, 16.3, yy, 14.8, yy, r, lab, GL, side="left")
for yy, lab in ((-5.55, r"$n_1$"), (-6.45, r"$s_1$")):
    slab_leg(ax, 20.7, yy, 22.2, yy, 1, lab, GL, side="right")
ax.text(18.5, -2.5, "one tensor, six legs", ha="center", va="center",
        fontsize=13.5, color=INK)

# ================================================================ (D)  peak
ax = panel(gs[2, 1], (-6.0, 21.0), (-13.4, 8.6))
title(ax, (-5.8, 8.5),
      r"(D)  Peak: the six-leg object meets the CB quarter $-$ they share two legs")

# the six-leg object built in (C)
for yy, r, tx in ((4.4, 2, "replica 2"), (2.0, 1, "replica 1"), (-0.4, 3, "replica 3")):
    slab(ax, 0.0, yy, 5.0, 1.6, r, tx, fs=12.5)
seam(ax, (-1.5, 3.6), (-1.5, 2.8), COL[2], COL[1], lw=7.5)
seam(ax, (1.5, 1.2), (1.5, 0.4), COL[1], COL[3], lw=7.5)
ax.text(-2.6, 3.2, "AB", ha="right", va="center", fontsize=11.5, color=GREY)
ax.text(2.6, 0.8, "CA", ha="left", va="center", fontsize=11.5, color=GREY)
for yy, r, lab in ((4.90, 2, r"$e_2$"), (3.90, 2, r"$w_2$"),
                   (0.10, 3, r"$w_3$"), (-0.90, 3, r"$s_3$")):
    slab_leg(ax, -2.5, yy, -4.1, yy, r, lab, GL, side="left")
slab_leg(ax, 2.5, 1.65, 4.1, 1.65, 1, None, GC)
slab_leg(ax, 2.5, 2.35, 4.1, 2.35, 1, None, GC)
ax.text(0.0, 5.9, r"from (C):  $\chi_1^{6}D^{6}$", ha="center", va="center",
        fontsize=13.5, color=INK)

# CB quarter: layers 1 and 4, drawn mirrored so its replica-1 legs face left
CB1, CB4 = quarter(ax, (14.5, 1.65), (19.5, -5.0), 1, 4, GC,
                   (None, None), (r"$e_4$", r"$s_4$"), sx=-1, sy=-1)
for pts in (([4.1, 8.6], [1.65, 1.65]),
            ([14.5, 4.1, 4.1], [6.6, 6.6, 2.35])):
    ax.plot(pts[0], pts[1], color=COL[1], lw=GC["lwc"] + 2.4, zorder=9,
            solid_capstyle="round", solid_joinstyle="round",
            path_effects=[pe.withStroke(linewidth=GC["lwc"] + 8.0, foreground="white")])
ax.text(6.6, 1.15, r"$n_1$", ha="center", va="top", fontsize=16,
        color=COL[1], fontweight="bold", zorder=11)
ax.text(9.5, 6.95, r"$s_1$", ha="center", va="bottom", fontsize=16,
        color=COL[1], fontweight="bold", zorder=11)
ax.text(6.8, 4.35, "2 shared legs", ha="center", va="center", fontsize=13,
        color=COL[1], zorder=11)

# result: replica 1 is gone
ax.add_patch(FancyArrowPatch((8.0, -2.6), (2.2, -4.4), arrowstyle="-|>",
                             mutation_scale=26, lw=2.6, color="#333333",
                             connectionstyle="arc3,rad=-0.22", zorder=11))
for yy, r, tx in ((-4.6, 2, "replica 2"), (-6.2, 3, "replica 3"), (-7.8, 4, "replica 4")):
    slab(ax, -1.0, yy, 4.0, 1.2, r, tx, fs=11.5)
seam(ax, (-2.4, -5.2), (-2.4, -5.6), COL[2], COL[3], lw=6.5)
seam(ax, (0.4, -6.8), (0.4, -7.2), COL[3], COL[4], lw=6.5)
for yy, r, lab in ((-4.28, 2, r"$e_2$"), (-4.92, 2, r"$w_2$"),
                   (-5.88, 3, r"$w_3$"), (-6.52, 3, r"$s_3$")):
    slab_leg(ax, -3.0, yy, -4.4, yy, r, lab, GL, side="left")
for yy, lab in ((-7.48, r"$e_4$"), (-8.12, r"$s_4$")):
    slab_leg(ax, 1.0, yy, 2.4, yy, 4, lab, GL, side="right")
ax.text(-1.0, -3.3, r"replicas 2, 3, 4 only  $-$  $\chi_1^{6}D^{6}$", ha="center",
        va="center", fontsize=13.5, color=INK)

ax.text(-5.8, -9.9, r"cost $=\ \chi_1^{6}\times\chi_1^{2}=\chi_1^{8}$"
                    r"    $-$    48% of the entire calculation",
        fontsize=16, color=INK, va="center")
ax.text(-5.8, -11.3, "replica 1 is eliminated here", fontsize=15,
        color=COL[1], va="center", fontweight="bold")
ax.text(-5.8, -12.7, r"6 legs $=$ 2 legs $\times$ 3 seams  $\Rightarrow$"
                     r"  the memory cannot be lower",
        fontsize=15, color=INK, va="center")

fig.suptitle("Four-replica CTMRG window: two-replica quarters, one join, and the peak",
             fontsize=21, fontweight="bold", y=0.988)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
