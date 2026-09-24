#!/usr/bin/env python3
"""
Tensor-network diagram (railroad / double-line style) of the four-replica
CTMRG window contraction:  AB quarter + CA quarter -> join on w_1 -> peak
against the CB quarter.

Every replica owns a coloured rail.  One-replica tensors (C, T) sit on a
single rail; the seam objects (E*, script-A) are two-tone shapes that two
rails run through.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch

OUT = ("/ix/zdai/kangw/PEPS3EE/PEPS/clean/data/rk_ising/ctmrg/figures/"
       "beta_chi_scan_split/quarter_join_v2.png")

# ----------------------------------------------------------------- palette
COL  = {1: "#1a4f9c", 2: "#cf4f16", 3: "#1f7a3f", 4: "#7b3fa0"}
TINT = {1: "#d3e2f6", 2: "#fbdcc9", 3: "#cfe9d7", 4: "#e7d6f3"}
INK  = "#191919"
GREY = "#585858"

LW_CHI, LW_D = 4.4, 2.0
FS_LEG, FS_TEN, FS_TITLE, FS_NOTE, FS_KEY = 15, 16, 18, 14, 13

plt.rcParams.update({"font.size": 14, "mathtext.fontset": "dejavusans"})

# ----------------------------------------------------------------- helpers
def rail(ax, pts, c, lw=LW_CHI, z=2, ls="-"):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=lw, ls=ls,
            solid_capstyle="round", solid_joinstyle="round", zorder=z)

def dbond(ax, pts, c, z=2):
    rail(ax, pts, c, lw=LW_D, z=z)

def tensor_sq(ax, x, y, c, tint, lab, s=1.06, z=5, fs=FS_TEN):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=tint, ec=c,
                           lw=2.6, zorder=z))
    ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK,
            zorder=z + 2)

def tensor_ci(ax, x, y, c, tint, lab, r=0.56, z=5, fs=FS_TEN):
    ax.add_patch(Circle((x, y), r, fc=tint, ec=c, lw=2.6, zorder=z))
    ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK,
            zorder=z + 2)

def hex_seam(ax, x, y, tA, tB, lab, hh=1.75, ww=0.85, z=5, fs=FS_TEN):
    """Two-replica seam edge tensor E*: elongated hexagon, two-tone."""
    sh = hh * 0.486
    top = [(x, y + hh), (x + ww, y + sh), (x + ww, y), (x - ww, y), (x - ww, y + sh)]
    bot = [(x, y - hh), (x + ww, y - sh), (x + ww, y), (x - ww, y), (x - ww, y - sh)]
    full = [(x, y + hh), (x + ww, y + sh), (x + ww, y - sh), (x, y - hh),
            (x - ww, y - sh), (x - ww, y + sh)]
    ax.add_patch(Polygon(top, closed=True, fc=tA, ec="none", zorder=z))
    ax.add_patch(Polygon(bot, closed=True, fc=tB, ec="none", zorder=z))
    ax.add_patch(Polygon(full, closed=True, fc="none", ec=INK, lw=2.6, zorder=z + 1))
    ax.plot([x - ww, x + ww], [y, y], color=INK, lw=2.0, ls=(0, (3, 2)), zorder=z + 1)
    ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 2,
            bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2))

def sq_seam(ax, x, y, tA, tB, lab, s=2.70, z=5, fs=FS_TEN + 2):
    """Two-replica dressed region tensor script-A: big two-tone square."""
    ax.add_patch(Rectangle((x - s / 2, y), s, s / 2, fc=tA, ec="none", zorder=z))
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s / 2, fc=tB, ec="none", zorder=z))
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc="none", ec=INK,
                           lw=2.6, zorder=z + 1))
    ax.plot([x - s / 2, x + s / 2], [y, y], color=INK, lw=2.0, ls=(0, (3, 2)),
            zorder=z + 1)
    ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 2,
            bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2))

# ------------------------------------------------------------ one quarter
YT, XC, XT = 3.3, 0.0, 3.2
HH, WW, SA, RT = 1.75, 0.85, 2.70, 0.56

def draw_quarter(ax, cx, cy, rA, rB, legs, mirror=False):
    """Draw a seam quarter (C, T on each rail + shared E*, script-A).

    Returns the anchor points of the four chi1 boundary legs."""
    m = -1.0 if mirror else 1.0
    X = lambda lx: cx + m * lx
    Y = lambda ly: cy + ly
    cA, tA = COL[rA], TINT[rA]
    cB, tB = COL[rB], TINT[rB]
    ports = {}
    for tag, c, sgn in (("A", cA, 1.0), ("B", cB, -1.0)):
        yT = sgn * YT
        rail(ax, [(X(XC), Y(yT)), (X(XT), Y(yT))], c)                       # C - T
        rail(ax, [(X(XC), Y(yT)), (X(XC), Y(sgn * HH))], c)                 # C - E*
        dbond(ax, [(X(XT), Y(yT)), (X(XT), Y(sgn * SA / 2))], c)            # T - A
        dbond(ax, [(X(WW), Y(sgn * 0.62)), (X(XT - SA / 2), Y(sgn * 0.62))], c)   # E* - A
        dbond(ax, [(X(XT + SA / 2), Y(sgn * 0.62)),
                   (X(XT + SA / 2 + 0.85), Y(sgn * 0.62))], c)              # dangling D
        dbond(ax, [(X(XT + 0.8), Y(sgn * SA / 2)),
                   (X(XT + 0.8), Y(sgn * (SA / 2 + 1.0)))], c)              # dangling D
        ports["T" + tag] = (X(XT), Y(yT))
        ports["E" + tag] = (X(-WW), Y(sgn * 0.62))

        cfg = legs.get("T" + tag)
        if cfg and cfg.get("l", 0) > 0:
            xe = X(XT + cfg["l"])
            rail(ax, [(X(XT), Y(yT)), (xe, Y(yT))], c)
            if cfg.get("lab"):
                ax.text(xe + m * 0.34, Y(yT), cfg["lab"], fontsize=FS_LEG, color=c,
                        fontweight="bold", va="center",
                        ha=("left" if m > 0 else "right"))
        cfg = legs.get("E" + tag)
        if cfg and cfg.get("l", 0) > 0:
            xe = X(-WW - cfg["l"])
            rail(ax, [(X(-WW), Y(sgn * 0.62)), (xe, Y(sgn * 0.62))], c)
            if cfg.get("lab"):
                ax.text(xe - m * 0.34, Y(sgn * 0.62), cfg["lab"], fontsize=FS_LEG,
                        color=c, fontweight="bold", va="center",
                        ha=("right" if m > 0 else "left"))

    tensor_sq(ax, X(XC), Y(YT),  cA, tA, r"$C$")
    tensor_sq(ax, X(XC), Y(-YT), cB, tB, r"$C$")
    tensor_ci(ax, X(XT), Y(YT),  cA, tA, r"$T$")
    tensor_ci(ax, X(XT), Y(-YT), cB, tB, r"$T$")
    hex_seam(ax, X(XC), Y(0.0), tA, tB, r"$E^{*}$")
    sq_seam(ax, X(XT), Y(0.0), tA, tB, r"$\mathcal{A}$")
    return ports

# --------------------------------------------------- stacked result block
def stripe_block(ax, cx, cy, reps, w=3.6, h=5.7, z=5):
    n = len(reps)
    hs = h / n
    ytop = cy + h / 2
    centres = {}
    for i, r in enumerate(reps):
        yt = ytop - i * hs
        ax.add_patch(Rectangle((cx - w / 2, yt - hs), w, hs, fc=TINT[r], ec="none",
                               zorder=z))
        ax.text(cx - w / 2 + 0.5, yt - hs / 2, str(r), fontsize=15, fontweight="bold",
                color=COL[r], ha="center", va="center", zorder=z + 2)
        centres[r] = yt - hs / 2
        if i:
            ax.plot([cx - w / 2, cx + w / 2], [yt, yt], color=INK, lw=1.8,
                    ls=(0, (3, 2)), zorder=z + 1)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="none", ec=INK, lw=2.8,
                           zorder=z + 1))
    return centres

def block_leg(ax, x0, y, r, lab, length=1.7, side=1):
    xe = x0 + side * length
    rail(ax, [(x0, y), (xe, y)], COL[r])
    ax.text(xe + side * 0.34, y, lab, fontsize=FS_LEG, color=COL[r], fontweight="bold",
            va="center", ha=("left" if side > 0 else "right"))

# ==========================================================================
fig, ax = plt.subplots(figsize=(17.3, 15.4))
ax.set_xlim(-1.8, 32.8)
ax.set_ylim(-16.4, 14.4)
ax.set_aspect("equal")
ax.axis("off")

ax.text(15.5, 13.7, "Contracting the four-replica CTMRG window: "
        "three seam quarters, one join, one peak",
        ha="center", va="center", fontsize=FS_TITLE + 2, fontweight="bold", color=INK)

R1 = 6.0
# ------------------------------------------------------------- (A) AB quarter
pA = draw_quarter(ax, 6.5, R1, 1, 2,
                  {"TA": dict(l=3.3, lab=None),
                   "TB": dict(l=1.9, lab=r"$w_2$"),
                   "EA": dict(l=2.55, lab=r"$n_1$"),
                   "EB": dict(l=2.55, lab=r"$e_2$")})
ax.text(6.5, 12.35, r"(A)   AB quarter", ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color=INK)
ax.text(6.5, 11.35, "north seam welds replicas 1 and 2", ha="center", va="center",
        fontsize=FS_NOTE + 1, color=INK)
ax.text(6.5, 1.35, r"stores $\chi_1^4 D^4$    (contract cost $\chi_1^6 D^4$)",
        ha="center", va="center", fontsize=FS_NOTE, color=INK)

# ------------------------------------------------------------- (B) CA quarter
pB = draw_quarter(ax, 19.5, R1, 1, 3,
                  {"TA": dict(l=3.3, lab=None),
                   "TB": dict(l=1.9, lab=r"$w_3$"),
                   "EA": dict(l=2.55, lab=r"$s_1$"),
                   "EB": dict(l=2.55, lab=r"$s_3$")}, mirror=True)
ax.text(19.5, 12.35, r"(B)   CA quarter", ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color=INK)
ax.text(19.5, 11.35, "west seam welds replicas 1 and 3", ha="center", va="center",
        fontsize=FS_NOTE + 1, color=INK)
ax.text(19.5, 1.35, r"stores $\chi_1^4 D^4$    (contract cost $\chi_1^6 D^4$)",
        ha="center", va="center", fontsize=FS_NOTE, color=INK)

# the single shared rail w_1 running from one quarter into the next
ax.plot([13.0], [9.3], marker="o", ms=10, color=COL[1], zorder=6)
ax.text(13.0, 10.05, r"$w_1$ : the only shared rail", ha="center", va="bottom",
        fontsize=FS_LEG, color=COL[1], fontweight="bold")

# ---------------------------------------------------------------- seam key
kx, ky, ks = 26.6, 7.4, 1.15
pos = {1: (kx, ky + 1.5), 2: (kx + 3.4, ky + 1.5),
       3: (kx, ky - 1.5), 4: (kx + 3.4, ky - 1.5)}
ax.plot([pos[1][0], pos[2][0]], [pos[1][1], pos[2][1]], color=GREY, lw=3.0, zorder=2)
ax.plot([pos[3][0], pos[4][0]], [pos[3][1], pos[4][1]], color=GREY, lw=3.0, zorder=2)
ax.plot([pos[1][0], pos[3][0]], [pos[1][1], pos[3][1]], color=GREY, lw=3.0,
        ls=(0, (5, 2)), zorder=2)
ax.plot([pos[2][0], pos[4][0]], [pos[2][1], pos[4][1]], color=GREY, lw=3.0,
        ls=(0, (5, 2)), zorder=2)
ax.plot([pos[1][0], pos[4][0]], [pos[1][1], pos[4][1]], color=GREY, lw=3.0,
        ls=(0, (1, 2.2)), zorder=2)
ax.plot([pos[2][0], pos[3][0]], [pos[2][1], pos[3][1]], color=GREY, lw=3.0,
        ls=(0, (1, 2.2)), zorder=2)
for r, (px, py) in pos.items():
    tensor_sq(ax, px, py, COL[r], TINT[r], str(r), s=ks, fs=FS_TEN)
ax.text(kx + 1.7, ky + 2.35, "AB", ha="center", va="bottom", fontsize=FS_KEY,
        fontweight="bold", color=GREY)
ax.text(kx - 0.8, ky, "CA", ha="right", va="center", fontsize=FS_KEY,
        fontweight="bold", color=GREY)
ax.text(kx + 1.7, ky, "CB", ha="center", va="center", fontsize=FS_KEY,
        fontweight="bold", color=GREY,
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.0))
ax.text(kx + 1.7, ky + 3.35, "the 3 seams pair the 4 replicas", ha="center",
        va="center", fontsize=FS_NOTE, color=INK)
ax.text(kx + 1.7, ky - 2.75, r"each seam $=$ one $E^{*}$ $+$ one $\mathcal{A}$",
        ha="center", va="center", fontsize=FS_NOTE, color=INK)

# ------------------------------------------------------- wrap row1 -> row2
ax.plot([20.0, -0.1], [0.20, 0.20], color=GREY, lw=2.6, zorder=3,
        solid_capstyle="round")
ax.plot([-0.1, -0.1], [0.20, -4.60], color=GREY, lw=2.6, zorder=3,
        solid_capstyle="round")
ax.add_patch(FancyArrowPatch((-0.1, -4.60), (1.72, -4.60), arrowstyle="-|>",
                             mutation_scale=40, lw=2.6, color=GREY, zorder=3))
ax.text(11.6, -0.15, r"contract $w_1$", ha="center", va="top", fontsize=FS_NOTE + 1,
        color=GREY, fontweight="bold")

R2 = -7.0
# ------------------------------------------------------------- (C) joined
cC = stripe_block(ax, 3.6, R2, [1, 2, 3])
block_leg(ax, 1.8, cC[2] + 0.55, 2, r"$e_2$", side=-1)
block_leg(ax, 1.8, cC[2] - 0.55, 2, r"$w_2$", side=-1)
block_leg(ax, 1.8, cC[3] + 0.55, 3, r"$w_3$", side=-1)
block_leg(ax, 1.8, cC[3] - 0.55, 3, r"$s_3$", side=-1)
ax.text(3.7, -3.05, r"(C)   joined on $w_1$", ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color=INK)
ax.text(3.6, -10.95, r"6 rails: $\chi_1^6 D^6$ = 91 GiB at $\chi_1$=24",
        ha="center", va="center", fontsize=FS_NOTE, color=INK)
ax.text(3.6, -11.85, r"cost $\chi_1^7$   $-$   1% of the run", ha="center",
        va="center", fontsize=FS_NOTE, color=INK)

# ------------------------------------------------------------ (D) CB quarter
pD = draw_quarter(ax, 15.4, R2, 1, 4,
                  {"TA": dict(l=0.0), "EA": dict(l=0.0),
                   "TB": dict(l=2.4, lab=r"$s_4$"),
                   "EB": dict(l=2.55, lab=r"$e_4$")})
ax.text(15.6, -1.15, r"(D)   peak  $-$  the CB quarter (replicas 1 and 4) arrives",
        ha="center", va="center", fontsize=FS_TITLE, fontweight="bold", color=INK)

# the two shared rails n_1, s_1  (replica 1's last bonds)
rail(ax, [(5.4, cC[1] - 0.60), (11.2, cC[1] - 0.60), (12.4, pD["EA"][1]),
          (pD["EA"][0], pD["EA"][1])], COL[1])
rail(ax, [(5.4, cC[1] + 0.60), (7.8, cC[1] + 0.60), (7.8, -2.45), (18.6, -2.45),
          (18.6, pD["TA"][1])], COL[1])
ax.text(11.9, -2.15, r"$s_1$", ha="center", va="bottom", fontsize=FS_LEG + 1,
        color=COL[1], fontweight="bold")
ax.text(9.6, -5.55, r"$n_1$", ha="center", va="bottom", fontsize=FS_LEG + 1,
        color=COL[1], fontweight="bold")
ax.text(10.9, -3.60, "replica 1's last 2 rails", ha="center",
        va="center", fontsize=FS_NOTE, color=COL[1], fontweight="bold")
ax.text(15.6, -11.85, r"cost $\chi_1^6\times\chi_1^2=\chi_1^8$   $-$   "
        r"48% of the whole calculation", ha="center", va="center",
        fontsize=FS_NOTE, color=INK)
ax.text(15.6, -12.75, "replica 1 is eliminated here", ha="center", va="center",
        fontsize=FS_NOTE, color=COL[1], fontweight="bold")

# -------------------------------------------------------------- peak result
ax.add_patch(FancyArrowPatch((22.6, R2), (24.8, R2), arrowstyle="-|>",
                             mutation_scale=42, lw=3.2, color=GREY, zorder=3))
cP = stripe_block(ax, 27.4, R2, [2, 3, 4])
block_leg(ax, 29.2, cP[2] + 0.55, 2, r"$e_2$")
block_leg(ax, 29.2, cP[2] - 0.55, 2, r"$w_2$")
block_leg(ax, 29.2, cP[3] + 0.55, 3, r"$w_3$")
block_leg(ax, 29.2, cP[3] - 0.55, 3, r"$s_3$")
block_leg(ax, 29.2, cP[4] + 0.55, 4, r"$e_4$")
block_leg(ax, 29.2, cP[4] - 0.55, 4, r"$s_4$")
ax.text(27.4, -3.05, "result", ha="center", va="center", fontsize=FS_TITLE,
        fontweight="bold", color=INK)
ax.text(27.4, -10.95, r"6 rails $=$ 2 rails $\times$ 3 seams", ha="center",
        va="center", fontsize=FS_NOTE, color=INK)
ax.text(27.4, -11.85, "the memory cannot be lower", ha="center", va="center",
        fontsize=FS_NOTE, color=INK)

# ------------------------------------------------------------------ legend
ly = -14.9
ax.plot([-1.3, 32.3], [-13.7, -13.7], color="#bbbbbb", lw=1.4, zorder=1)
rail(ax, [(-0.9, ly), (1.1, ly)], COL[1])
rail(ax, [(0.1, ly - 0.8), (0.1, ly)], COL[1])
tensor_sq(ax, 0.1, ly, COL[1], TINT[1], r"$C$", s=0.95, fs=14)
ax.text(1.35, ly, "corner", fontsize=FS_KEY, va="center", ha="left", color=INK)
rail(ax, [(3.5, ly), (5.5, ly)], COL[1])
dbond(ax, [(4.5, ly - 0.8), (4.5, ly)], COL[1])
tensor_ci(ax, 4.5, ly, COL[1], TINT[1], r"$T$", r=0.5, fs=14)
ax.text(5.75, ly, "edge", fontsize=FS_KEY, va="center", ha="left", color=INK)
rail(ax, [(8.4, ly + 0.45), (9.9, ly + 0.45)], COL[1])
rail(ax, [(8.4, ly - 0.45), (9.9, ly - 0.45)], COL[2])
hex_seam(ax, 9.9, ly, TINT[1], TINT[2], r"$E^{*}$", hh=1.15, ww=0.6, fs=13)
ax.text(10.7, ly, "seam edge: 2 replicas", fontsize=FS_KEY, va="center", ha="left",
        color=INK)
dbond(ax, [(16.2, ly + 0.45), (17.5, ly + 0.45)], COL[1])
dbond(ax, [(16.2, ly - 0.45), (17.5, ly - 0.45)], COL[2])
sq_seam(ax, 18.2, ly, TINT[1], TINT[2], r"$\mathcal{A}$", s=1.5, fs=14)
ax.text(19.1, ly, "dressed region: 2 replicas", fontsize=FS_KEY, va="center",
        ha="left", color=INK)
rail(ax, [(25.6, ly + 0.45), (27.2, ly + 0.45)], INK)
ax.text(27.4, ly + 0.45, r"thick $=$ $\chi_1$ boundary rail", fontsize=FS_KEY,
        va="center", ha="left", color=INK)
dbond(ax, [(25.6, ly - 0.45), (27.2, ly - 0.45)], INK)
ax.text(27.4, ly - 0.45, r"thin $=$ $D=2$ bond", fontsize=FS_KEY, va="center",
        ha="left", color=INK)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
