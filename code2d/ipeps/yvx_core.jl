# yvx_core.jl -- THE consolidated Y-vertex library (v13 cleanup, 2026-08-06).
# =============================================================================
# Infinite-lattice iPEPS contraction of the tripartite junction (T-junction of
# regions A/B/C) for the RK-Ising testbed.  Physics + certification status:
# note_yvertex/main.pdf.  Pipeline (one section per block below):
#
#   1. BULK + EXACT REFERENCES   ising_tensors (via rk_ctmrg.jl), onsager_ss,
#                                twist weights Qpair/seam_G (via rk_bmps.jl).
#   2. ARM (boundary MPS)        solve_arm/arm_step: absorb rows, density
#                                truncation (multiplet-safe) + polish;
#                                canonicalize: sum M M' = Id + junk trim.
#   3. WALLS (intertwiners)      doubled column D, left wall E'D=kME',
#                                right wall DE=kEM (both needed: E != E'^T),
#                                dense-SVD solve, spectral-tail budget floor.
#   4. CORNER                    the user's generalized eigen equation
#                                (E' c E a = lam M c M), all branches;
#                                the physical branch is selected by the
#                                EXACT-ANSWER cross gates, never by |lam|.
#   5. CAPS + CROSS READOUT      channel fixed-point caps (growL/R/U/D),
#                                cross diagram (4 caps + 4 corners + center),
#                                gates X1 (NN vs Onsager) / X2 (diagonal vs
#                                corner-free benchmark) / readout mirror;
#                                corner-free strips strip_nn/diag_benchmark.
#   6. SEAM BLOCKS               zip_eig ladder fixed points; universal
#                                cycle blocks W1 (untwisted) / W2 (swap
#                                double); seam_w assembles full-seam caps.
#   7. ASSEMBLIES                octahedron K_{2,2,2} (octa_*, peak chi^8)
#                                and the equivalent ring (ring_Omega2);
#                                proven equal to 1e-16 (yvertex12 gate).
#
# NEGATIVE CONTROL kept on purpose: solve_corner (the REFUTED v3 closed-ring
# corner) -- yvertex11.jl reruns its refutation (X1=0.16) as a standing gate.
# HISTORY: everything superseded lives in code2d/trash/ (never deleted).
# =============================================================================
include(joinpath(@__DIR__, "rk_ctmrg.jl"))                 # ising_tensors, BETA_C
include(joinpath(@__DIR__, "support", "rk_bmps.jl"))    # bond_M, PSWAP2, SA4/SB4/SC4
using Printf, LinearAlgebra, Random

# ---------------------------------------------------------------------------
# Qpair: split one Ising bond weight W into two symmetric halves.
#
#     ---W---   =   ---Q---Q---        Q = sqrt(W) (symmetric), Qi = Q^-1
#
# in : beta;  out: (Q, Q^-1), both 2x2 symmetric.  W = [[e^b,e^-b],[e^-b,e^b]];
# closed form via c = sqrt(2cosh b), s = sqrt(2sinh b): Q = [(c+s)/2 (c-s)/2;
# (c-s)/2 (c+s)/2]  (check: (Q^2)_diag = (c^2+s^2)/2 = e^b, off = (c^2-s^2)/2
# = e^-b).  Q dresses every bulk-tensor leg; Qi undoes it on seam legs
# (seam_G), so an inserted untwisted rung is exactly a no-op.
# ---------------------------------------------------------------------------
"Q with Q*Q = W (symmetric sqrt), and its inverse.
 GUARD (review 2026-08-10): Q is singular at beta=0 (s=0) and inv(Q) loses
 precision for beta << 1e-3; production betas are >= 0.05."
function Qpair(beta)
    @assert beta > 1e-6 "Qpair: beta=$(beta) too small -- inv(Q) singular/ill-conditioned"
    c = sqrt(2cosh(beta)); s = sqrt(2sinh(beta))
    Q = [(c+s)/2 (c-s)/2; (c-s)/2 (c+s)/2]
    (Q, inv(Q))
end

# ---------------------------------------------------------------------------
# seam_G: the k-replica twist weight that sits ON one seam rung (bare legs).
#
# in : k           number of replicas the weight spans (k=1 untwisted is
#                  exactly Id; this file's universal blocks call it with k=2)
#      beta        inverse temperature (fixes Q of Qpair)
#      Xi, Xj      region labels ('A'/'B'/'C') on the two sides of the rung
#      pA, pB, pC  replica permutations of the three regions (sector choice)
# out: Symmetric 2^k x 2^k weight G on the two BARE half-bond legs;
#      G(k=1, untwisted) = Id exactly (inserting it anywhere is a no-op).
#
#        |  (2^k bare leg, side Xi)
#       {G}        G = (Q^-1)^(x)k  M_twist  (Q^-1)^(x)k,   G(untwisted) = Id
#        |  (2^k bare leg, side Xj)
#
# STEP 1 -- UNDRESS FACTOR: Qi = Q^-1 from Qpair.  Every bulk-tensor leg
#   already carries one Q half-bond, so a seam rung must strip one Q from
#   EACH side before inserting the twist -- otherwise the bond weight is
#   double-counted: with G in place, Q^(x)k . G . Q^(x)k = M_twist exactly.
# STEP 2 -- TWISTED BOND: M = bond_M(k, beta, Xi, Xj, pA, pB, pC), the
#   k-replica bond weight with the two sides' sector permutations applied.
# STEP 3 -- SANDWICH: G = Qi^(x)k * M * (Qi^(x)k)' (kron over replicas,
#   k = 1 shortcut skips the kron), returned Symmetric.
#
# WHY relative twist only: bond_M depends only on the RELATIVE permutation
# p_Xj p_Xi^-1 -- that is why the universal 2-cycle block is built on the
# fixed ('A','B') seam.  Historical BC-seam bug: pinning the swap to 'A'
# while passing the real ('C','B') labels left the BC seam UNTWISTED
# (affected O4/O2B/O2C before the v11 fix in seam_w).
# ---------------------------------------------------------------------------
"gauged k-replica seam weight on bare half-bond legs; G(k=1,untwisted)=I."
function seam_G(k, beta, Xi, Xj, pA, pB, pC)
    # STEP 1: Qi = Q^-1 -- undress one Q half-bond from each side of the rung
    _, Qi = Qpair(beta)
    # STEP 2: sector-twisted k-replica bond weight (only the relative perm matters)
    M = bond_M(k, beta, Xi, Xj, pA, pB, pC)
    # STEP 3: build Qi^(x)k and sandwich: G = Qk * M * Qk' (symmetrized)
    Qk = k == 1 ? Qi : kron(ntuple(_->Qi, k)...)
    Symmetric(Qk * M * Qk')
end

# ---------------------------------------------------------------------------
# arm_step: absorb ONE bulk row into the boundary MPS, truncate back to chi.
#
# in : M[chi,2,chi]  current arm (left virtual l, bare middle u, right r)
#      a[u,l,d,r]    bulk tensor (fully leg-symmetric, entrywise > 0)
#      chi           target bond; strict=true skips multiplet guard (polish)
# out: (Mn, U)       new arm + this step's truncation isometry U[2chi x keep]
#
# STEP 1 -- GROW: the arm swallows one bulk row, virtual bond chi -> 2chi.
#
#     l --(M)-- r                (l,al) --[B]-- (r,ar)
#         |u                                |
#         |al          =>                   |dn (new bare leg)
#    al --[a]-- ar
#         |dn              B[(l,al), dn, (r,ar)] = sum_u M[l,u,r] a[u,al,dn,ar]
#
#   M's bare leg u contracts a's top leg; a's side legs (al,ar) join the
#   virtual bond.  Packing convention: (al-1)*chi + l  (al slow, l fast) --
#   the SAME convention as doubled_col/E-walls, do not change one alone.
#
# STEP 2 -- WEIGHT: channel density matrix rho = environment of the grown arm.
#   INIT: rho = Id/(2chi), the maximally-mixed state.  Any full-rank PSD
#   init works: the map is completely positive, its leading eigenmatrix
#   rho* is PSD, and <Id, rho*> = tr(rho*) > 0, so the power iteration can
#   never be orthogonal to the leading mode (Perron-Frobenius) -- it
#   converges geometrically at rate (lam2/lam1)^n.
#   Iterate   rho <- sum_p B_p' rho B_p  (symmetrize, trace-normalize)
#   to its fixed point (<=400 sweeps, tol 1e-14).  NOTHING GROWS here:
#   rho stays a FIXED 2chi x 2chi matrix; each sweep adds one rung of the
#   semi-infinite double-layer ladder and the rung's bare leg p is summed
#   between B' and B on the spot (the sum_p) -- physical legs never dangle
#   or accumulate.  The infinite tail ...BBB is REPRESENTED by the fixed
#   point, never contracted explicitly.  (Same pattern as every other
#   "absorb infinity" object in this file: right_fixed_point, cap_fp,
#   zip_eig -- all fixed-dimension channel power iterations.)
#   Physics: rho is the reduced density handed down by the semi-infinite
#   chain -- choosing the basis from rho is optimal for the WHOLE
#   half-infinite boundary, not just for one local column.
#
# STEP 3 -- SELECT: eigendecompose rho, keep the `keep` heaviest directions.
#   keep = min(chi, 2chi).  Multiplet guard (skipped when strict=true):
#   if gap(value[keep], value[keep+1]) < 1e-6 * value[1], the cut falls
#   INSIDE a degenerate multiplet -> keep fewer until the cut sits in a
#   true gap.  Why: cutting through a multiplet lets U rotate arbitrarily
#   between sweeps, the (M,U) pairing decoheres, and every downstream
#   object jitters (measured v3-era lesson).
#
# STEP 4 -- PROJECT: Mn_p = U' B_p U  for each bare component p
#   (2chi -> keep on both virtual legs; the bare leg is untouched --
#   twist-before-compress).
#
# STEP 5 -- NORMALIZE: divide by the Frobenius norm (the channel eigenvalue
#   is bookkept elsewhere; only the tensor's direction matters here).
#   U is returned because the negative-control corner needs the isometry
#   PAIRED with this M.
function arm_step(M, a, chi; strict=false)
    cm = size(M,1); d = size(a,1)
    # STEP 1: grow -- B[(l,al), dn, (r,ar)] = sum_u M[l,u,r] a[u,al,dn,ar]
    B = zeros(cm*d, d, cm*d)                       # [(l,al), dnew, (r,ar)]
    @inbounds for u in 1:d, al in 1:d, dn in 1:d, ar in 1:d
        w = a[u,al,dn,ar]; w == 0 && continue
        B[(al-1)*cm .+ (1:cm), dn, (ar-1)*cm .+ (1:cm)] .+= w .* (@view M[:,u,:])
    end
    dm = size(B,1)
    # STEP 2: channel density fixed point  rho <- sum_p B_p' rho B_p
    rho = Matrix{Float64}(I, dm, dm)/dm
    for it in 1:400
        nr = zeros(dm,dm)
        for p in 1:d
            Bp = @view B[:,p,:]
            nr .+= Bp' * rho * Bp
        end
        nr = (nr+nr')/2; nr ./= tr(nr)
        conv = norm(nr - rho); rho = nr
        conv < 1e-14 && break
    end
    # STEP 3: spectral selection + multiplet-safe cut
    F = eigen(Symmetric(rho)); ord = sortperm(F.values, rev=true)
    keep = min(chi, dm)
    if !strict
        while keep > 2 && keep < dm &&
              (F.values[ord[keep]] - F.values[ord[keep+1]]) < 1e-6*F.values[ord[1]]
            keep -= 1
        end
    end
    U = F.vectors[:, ord[1:keep]]
    # STEP 4: project both virtual legs back (bare leg untouched)
    Mn = zeros(size(U,2), d, size(U,2))
    for p in 1:d; Mn[:,p,:] = U' * (@view B[:,p,:]) * U; end
    # STEP 5: normalize; return the paired isometry too
    eta = sqrt(sum(abs2, Mn)); Mn ./= eta
    (Mn, U)
end

# ---------------------------------------------------------------------------
# solve_arm: uniform boundary MPS of a half plane = fixed point of arm_step.
#
# in : beta, chi   bulk temperature and target bond dimension
#      iters=2000  power-iteration budget;  tol=5e-14 stationarity tolerance
# out: (M, U)      M[chi_kept,2,chi_kept] stationary arm tensor, plus
#                  U[2*chi_kept x chi_kept], the truncation isometry of the
#                  LAST arm_step applied to this very M (consistent pair).
#
#    (M) . row  ~  lambda_row (M)      --(M)--(M)--(M)--   bare legs |
#                                          |    |    |     NEVER truncated
#
# The middle (bare) leg is NEVER truncated: twist-before-compress -- the
# replica twist weights {G} attach to bare half-bond legs (seam_G/zip_eig),
# so any compression there would bias every twisted sector.
#
# STEP 1 -- SEED: chi=1 start, M[1,:,1] = (1.0, 0.7)/norm -- a generic seed
#   with deliberately unequal bare components.  The virtual bond then
#   doubles 1 -> 2 -> 4 -> ... until it caps at chi: arm_step keeps
#   min(chi_target, 2*chi_current) (or fewer when the multiplet guard bites).
# STEP 2 -- POWER ITERATION: M <- arm_step(M, a, chi), up to `iters` times.
#   While the bond is still growing (size mismatch) just accept the new M.
#   Once sizes match, sign-fix first (sg = sign(sum(Mn .* M)); the channel
#   fixed point is defined only up to +-) and stop when ||Mn - M|| < tol.
# STEP 3 -- POLISH at frozen size (<= 100 sweeps, strict=true): rerun
#   arm_step with chi = current size and the multiplet guard OFF.  Why:
#   (a) the guard could shrink the frozen size mid-polish and desync (M,U);
#   (b) the RETURNED U must be the isometry of the very last step applied
#   to the returned M -- the negative-control corner consumes the PAIR.
# STEP 4 -- GUARD: assert size(U,1) == size(M,1)*size(M,2), i.e. U really
#   maps the grown 2chi space of THIS M (arm (M,U) consistency).
# ---------------------------------------------------------------------------
"uniform boundary MPS of a half-plane: power-iterate arm_step until the
 (sign-fixed) tensor is stationary, then polish at frozen size so the
 returned (M, U) pair is consistent.  M[chi,2,chi]: middle leg = BARE bond
 (never truncated -- twist-before-compress principle)."
function solve_arm(beta, chi; iters=2000, tol=5e-14)
    a, _ = ising_tensors(beta)
    # STEP 1: generic chi=1 seed (deliberately unequal bare components)
    M = zeros(1,2,1); M[1,1,1]=1.0; M[1,2,1]=0.7; M ./= norm(M)
    U = nothing
    for it in 1:iters
        # STEP 2: power iteration; sign-fix + tol test only once the size is stationary
        Mn, Un = arm_step(M, a, chi)
        if size(Mn,1) == size(M,1)
            sg = sign(sum(Mn .* M)); sg == 0 && (sg = 1.0)
            Mn .*= sg
            dl = norm(Mn - M); M = Mn; U = Un
            dl < tol && break
        else
            M = Mn; U = Un
        end
    end
    # STEP 3: polish at frozen size, strict=true (multiplet guard off) keeps (M,U) paired
    for j in 1:100                             # polish at frozen size
        Mn, Un = arm_step(M, a, size(M,1); strict=true)
        sg = sign(sum(Mn .* M)); sg == 0 && (sg = 1.0); Mn .*= sg
        dl = norm(Mn - M); M = Mn; U = Un
        dl < tol && break
    end
    # STEP 4: consistency guard -- U must map the grown 2chi space of the returned M
    @assert size(U,1) == size(M,1)*size(M,2) "arm (M,U) inconsistent"
    (M, U)
end


# ---------------------------------------------------------------------------
# onsager_ss: EXACT nearest-neighbour <s s'> of the 2D Ising model at
# inverse temperature beta (Onsager closed form, AGM elliptic integrals).
# External, code-independent anchor for the XA / X1 gates.
#
# in : beta      isotropic square-lattice inverse temperature
# out: <s0 s1>   on one NN bond   (s)--(s')   (pure number, no lattice built)
#
# STEP 1 -- MODULUS: k1 = 2 sinh(2b) / cosh(2b)^2   (Onsager's modulus).
# STEP 2 -- COMPLETE ELLIPTIC K(k1) by AGM: iterate (A,B) <- ((A+B)/2,
#   sqrt(AB)) from (1, sqrt(1-k1^2)) down to |A-B| < 1e-15, then
#   K = pi / (2 AGM).  AGM converges quadratically, so the reference is
#   machine-exact in a handful of steps -- the point of an EXACT-ANSWER
#   gate anchor is that it carries no numerical tail of its own.
# STEP 3 -- CLOSED FORM:
#   <s s'> = coth(2b) [ 1 + (2/pi)(2 tanh(2b)^2 - 1) K(k1) ] / 2 ;
#   the trailing /2 converts Onsager's per-site energy expression (two
#   bonds per site) into the per-bond correlator.
# ---------------------------------------------------------------------------
"exact NN correlation via AGM."
function onsager_ss(beta)
    # STEP 1: Onsager modulus k1
    k1 = 2*sinh(2beta)/cosh(2beta)^2
    A, B = 1.0, sqrt(1-k1^2)
    # STEP 2: AGM loop -> complete elliptic K(k1) = pi/(2*AGM(1, sqrt(1-k1^2)))
    while abs(A-B) > 1e-15; A, B = (A+B)/2, sqrt(A*B); end
    K = pi/(2A)
    # STEP 3: closed form; /2 turns the per-site (2 bonds/site) form into per-bond <ss'>
    coth(2beta)*(1 + (2/pi)*(2*tanh(2beta)^2 - 1)*K)/2
end

# ---------------------------------------------------------------------------
# !!! NEGATIVE CONTROL -- REFUTED, DO NOT USE FOR PHYSICS !!!
# solve_corner [v3 closed-ring corner, kept ONLY as a standing gate].
#
# in : M[chi,2,chi]  arm tensor (from solve_arm)
#      U[2chi,chi]   the arm-step truncation isometry PAIRED with this M
#      a[u,l,d,r]    bulk tensor
# out: c[chi,chi]    fixed point of the ring-grown corner map (normalized)
#
# THE REFUTATION (why this function survives): it converges cleanly and
# passes its OWN ring gate (tr[c M c M c M c M] self-consistency) yet FAILS
# the environment-consistent cross gates at O(1): X1 = 0.16, X2 = 0.53.
# Lesson: self-consistency of a closed ring does NOT certify a corner --
# only the EXACT-ANSWER cross gates do.  yvertex11.jl reruns this failure
# as a standing regression gate (if it ever "passes", the gates broke).
# NEVER feed its c into pick_corner / ring_Omega2 / octa_* physics; the
# certified corner comes from corner_maps11 + corner_branches + cross_gates.
#
#        (Mv)
#          |
#   (Mh)--<c>     grown L-hook  c' = U' [ Mh c Mv a ] U  (arm isometry U)
#      [a]
#
# STEP 1 -- INIT: c = Id/||Id||, then fixed-point iterate (<= 3000 sweeps).
# STEP 2 -- HOOK: pre-contract the horizontal arm onto the old corner,
#   Mc[:,u,:] = M[:,u,:] * c   -- Mc[gh, u, be], be = inner virtual bond.
# STEP 3 -- GROW THE L: absorb one bulk tensor a and the vertical arm,
#
#   gh --(Mh)--<c>--(Mv)-- gv       cb[(dn-1)*chi+gh, (rr-1)*chi+gv] +=
#          |u         |l               a[u,l,dn,rr] * (Mc_u * M_l)[gh,gv]
#         [a]-- rr
#          |dn            blk = Mc[:,u,:] * M[:,l,:] : [gh,be]*[be,gv];
#                         packing (dn-1)*chi + gh (dn slow, gh fast) -- the
#                         SAME convention as arm_step's grown B ((al-1)*chi+l).
# STEP 4 -- COMPRESS with the ARM isometry: cn = U' cb U.  THIS is the
#   refuted assumption -- it recycles the arm channel's truncation basis
#   for the corner environment, which the cross gates show is wrong at O(1).
# STEP 5 -- NORMALIZE + SIGN-FIX: cn /= ||cn||, align sign with the previous
#   c (fixed points are defined up to +-), stop at ||cn - c|| < 1e-13.
# ---------------------------------------------------------------------------
function solve_corner(M, U, a)
    chi = size(M,1); d = size(a,1)
    # STEP 1: init corner = normalized identity, then fixed-point iterate
    c = Matrix{Float64}(I, chi, chi); c ./= norm(c)
    for it in 1:3000
        cb = zeros(chi*d, chi*d)                 # [(gh,dn),(gv,rr)] idx (dn-1)*chi+gh
        Mc = zeros(chi, d, chi)                  # Mh . c : [gh, u, be]
        # STEP 2: hook the horizontal arm onto the old corner: Mc_u = Mh_u * c
        for u in 1:d; Mc[:,u,:] = (@view M[:,u,:]) * c; end
        # STEP 3: grow the L-hook: cb[(dn-1)*chi+gh,(rr-1)*chi+gv] += a[u,l,dn,rr]*(Mc_u*M_l)
        for u in 1:d, l in 1:d, dn in 1:d, rr in 1:d
            w = a[u,l,dn,rr]; w == 0 && continue
            blk = (@view Mc[:,u,:]) * (@view M[:,l,:])    # [gh,be]*[be,gv] = [gh,gv]
            cb[(dn-1)*chi .+ (1:chi), (rr-1)*chi .+ (1:chi)] .+= w .* blk
        end
        # STEP 4: compress with the ARM isometry U -- the REFUTED assumption
        cn = U' * cb * U
        # STEP 5: normalize, sign-fix against previous c, converge at 1e-13
        nn = norm(cn); cn ./= nn
        sg = sign(sum(cn .* c)); sg == 0 && (sg=1.0); cn .*= sg
        conv = norm(cn - c); c = cn
        conv < 1e-13 && break
    end
    c
end



# ---------------------------------------------------------------------------
# zip_eig: seam-ladder transfer fixed point ("what the semi-infinite seam
# tail presents at the junction").
#
# in : M[chi,2,chi]  arm;  G [2^nrep x 2^nrep]  gauged seam weight on the
#      bare rung legs (seam_G; G = Id -> untwisted);  nrep replicas;
#      dir = :R / :L  which side the transfer sweeps from
# out: (W, lam)   W[bot,top], chi^nrep x chi^nrep leading eigenmatrix,
#      lam its eigenvalue.  The extensive part lives in lam^N and cancels
#      in the 5-sector combination -- only W's direction is consumed.
#
#   --(Mk)--(Mk)-- ...
#      |      |               W  <-  sum_pq G[p,q] Mk_q' W Mk_p    (:R)
#     {G}    {G}       ->     W  <-  sum_pq G[p,q] Mk_q  W Mk_p'   (:L)
#      |      |
#   --(Mk)--(Mk)-- ...
#
# STEP 1 -- REPLICATE: build the nrep-replica arm Mk by repeated kron.
#   Bare-leg packing (q-1)*d0 + p (new copy q slow, existing p fast);
#   virtual legs kron(M_q, Mk_p) (existing block fast).  Result: replica
#   axis order is fastest-first 1,2,...,nrep -- the SAME order seam_w
#   assumes when it permutes cycle blocks back to standard order.
# STEP 2 -- INIT: W = Id/sqrt(d), legs w[bot, top].
# STEP 3 -- ITERATE the ladder transfer: one rung applies both arm copies
#   and the twist weight, NW += G[p,q] * Mk_q' W Mk_p for dir=:R (:L is
#   the mirror with the adjoints swapped).  Zero G entries are skipped.
# STEP 4 -- NORMALIZE (lam = norm) + SIGN-FIX against the previous iterate
#   (power iteration is sign-blind), converge to 1e-13; return (W, lam).
# ---------------------------------------------------------------------------
"leading eigenvector of the zip with gauged weight G on nrep replicas."
function zip_eig(M, G, nrep; dir=:R, iters=30000)
    chi = size(M,1)
    Mk = M
    # STEP 1: replicate the arm; bare packing (q-1)*d0+p, replica 1 fastest
    for j in 2:nrep
        chi0 = size(Mk,1); d0 = size(Mk,2)
        Mk2 = zeros(chi0*chi, d0*2, chi0*chi)
        for p in 1:d0, q in 1:2
            Mk2[:, (q-1)*d0 + p, :] = kron((@view M[:,q,:]), (@view Mk[:,p,:]))
        end
        Mk = Mk2
    end
    d = size(Mk,1); dd = size(Mk,2)
    # STEP 2: init the cap
    W = Matrix{Float64}(I, d, d) ./ sqrt(d)      # w[bot, top]
    lam = 0.0
    for it in 1:iters
        NW = zeros(d, d)
        for p in 1:dd, q in 1:dd
            # STEP 3: one ladder rung -- both arm copies + twist weight G[p,q]
            g = G[p,q]; g == 0 && continue
            if dir === :R
                NW .+= g .* ((@view Mk[:,q,:])' * W * (@view Mk[:,p,:]))
            else
                NW .+= g .* ((@view Mk[:,q,:]) * W * (@view Mk[:,p,:])')
            end
        end
        # STEP 4: normalize, sign-fix, converge to 1e-13
        lam = norm(NW); NW ./= lam
        sg = sign(sum(NW .* W)); sg == 0 && (sg=1.0); NW .*= sg
        cv = norm(NW - W); W = NW
        cv < 1e-13 && break
    end
    (W, lam)
end

# ---------------------------------------------------------------------------
# perm_cycles: cycle decomposition of a one-line permutation array,
# e.g. (13)(24) = [3,4,1,2] -> [[1,3],[2,4]].  Decides which replicas
# share a block on each seam (1-cycle -> W1, 2-cycle -> W2 in seam_w).
# ---------------------------------------------------------------------------
"cycles of a permutation."
function perm_cycles(p)
    n = length(p); seen = falses(n); cyc = Vector{Vector{Int}}()
    for i in 1:n
        seen[i] && continue
        c = Int[]; j = i
        while !seen[j]; push!(c, j); seen[j] = true; j = p[j]; end
        push!(cyc, c)
    end
    cyc
end

# ---------------------------------------------------------------------------
# seam_w: full k-replica seam cap, assembled from the UNIVERSAL cycle
# blocks (W1 untwisted / W2 swap) instead of one chi^k zip_eig solve.
#
# in : M[chi,2,chi], beta, k, seam sides (Xi,Xj), replica perms pA,pB,pC,
#      dir (forwarded to zip_eig)
# out: (Wfull/|Wfull|, lams, cycle lengths)   Wfull chi^k x chi^k in
#      STANDARD replica order 1..k; lams = per-cycle eigenvalues.
#      Frobenius norm 1 per block -- the 5-sector counting cancels all
#      of it, so only directions matter.
#
#   q = p_Xj^-1 p_Xi -> cycles -> 1-cycle: W1 -> kron -> permute axes 1..k
#                                 2-cycle: W2    blocks
#
# STEP 1 -- RELATIVE TWIST: q[r] = invperm(permof(Xj))[permof(Xi)[r]].
#   Only the relative permutation matters (bond_M's exponent depends on
#   p_j p_i^-1 alone -- the bond_M property in the seam_G header).
# STEP 2 -- CYCLES: perm_cycles(q); only involutions are expected
#   (asserted), so every cycle is a 1-cycle or a 2-cycle.
# STEP 3 -- BLOCKS: per cycle, G = Id_2 (1-cycle, untwisted) or the
#   universal swap weight seam_G(2,beta,'A','B',PSWAP2,[1,2],[1,2])
#   (2-cycle), then zip_eig(M,G,n).  BUGFIX (v11 review, 08-06): the
#   swap-twisted 2-cycle block must carry a RELATIVE swap.  bond_M only
#   sees permof(Xi)/permof(Xj); passing the real (Xi,Xj) with the swap
#   pinned to 'A' left the BC seam ('C','B') UNTWISTED.  Since only the
#   relative twist matters, the block is built on the fixed ('A','B')
#   seam.  Affected before the fix: O4 / O2B / O2C BC-seam blocks.
# STEP 4 -- ASSEMBLE: Wfull = kron(block_j, ..., block_1) -- earlier
#   blocks fastest, so the current replica axis order (fastest-first) is
#   exactly `order` = the concatenated cycle members.
# STEP 5 -- PERMUTE back to standard order 1..k when order != 1..k:
#   reshape to 2k chi-axes (k bot axes then k top axes, BOTH in `order`),
#   permutedims with perm = [invperm(order); k .+ invperm(order)],
#   reshape back to chi^k x chi^k.
# STEP 6 -- NORMALIZE (Frobenius) and return with lams + cycle lengths.
# ---------------------------------------------------------------------------
"full seam eigenvector in replica order + eigenvalue list per cycle."
function seam_w(M, beta, k, Xi, Xj, pA, pB, pC; dir=:R)
    chi = size(M,1)
    permof(X) = X=='A' ? pA : (X=='B' ? pB : pC)
    # STEP 1: only the RELATIVE permutation matters (bond_M property)
    q = [invperm(permof(Xj))[permof(Xi)[r]] for r in 1:k]   # relative twist
    # STEP 2: cycle decomposition (involutions only: 1- and 2-cycles)
    cycs = perm_cycles(q)
    blocks = Matrix{Float64}[]; lams = Float64[]
    order = Int[]
    for c in cycs
        n = length(c)
        @assert n <= 2 "only involutions expected"
        # BUGFIX (v11 review, 08-06): the universal swap-twisted 2-cycle block
        # must carry a RELATIVE swap.  bond_M only sees permof(Xi)/permof(Xj);
        # passing the real (Xi,Xj) with the swap pinned to 'A' left the BC seam
        # ('C','B') UNTWISTED.  Only the relative twist matters (bond_M's
        # exponent depends on p_j p_i^-1), so build the block on the fixed
        # ('A','B') seam.  Affected before the fix: O4/O2B/O2C BC-seam blocks.
        # STEP 3: universal block per cycle -- W1 (Id) or W2 (fixed ('A','B') relative swap)
        G = n == 1 ? Matrix{Float64}(I,2,2) :
                     Matrix(seam_G(2, beta, 'A', 'B', PSWAP2, [1,2], [1,2]))
        Wb, lb = zip_eig(M, G, n; dir=dir)
        push!(blocks, Wb); push!(lams, lb)
        append!(order, c)
    end
    # assemble kron over blocks (cycle order), then permute replica axes to 1..k
    # STEP 4: kron blocks in cycle order (earlier blocks fastest)
    Wfull = blocks[1]
    for j in 2:length(blocks); Wfull = kron(blocks[j], Wfull); end
    # current axis order (fastest-first) = order[1], order[2], ...
    # STEP 5: permute replica axes back to standard order 1..k (bot and top alike)
    if order != collect(1:k)
        dims = ntuple(_->chi, 2k)  # (bot axes k, top axes k) both in `order`
        A = reshape(Wfull, ntuple(_->chi, 2k))
        pos = invperm(order)
        perm = vcat(pos, k .+ pos)
        A = permutedims(A, Tuple(perm))
        Wfull = reshape(A, chi^k, chi^k)
    end
    (Wfull ./ norm(Wfull), lams, [length(c) for c in cycs])
end



# ---------------------------------------------------------------------------
# right_fixed_point: the SPD channel fixed point that defines the gauge.
#
# in : M[chi,d,chi]   arm tensor (d = bare dim, 2 here)
# out: (X, nu)        X[chi,chi] symmetric, Frobenius-normalized, with
#                     sum_p M_p X M_p' = nu X;  nu = channel eigenvalue.
#
#   --(M)--+              +--
#      |   X    =   nu    X          X = R^2  (R feeds canonicalize)
#   --(M)--+              +--
#
# STEP 1 -- INIT: X = Id/||Id||.  An SPD start keeps the iteration inside
#   the PSD cone: the map X -> sum_p M_p X M_p' is completely positive, so
#   its leading eigenmatrix is PSD and NO sign-fix is needed (contrast
#   solve_arm / zip_eig / cap_fp, whose iterates need the +- alignment).
# STEP 2 -- POWER ITERATION (<= 6000 sweeps): X <- sum_p M_p X M_p',
#   re-symmetrize (X+X')/2 against roundoff, normalize by nu = ||X||_F
#   (nu converges to the channel eigenvalue).
# STEP 3 -- CONVERGE when ||Xn - X|| < 1e-15.  The near-null directions of
#   the converged X are exactly the junk that canonicalize trims away.
# ---------------------------------------------------------------------------
"right fixed point of the M-channel: sum_p M_p X M_p' = nu X (X SPD)."
function right_fixed_point(M)
    chi = size(M,1); d = size(M,2)
    # STEP 1: SPD start -- the CP channel map keeps the iterate PSD, no sign-fix needed
    X = Matrix{Float64}(I, chi, chi); X ./= norm(X); nu = 0.0
    for it in 1:6000
        Xn = zeros(chi, chi)
        # STEP 2: apply the channel; then symmetrize and normalize by nu = ||X||_F
        for p in 1:d; Xn .+= (@view M[:,p,:]) * X * (@view M[:,p,:])'; end
        Xn = (Xn + Xn')/2
        nu = norm(Xn); Xn ./= nu
        cv = norm(Xn - X); X = Xn
        cv < 1e-15 && break
    end
    (X, nu)
end


# ---------------------------------------------------------------------------
# canonicalize: gauge the arm right-canonical AND trim junk directions.
#
# in : M[chi,d,chi]   arm tensor;  rtol (default 1e-12) = null-space cutoff,
#                     scanned by the drivers with the left-wall residual as
#                     merit function -> chi_eff.
# out: (Mt[chi_eff,d,chi_eff], canon_err = ||sum_p Mt_p Mt_p' - I||, chi_eff)
#
#   Mt_p = Rhi M_p Rh / sqrt(nu)   =>   --(Mt)--+
#                                          |    | = Id
#                                       --(Mt)--+
#
# STEP 1 -- FIXED POINT: (X, nu) = right_fixed_point(M); X plays R^2.
# STEP 2 -- TRIM: eigendecompose X, keep only eigendirections with value >
#   rtol * max.  WHY (measured lesson, beta=0.05, chi=4): near-null
#   directions of X carry no state weight but POISON the intertwiner's
#   leading eigenspace -- fidelity saturated exactly while the per-p wall
#   residual sat at 2.5e-3.  All downstream objects live in chi_eff = #kept.
# STEP 3 -- GAUGE FACTORS: Rh = V diag(sqrt vals)  (chi x chi_eff) and
#   Rhi = diag(1/sqrt vals) V'  (chi_eff x chi), so Rhi*Rh = Id_chi_eff --
#   a rectangular square-root gauge that trims and transforms in one go.
# STEP 4 -- TRANSFORM: Mt_p = Rhi M_p Rh / sqrt(nu).  The 1/sqrt(nu)
#   rescales the channel eigenvalue to 1 so the canonical identity closes:
#   sum_p M_p X M_p' = nu X  and  X = Rh Rh'  (exact up to the trimmed
#   tail)  =>  sum_p Mt_p Mt_p' = Rhi X Rhi' = Id.
# STEP 5 -- CERTIFY: recompute S = sum_p Mt_p Mt_p' and return
#   canon_err = ||S - I|| so callers gate on the gauge instead of trusting it.
# ---------------------------------------------------------------------------
"Right-canonicalize AND trim the null space of R (junk directions from an
 oversized chi carry no state weight and pollute the intertwiner's leading
 eigenspace -- diagnosed at beta=0.05, chi=4: fidelity saturated exactly while
 the per-p residual sat at 2.5e-3).  Keeps directions with R-eigenvalue
 > rtol * max; all downstream objects then live in chi_eff.
 Returns (Mt[chi_eff,d,chi_eff], canon_err, chi_eff)."
function canonicalize(M; rtol=1e-12)
    # STEP 1: SPD channel fixed point X = R^2 and channel eigenvalue nu
    X, nu = right_fixed_point(M)
    F = eigen(Symmetric((X + X')/2))
    # STEP 2: junk trim -- drop near-null X-directions (beta=0.05, chi=4 wall lesson)
    keep = findall(F.values .> rtol * maximum(F.values))
    V = F.vectors[:, keep]; vals = F.values[keep]
    # STEP 3: rectangular square-root gauge factors, Rhi*Rh = Id_k
    Rh  = V * Diagonal(sqrt.(vals))        # chi x k
    Rhi = Diagonal(1 ./ sqrt.(vals)) * V'  # k x chi
    d = size(M,2); k = length(keep)
    Mt = zeros(k, d, k)
    # STEP 4: gauge + 1/sqrt(nu) rescale => sum_p Mt_p Mt_p' = Id (up to trimmed tail)
    for p in 1:d; Mt[:,p,:] = Rhi * (@view M[:,p,:]) * Rh ./ sqrt(nu); end
    # STEP 5: certify right-canonicality; return the error, do not trust it
    S = zeros(k, k)
    for p in 1:d; S .+= (@view Mt[:,p,:]) * (@view Mt[:,p,:])'; end
    (Mt, norm(S - I), k)
end


# ---------------------------------------------------------------------------
# doubled_col: the THICKENED arm column (one bulk tensor absorbed, nothing
# compressed yet) -- the "before" side of BOTH walls E' and E.
#
# in : M[chi,2,chi]  compressed arm (left virtual l, bare middle u, right r)
#      a[u,l,d,r]    bulk tensor (fully leg-symmetric, entrywise > 0)
# out: D[chi*d,d,chi*d]  thick column; virtual bond chi*d (= 2chi for the
#      Ising d=2), NEW bare leg p left OPEN
#
# STEP 1 -- GROW: hang one bulk tensor under the arm; the arm's bare leg u
#   contracts a's top leg, a's side legs (mu,rho) join the virtual bond:
#
#   --(M)--        D[(l,mu), p, (r,rho)] = sum_u M[l,u,r] a[u,mu,p,rho]
#      |u
#   --[a]--        thick '=' (2chi) bond on both sides; the bare leg p is
#      |p          NOT touched (twist-before-compress).
#
# STEP 2 -- PACK: block index (mu-1)*chi + l (mu slow, l fast); same on the
#   right, (rho-1)*chi + r.  This is the SAME packing as arm_step's grown B
#   and the walls' thick slots (Ep[g,(x-1)*chi+al], Er[(y-1)*chi+be,gp]) --
#   never change one alone.
#
# Unlike arm_step, NOTHING is compressed here: D is exactly the object the
# walls must intertwine away (E' D_p = kap Mt_p E'  and  D_p E = kap E Mt_p).
# ---------------------------------------------------------------------------
"D[(l,mu),p,(r,rho)] = sum_u M[l,u,r] a[u,mu,p,rho]; block index (mu-1)*chi+l."
function doubled_col(M, a)
    chi = size(M,1); d = size(a,1)
    # STEP 1: thick column, virtual bond chi -> chi*d, bare leg p stays open
    D = zeros(chi*d, d, chi*d)
    # STEP 2: contract arm bare leg u into a's top; pack (mu-1)*chi+l (mu slow, l fast)
    for l in 1:chi, r in 1:chi, u in 1:d, mu in 1:d, p in 1:d, rho in 1:d
        w = M[l,u,r] * a[u,mu,p,rho]
        w == 0 && continue
        D[(mu-1)*chi + l, p, (rho-1)*chi + r] += w
    end
    D
end


# ---------------------------------------------------------------------------
# channel_lambda: leading eigenvalue of a 3-leg tensor's transfer channel
# (power method on X <- sum_p T_p X T_p').
#
# in : T3[n,d,n]  any 3-leg row tensor (arm M, thick column D, ...)
# out: lam        leading channel eigenvalue; sets kappa = sqrt(lambda_D)
#                 for both walls (also the fidelity-gate scale)
#
# STEP 1 -- INIT: X = Id/||Id||, an SPD seed for the power method.
# STEP 2 -- POWER: iterate the completely positive channel
#
#   --(T3)--+              +--
#       |p  X    =   lam   X          X <- sum_p T3_p X T3_p'
#   --(T3)--+              +--
#
#   symmetrize (X+X')/2 each sweep (kills numerical asymmetry drift), set
#   lam = ||X|| and renormalize; stop when stationary to 1e-15 (<=6000 its).
# STEP 3 -- RETURN lam only; the fixed-point matrix itself is discarded
#   (expected_floor11 re-derives the D-density spectrum when it needs it).
# ---------------------------------------------------------------------------
"channel eigenvalue of a 3-leg tensor row (for fidelity gate)."
function channel_lambda(T3)
    n = size(T3,1); d = size(T3,2)
    # STEP 1: SPD seed X = Id/||Id||
    X = Matrix{Float64}(I, n, n); X ./= norm(X); lam = 0.0
    for it in 1:6000
        Xn = zeros(n, n)
        # STEP 2: apply the channel  X <- sum_p T_p X T_p'
        for p in 1:d; Xn .+= (@view T3[:,p,:]) * X * (@view T3[:,p,:])'; end
        Xn = (Xn + Xn')/2
        # STEP 2b: lam <- ||Xn||, renormalize; lam converges to the channel eigenvalue
        lam = norm(Xn); Xn ./= lam
        cv = norm(Xn - X); X = Xn
        cv < 1e-15 && break
    end
    lam
end


# ---------------------------------------------------------------------------
# intertwiner_left: the LEFT wall E' (compressed arm on its LEFT, thick
# column on its RIGHT).  Used for the HORIZONTAL arm of the corner: this is
# the truncation-flow direction (apply-then-compress = solve_arm's own
# update).
#
# in : Mt[chi,d,chi]  canonicalized arm (chi = chi_eff after trim)
#      a[u,l,d,r]     bulk tensor
# out: (Ep, kap, res, D)  Ep[chi, chi*d] wall, kap = sqrt(lambda_D),
#      res = relative wall residual, D = thick column (returned for reuse)
#
# STEP 1 -- THICK COLUMN: D = doubled_col(Mt, a); packing (mu-1)*chi + l
#   (mu slow, l fast) fixes the thick-slot convention below.
# STEP 2 -- SCALE: kap = sqrt(channel_lambda(D)).  channel_lambda is the
#   DOUBLE-layer (bra x ket) eigenvalue; the wall equation is single-layer,
#   hence the square root.  Without kap the equation has no nontrivial
#   solution (thick and compressed channels grow at different rates).
# STEP 3 -- STACK: the wall equation, one block per bare component p,
#
#   +----+ ||                       +----+
#   | E' |=D_p=   =  kappa --(Mt_p)-| E' |=       E' D_p = kap Mt_p E'
#   +----+ |p                   |p  +----+
#
#   vectorized column-major on vec(E') (E' is chi x chi*d):
#     E' D_p  -> (D_p^T ox I_chi)   vec
#     Mt_p E' -> (I_chid ox Mt_p)   vec
#   so  Lp = kron(D_p', Ic) - kap * kron(Icd, Mt_p), all p stacked into L.
# STEP 4 -- SOLVE: dense SVD of L, take the SMALLEST right singular vector
#   F.V[:, end], reshape to Ep[chi, chi*d].  Why dense SVD: degeneracy-
#   proof -- the power method stalls on a 1e-8-split top space.
# STEP 5 -- CERTIFY: relative residual over the stacked equations,
#   res = sqrt( sum_p ||E'D_p - kap Mt_p E'||^2 / sum_p ||kap Mt_p E'||^2 ).
#   Judge it against expected_floor11: a wall is "as good as possible"
#   when res ~ floor (measured 0.63x), NOT when res ~ 0 -- the spectral
#   tail beyond chi is an honest, irreducible budget.
# ---------------------------------------------------------------------------
"LEFT wall E' (chi x chi*d): E' D_p = kappa M_p E' -- the truncation-flow
 direction (apply-then-compress, i.e. solve_arm's own update).  Direct SVD.
 vec: E'D_p -> (D_p^T ox I_chi) vec;  M_p E' -> (I_chid ox M_p) vec."
function intertwiner_left(Mt, a)
    D = doubled_col(Mt, a)
    chi = size(Mt,1); d = size(Mt,2); chid = chi*d
    # STEP 2: kap = sqrt(lambda_D) -- single-layer scale of the double-layer eigenvalue
    lamD = channel_lambda(D)
    kap = sqrt(lamD)
    n = chi*chid
    L = zeros(d*n, n)
    Ic = Matrix{Float64}(I, chi, chi); Icd = Matrix{Float64}(I, chid, chid)
    for p in 1:d
        # STEP 3: Lp = kron(D_p',Ic) - kap*kron(Icd,Mt_p), one block per p
        Lp = kron((@view D[:,p,:])', Ic) .- kap .* kron(Icd, @view Mt[:,p,:])
        L[(p-1)*n .+ (1:n), :] .= Lp
    end
    F = svd(L)
    # STEP 4: smallest right singular vector of the dense SVD (degeneracy-proof)
    Ep = reshape(F.V[:, end], chi, chid)
    num = 0.0; den = 0.0
    for p in 1:d
        # STEP 5: relative residual of E' D_p = kap Mt_p E' (compare to expected_floor11)
        LHS = Ep * (@view D[:,p,:])
        RHS = kap .* ((@view Mt[:,p,:]) * Ep)
        num += norm(LHS - RHS)^2; den += norm(RHS)^2
    end
    (Ep, kap, sqrt(num/den), D)
end


# ---------------------------------------------------------------------------
# intertwiner_right: the RIGHT wall E (thick column on its LEFT, compressed
# arm on its RIGHT).  Used for the VERTICAL arm (chain order AFTER the
# corner).
#
# in : Mt[chi,d,chi]  canonicalized arm
#      a[u,l,d,r]     bulk tensor
# out: (E, kap, res)  E[chi*d, chi] wall, kap = sqrt(lambda_D),
#      res = relative wall residual
#
# STEP 1 -- THICK COLUMN: D = doubled_col(Mt, a), same packing as the left
#   wall ((mu-1)*chi + l, mu slow, l fast).
# STEP 2 -- SCALE: kap = sqrt(channel_lambda(D)), exactly as for E'.
# STEP 3 -- STACK: the mirrored wall equation, one block per p,
#
#    ||  +---+                +---+
#   =D_p=| E |  =  kappa =====| E |--(Mt_p)--      D_p E = kap E Mt_p
#    |p  +---+                +---+  |p
#
#   vectorized column-major on vec(E) (E is chi*d x chi):
#     D_p E  -> (I_chi ox D_p)     vec
#     E Mt_p -> (Mt_p^T ox I_chid) vec
#   so  Lp = kron(Ic, D_p) - kap * kron(Mt_p', Icd), all p stacked into L.
# STEP 4 -- SOLVE: dense SVD of L, smallest right singular vector, reshape
#   to E[chi*d, chi] (degeneracy-proof, same reason as the left wall).
# STEP 5 -- CERTIFY: res = sqrt( sum_p ||D_p E - kap E Mt_p||^2
#                                / sum_p ||E Mt_p||^2 ).
#
# WHY A SEPARATE SOLVE: E != E'^T -- the canonical gauge is NOT reflection
# covariant.  Using E'^T in place of E was the v8 wall bug (cross gates
# 0.33 -> 3e-3 once E was solved in its own right).
# ---------------------------------------------------------------------------
"right wall E (chi*d x chi): D_p E = kap E Mt_p.  Direct stacked SVD.
 vec: D_p E -> (I_chi ox D_p) vec;  E Mt_p -> (Mt_p^T ox I_chid) vec."
function intertwiner_right(Mt, a)
    D = doubled_col(Mt, a)
    chi = size(Mt,1); d = size(Mt,2); chid = chi*d
    # STEP 2: kap = sqrt(lambda_D), as for E'
    kap = sqrt(channel_lambda(D))
    n = chid*chi
    L = zeros(d*n, n)
    Ic = Matrix{Float64}(I, chi, chi); Icd = Matrix{Float64}(I, chid, chid)
    for p in 1:d
        # STEP 3: Lp = kron(Ic,D_p) - kap*kron(Mt_p',Icd), one block per p
        Lp = kron(Ic, @view D[:,p,:]) .- kap .* kron((@view Mt[:,p,:])', Icd)
        L[(p-1)*n .+ (1:n), :] .= Lp
    end
    F = svd(L)
    # STEP 4: smallest right singular vector (degeneracy-proof; E != E'^T, v8 bug)
    E = reshape(F.V[:, end], chid, chi)
    num = 0.0; den = 0.0
    for p in 1:d
        # STEP 5: relative residual of D_p E = kap E Mt_p
        num += norm((@view D[:,p,:]) * E .- kap .* (E * (@view Mt[:,p,:])))^2
        den += norm(E * (@view Mt[:,p,:]))^2
    end
    (E, kap, sqrt(num/den))
end

# ---------------------------------------------------------------------------
# expected_floor11: HONEST precision floor of the walls (spectral-tail
# budget).  A wall maps the chi*d-dim thick channel onto chi compressed
# directions; whatever density weight lives beyond chi can never be
# intertwined away, and truncation enters the wall residual as AMPLITUDES
# ~ sqrt(weight):
#     floor = sqrt( D-channel-density spectrum beyond chi / total ).
#
# in : Mt[chi,d,chi], a[u,l,d,r]
# out: floor (scalar) -- the residual a PERFECT wall would still show
#
# STEP 1 -- THICK COLUMN: D = doubled_col(Mt, a).
# STEP 2 -- DENSITY: power-iterate the D-channel fixed point
#
#   --(D)--+            +--
#      |p  X    ~   nu  X        X <- sum_p D_p X D_p'
#   --(D)--+            +--
#
#   (symmetrize + normalize each sweep; <=6000 its, tol 1e-15).
# STEP 3 -- SPECTRUM: eigenvalues of X, |.|-sorted descending.
# STEP 4 -- BUDGET: floor = sqrt( max(sum ev[chi+1:end], 0) / sum ev ).
#
# WHY: this is the drivers' yardstick -- a wall is "as good as possible"
# when residual ~ floor (measured 0.63x floor); a residual stuck far above
# it (e.g. the 2.5e-3 plateau at beta=0.05, chi=4) signals junk directions
# to trim in canonicalize, not a solver failure.
# ---------------------------------------------------------------------------
"residual floor from the D-density spectrum tail (amplitudes ~ sqrt(weight))."
function expected_floor11(Mt, a)
    D = doubled_col(Mt, a)
    n = size(D,1); d = size(D,2)
    X = Matrix{Float64}(I, n, n); X ./= norm(X)
    for it in 1:6000
        Xn = zeros(n, n)
        # STEP 2: D-channel density fixed point  X <- sum_p D_p X D_p'
        for p in 1:d; Xn .+= (@view D[:,p,:]) * X * (@view D[:,p,:])'; end
        Xn = (Xn + Xn')/2; Xn ./= norm(Xn)
        cv = norm(Xn - X); X = Xn; cv < 1e-15 && break
    end
    # STEP 3: density spectrum, descending
    ev = sort(abs.(eigen(Symmetric((X+X')/2)).values), rev=true)
    chi = size(Mt,1)
    # STEP 4: floor = sqrt(tail beyond chi / total) -- amplitudes ~ sqrt(weight)
    sqrt(max(sum(ev[chi+1:end]), 0.0) / sum(ev))
end

# ---------------------------------------------------------------------------
# growU: absorb ONE row into the UP channel cap (cap advances one row DOWN
# toward the centre).
#
# in : F[chi,d,chi]  up cap, legs F_U[l,x,r]: l = in (from UL half-arm),
#                    x = bulk bond, r = out (to UR half-arm)
#      M[chi,2,chi]  arm tensor; op[u,l,d,r] row tensor (a or am)
# out: Fn[chi,d,chi] grown cap, same leg order; new legs one row BELOW the
#                    old ones; NOT normalized
#
# STEP 1 -- LOOP: run over the OLD cap entries F[lp,xp,rp], skipping zeros.
# STEP 2 -- SANDWICH: the arms flank op horizontally, one row down:
#
#         [F_U]
#     lp    |xp    rp        old cap legs
#    (M)-pw[op]pe-(M)     Fn[l,x,r] += M[l,pw,lp] * op[xp,pw,x,pe]
#     |l    |x     |r                  * M[rp,pe,r] * F[lp,xp,rp]
#
#   op slots (u,l,d,r): u = xp -> old bond (cap side), l = pw -> west bare
#   leg into the UL arm, d = x -> NEW open bond (toward the centre),
#   r = pe -> east bare leg into the UR arm.
#
# WHY: one cyclic sense (leg-order header above): UL flank l(new) ->
# lp(old) upward, UR flank rp(old) -> r(new) downward; cross_val uses
# FU[:,ux,:] as [ul,ur] directly.
# ---------------------------------------------------------------------------
function growL(F, M, op)
    chi = size(M,1); d = size(op,1)
    Fn = zeros(chi, d, chi)
    # STEP 1: run over OLD cap entries F[t,x,b] (u,dn = sandwich bare legs)
    @inbounds for t in 1:chi, x in 1:d, b in 1:chi, u in 1:d, dn in 1:d
        f = F[t,x,b]; f == 0 && continue
        for tp in 1:chi, xp in 1:d, bp in 1:chi
            # STEP 2: sandwich M(top)/op/M(bottom); op's r-leg xp opens toward the centre
            Fn[tp,xp,bp] += f * M[t,u,tp] * op[u,x,dn,xp] * M[bp,dn,b]
        end
    end
    Fn
end
function growR(F, M, op)
    chi = size(M,1); d = size(op,1)
    Fn = zeros(chi, d, chi)
    # STEP 1: run over OLD cap entries F[tp,xp,bp], skipping zeros
    @inbounds for tp in 1:chi, xp in 1:d, bp in 1:chi
        f = F[tp,xp,bp]; f == 0 && continue
        for t in 1:chi, x in 1:d, b in 1:chi, u in 1:d, dn in 1:d
            # STEP 2: sandwich; op's r-leg xp eats the old bond, l-leg x opens toward the centre
            Fn[t,x,b] += M[t,u,tp] * op[u,x,dn,xp] * M[bp,dn,b] * f
        end
    end
    Fn
end
function growU(F, M, op)
    chi = size(M,1); d = size(op,1)
    Fn = zeros(chi, d, chi)
    # STEP 1: run over OLD cap entries F[lp,xp,rp], skipping zeros
    @inbounds for lp in 1:chi, xp in 1:d, rp in 1:chi
        f = F[lp,xp,rp]; f == 0 && continue
        for l in 1:chi, x in 1:d, r in 1:chi, pw in 1:d, pe in 1:d
            # STEP 2: sandwich M(left)/op/M(right); op's u-leg eats the old bond xp, d-leg x opens
            Fn[l,x,r] += M[l,pw,lp] * op[xp,pw,x,pe] * M[rp,pe,r] * f
        end
    end
    Fn
end
# ---------------------------------------------------------------------------
# growD: absorb ONE row into the DOWN channel cap (cap advances one row UP
# toward the centre).
#
# in : F[chi,d,chi]  down cap, legs F_D[r,x,l]: r = in (from DR half-arm),
#                    x = bulk bond, l = out (to DL half-arm) -- NOTE the
#                    REVERSED [r,x,l] order vs F_U, dictated by the one
#                    cyclic sense
#      M[chi,2,chi]  arm tensor; op[u,l,d,r] row tensor (a or am)
# out: Fn[chi,d,chi] grown cap, same leg order; new legs one row ABOVE the
#                    old ones; NOT normalized
#
# STEP 1 -- LOOP: run over the OLD cap entries F[rp,xp,lp], skipping zeros.
# STEP 2 -- SANDWICH: the arms flank op horizontally, one row up (drawn
#   geometrically, DL arm west / DR arm east; STORED order is [r,x,l]):
#
#     |l    |x     |r        new cap legs (centre side)
#    (M)-pw[op]pe-(M)     Fn[r,x,l] += M[r,pe,rp] * op[x,pw,xp,pe]
#     lp    |xp    rp                  * M[lp,pw,l] * F[rp,xp,lp]
#         [F_D]
#
#   op slots (u,l,d,r): u = x -> NEW open bond (toward the centre),
#   l = pw -> west bare leg into the DL arm, d = xp -> old bond (cap
#   side), r = pe -> east bare leg into the DR arm.
#
# WHY: one cyclic sense (leg-order header above): DR flank r(new) ->
# rp(old) downward, DL flank lp(old) -> l(new) upward.  The stored [r,x,l]
# order means cross_val uses FD[:,dx,:] as [dr,dl] directly -- the cycle
# closes without extra transposes.
# ---------------------------------------------------------------------------
function growD(F, M, op)
    chi = size(M,1); d = size(op,1)
    Fn = zeros(chi, d, chi)
    # STEP 1: run over OLD cap entries F[rp,xp,lp] (stored [r,x,l], reversed order)
    @inbounds for rp in 1:chi, xp in 1:d, lp in 1:chi
        f = F[rp,xp,lp]; f == 0 && continue
        for r in 1:chi, x in 1:d, l in 1:chi, pw in 1:d, pe in 1:d
            # STEP 2: sandwich; op's d-leg eats the old bond xp, u-leg x opens upward
            Fn[r,x,l] += M[r,pe,rp] * op[x,pw,xp,pe] * M[lp,pw,l] * f
        end
    end
    Fn
end

# ---------------------------------------------------------------------------
# cap_fp: channel cap = fixed point of its OWN growth map (the three-leg
# rectangle of the paper):  grow(F) = lambda_col * F.
#
# in : grower       one of growL/growR/growU/growD (fixes the orientation)
#      M[chi,2,chi] arm tensor;  a[u,l,d,r] bulk tensor
#      iters/tol    power-iteration budget (30000 / 1e-14)
# out: (F, lam)     normalized cap + per-column channel eigenvalue lam
#
# STEP 1 -- INIT: F = ones(chi,d,chi)/||.||, a strictly positive seed --
#   safe because the bulk tensor is entrywise > 0 (Perron-Frobenius: the
#   leading cap has uniform sign, the positive seed overlaps it).
# STEP 2 -- POWER: Fn = grower(F, M, a); lam = ||Fn||; Fn /= lam:
#
#   [F]--(M)--             [F]--
#    |    |                 |
#   [F]--[a]--  =  lam     [F]--      (growL orientation shown; R/U/D
#    |    |                 |          approach the centre from the other
#   [F]--(M)--             [F]--      three directions)
#
# STEP 3 -- SIGN-FIX: align sign(sum Fn.*F) BEFORE the convergence test,
#   so a global -1 flip between sweeps cannot masquerade as non-convergence.
# STEP 4 -- STOP when ||Fn - F|| < tol; return (F, lam).
#
# WHY: the cap absorbs the semi-infinite channel EXACTLY -- there is NO
# isometry assumption anywhere in the cross readout (contrast the REFUTED
# v3 closed-ring corner, which leans on the arm isometry U and fails the
# cross gates at O(1)).
# ---------------------------------------------------------------------------
"cap = fixed point of its own growth map (absorb columns on the centre side)."
function cap_fp(grower, M, a; iters=30000, tol=1e-14)
    chi = size(M,1); d = size(a,1)
    # STEP 1: strictly positive seed (Perron-Frobenius channel)
    F = ones(chi, d, chi); F ./= norm(F); lam = 0.0
    for it in 1:iters
        # STEP 2: absorb one column/row on the centre side
        Fn = grower(F, M, a)
        # STEP 2b: lam = per-column channel eigenvalue
        lam = norm(Fn); Fn ./= lam
        # STEP 3: sign-fix before the convergence test
        sg = sign(sum(Fn .* F)); sg == 0 && (sg = 1.0); Fn .*= sg
        cv = norm(Fn - F); F = Fn
        cv < tol && break
    end
    (F, lam)
end

# ---------------------------------------------------------------------------
# cross_val: the cross readout (paper SM p.8, single-layer version) -- one
# scalar from 4 caps + 4 corner diamonds + the centre tensor.
#
# in : FU[l,x,r], FR[t,x,b], FD[r,x,l], FL[t,x,b]  channel caps (fixed
#      points of growU/R/D/L; leg orders from the cap header block above
#      growL: t/b = arm bonds, x = bulk bond)
#      cTL,cTR,cBR,cBL [chi,chi]  corner matrices (one transpose pattern)
#      op[u,l,d,r]                centre bulk tensor (a or am)
# out: N (scalar).  Only RATIOS of same-geometry crosses are physical --
#      they cancel every cap norm and channel eigenvalue.
#
#             [F_U]
#         <cTL>   <cTR>
#   [F_L]------[op]------[F_R]     N = sum_x op(ux,lx,dx,rx) *
#         <cBL>   <cBR>                tr[cTL FU(ux) cTR FR(rx) cBR FD(dx)
#             [F_D]                       cBL FL(lx)]
#
# STEP 1 -- SWEEP the centre tensor's four bare legs (ux,lx,dx,rx); zero
#   entries of op are skipped (the bulk tensor is entrywise sparse).
# STEP 2 -- SLICE each cap at its bulk leg to a chi x chi matrix:
#     Um = FU[:,ux,:] = [ul,ur]      Rm = FR[:,rx,:] = [rt,rb]
#     Dm = FD[:,dx,:] = [dr,dl]      Lm = FL[:,lx,:]' = [lb,lt]
#   Only FL needs a transpose: it is STORED [lt,x,lb] (top-arm bond first)
#   but the trace cycle enters it from the bottom -- all four half-arm
#   chains run in ONE cyclic sense (same sense as the v3 ring, see the
#   cap header block).
# STEP 3 -- TRACE the 8-factor cycle TL->U->TR->R->BR->D->BL->L and
#   accumulate with weight op[ux,lx,dx,rx].
# ---------------------------------------------------------------------------
"cross value: cycle tr[cTL*Umat*cTR*Rmat*cBR*Dmat*cBL*Lmat] weighted by op."
function cross_val(FU, FR, FD, FL, cTL, cTR, cBR, cBL, op)
    d = size(op,1)
    val = 0.0
    # STEP 1: sweep the centre tensor's 4 bare legs, skip zero op entries
    for ux in 1:d, lx in 1:d, dx in 1:d, rx in 1:d
        w = op[ux,lx,dx,rx]; w == 0 && continue
        # STEP 2: slice each cap at its bulk leg; only FL is transposed (stored [lt,x,lb])
        Um = @view FU[:,ux,:]              # [ul,ur]
        Rm = @view FR[:,rx,:]              # [rt,rb]
        Dm = @view FD[:,dx,:]              # [dr,dl]
        Lm = (@view FL[:,lx,:])'           # [lb,lt] (stored [lt,x,lb])
        # STEP 3: 8-factor trace cycle TL->U->TR->R->BR->D->BL->L
        val += w * tr(cTL * Um * cTR * Rm * cBR * Dm * cBL * Lm)
    end
    val
end

# ---------------------------------------------------------------------------
# corner_pattern: the 4 transpose assignments of one corner c to the four
# diamond slots (cTL,cTR,cBR,cBL).  Geometrically legal: pat1 (all c) and
# pat4 (all c^T -- the global transpose freedom of the readout).  Mixed
# pat2/pat3 are ILLEGAL and kept only as negative controls.  NOTE (review
# 2026-08-10): the old claim "2/3 never win the gate scan" FAILED at
# criticality (crit chi=5 pat2, chi=14 pat3 were selected) -- pick_corner
# now restricts its scan to (1, 4); 2/3 remain available for explicit
# negative-control calls only.
# ---------------------------------------------------------------------------
"the four transpose patterns for (cTL,cTR,cBR,cBL)."
corner_pattern(c, pat) = pat == 1 ? (c,c,c,c) :
                         pat == 2 ? (c,Matrix(c'),c,Matrix(c')) :
                         pat == 3 ? (Matrix(c'),c,Matrix(c'),c) :
                                    (Matrix(c'),Matrix(c'),Matrix(c'),Matrix(c'))

# ---------------------------------------------------------------------------
# cross_gates: the corner certification instrument -- EXACT-ANSWER gates.
# The physical corner branch is the one that reproduces exact correlators
# through an environment-consistent readout; |lam| is NEVER used for the
# selection (the refuted v3 ring corner passes its own self-consistency
# yet fails these gates at O(1): X1=0.16, X2=0.53).
#
# in : M, a          canonical arm + bulk tensor
#      am            spin-inserted bulk tensor (numerator insertions)
#      FU,FR,FD,FL   channel caps (fixed points from cap_fp)
#      c, pat        corner candidate + transpose pattern (corner_pattern)
#      ss_exact      Onsager NN <s s'> (onsager_ss)
#      diag_bench    corner-free diagonal correlator (diag_benchmark)
# out: (e1, e2, spread, x1R)
#
# STEP 1 -- PATTERN: expand c into (cTL,cTR,cBR,cBL) via corner_pattern.
# STEP 2 -- PRE-GROW: every cap absorbs one extra column/row, once plain
#   (F*a = growX(F,M,a)) and once spin-inserted (F*m = growX(F,M,am)).
#   Numerator and denominator of every gate then have IDENTICAL geometry,
#   so all cap norms / channel eigenvalues cancel in the ratio.
#
#                [FU]                        [FU]
#            <c>  |   <c>                <c>  |   <c>
#     [FL]------[am]-----[FRm]   /   [FL]------[a]------[FRa]   =  x1R
#            <c>  |   <c>                <c>  |   <c>
#                [FD]                        [FD]
#
# STEP 3 -- X1 (R read): centre=am + one am-column in the R channel over
#   the same all-plain cross  ->  <s00 s10>.  Guard: |den_R| < 1e-300
#   returns (Inf, Inf, Inf, NaN) so the branch scan simply discards it.
# STEP 4 -- MIRROR: the SAME NN correlator read through the L, U and D
#   channels (x1L, x1U, x1D).  A true corner makes the readout
#   direction-blind; the spread measures any residual anisotropy.
# STEP 5 -- X2 (diagonal): am-column in R + am-column in U, centre plain
#   -> <s10 s01>, checked against the corner-free diag_benchmark.
# STEP 6 -- SCORES: e1 = |x1R - ss_exact|/|ss_exact|, e2 likewise vs
#   diag_bench, spread = (max - min of the four X1 reads)/|ss_exact|.
# ---------------------------------------------------------------------------
"X gates for one corner c and one pattern; returns (e1,e2,spread, X1)."
function cross_gates(M, a, am, FU, FR, FD, FL, c, pat, ss_exact, diag_bench)
    cTL,cTR,cBR,cBL = corner_pattern(c, pat)
    # STEP 2: pre-grow every cap once plain (a) and once spin-inserted (am)
    FRa = growR(FR, M, a);  FRm = growR(FR, M, am)
    FLa = growL(FL, M, a);  FLm = growL(FL, M, am)
    FUa = growU(FU, M, a);  FUm = growU(FU, M, am)
    FDa = growD(FD, M, a);  FDm = growD(FD, M, am)
    # STEP 3: X1 through the R channel -- same-geometry ratio, guarded denominator
    den_R = cross_val(FU, FRa, FD, FL, cTL,cTR,cBR,cBL, a)
    abs(den_R) < 1e-300 && return (Inf, Inf, Inf, NaN)
    x1R = cross_val(FU, FRm, FD, FL, cTL,cTR,cBR,cBL, am) / den_R
    # STEP 4: mirror reads of X1 through the L / U / D channels
    x1L = cross_val(FU, FR, FD, growL(FL,M,am), cTL,cTR,cBR,cBL, am) /
          cross_val(FU, FR, FD, FLa,            cTL,cTR,cBR,cBL, a)
    x1U = cross_val(FUm, FR, FD, FL, cTL,cTR,cBR,cBL, am) /
          cross_val(FUa, FR, FD, FL, cTL,cTR,cBR,cBL, a)
    x1D = cross_val(FU, FR, FDm, FL, cTL,cTR,cBR,cBL, am) /
          cross_val(FU, FR, FDa, FL, cTL,cTR,cBR,cBL, a)
    # STEP 5: X2 diagonal gate (am in R and in U, plain centre)
    x2  = cross_val(FUm, FRm, FD, FL, cTL,cTR,cBR,cBL, a) /
          cross_val(FUa, FRa, FD, FL, cTL,cTR,cBR,cBL, a)
    # STEP 6: relative errors + readout spread
    e1 = abs(x1R - ss_exact) / abs(ss_exact)
    e2 = abs(x2 - diag_bench) / abs(diag_bench)
    xs = (x1R, x1L, x1U, x1D)
    spread = (maximum(xs) - minimum(xs)) / abs(ss_exact)
    (e1, e2, spread, x1R)
end

# ---------------------------------------------------------------------------
# strip_nn: corner-FREE one-row strip readout -- certifies arm + caps +
# wiring ALONE (gate XA vs Onsager), before any corner enters the game.
#
# in : M, a, am, FL, FR   (caps = fixed points of growL / growR)
# out: <s_0 s_1>          NN correlator on the strip row
#
#   [F_L]--col(am)--col(am)--[F_R]
#   -------------------------------  =  <s_0 s_1>   ->  onsager_ss exact
#   [F_L]--col(a)---col(a)---[F_R]
#
# STEP 1 -- NUM: grow FL by two spin-inserted columns, close on FR by the
#   full elementwise overlap sum(... .* FR) over the cap legs [t,x,b].
# STEP 2 -- DEN: identical geometry with two plain columns.
# STEP 3 -- RATIO: cancels lambda_col^2 and both cap norms exactly
#   (same-geometry principle used by every gate in this file).
# ---------------------------------------------------------------------------
"one-row strip NN <s s> via caps only (certifies arm + caps + wiring)."
function strip_nn(M, a, am, FL, FR)
    # STEP 1: numerator -- two am columns, closed on FR
    num = sum(growL(growL(FL, M, am), M, am) .* FR)
    # STEP 2-3: same-geometry plain denominator; the ratio cancels all norms
    den = sum(growL(growL(FL, M, a),  M, a)  .* FR)
    num / den
end
# ---------------------------------------------------------------------------
# grow2R: absorb ONE two-tensor column into the RIGHT two-row strip cap --
# the mirror of grow2L, approaching the centre from the right.
#
# in : F[tp, x1p, x2p, bp]  right cap;  M arm;  a1 upper / a2 lower tensor
# out: Fn[t, x1, x2, b]     cap with the new column absorbed on its
#      centre-facing (left) side -- the new open legs are the unprimed ones
#
#      t --(M)-- tp
#      |    |u               Fn[t,x1,x2,b] += M[t,u,tp] a1[u,x1,m,x1p]
#     x1 --[a1]-- x1p           a2[m,x2,dn,x2p] M[bp,dn,b] *
#      |    |m                  F[tp,x1p,x2p,bp]
#     x2 --[a2]-- x2p
#      |    |dn              same wiring as grow2L, but the OLD cap sits
#      b --(M)-- bp          on the primed (right) legs and the new open
#                            legs are the unprimed (left) ones.
#
# STEP 1 -- loop the old cap entries F[tp,x1p,x2p,bp], skipping zeros.
# STEP 2 -- attach the column on the cap's centre side; the arms keep the
#   one cyclic sense (top L->R, bottom reversed), so grown-left and
#   right caps close by a plain elementwise overlap.
# ---------------------------------------------------------------------------
"two-row strip caps and diagonal benchmark <s(0,up) s(1,low)>."
function grow2L(F, M, a1, a2)
    chi = size(M,1); d = size(a1,1)
    Fn = zeros(chi, d, d, chi)
    # STEP 1: loop old cap entries, skip zeros
    @inbounds for t in 1:chi, x1 in 1:d, x2 in 1:d, b in 1:chi
        f = F[t,x1,x2,b]; f == 0 && continue
        # STEP 2: attach the a1-over-a2 column; top arm L->R, bottom arm reversed
        for u in 1:d, m in 1:d, dn in 1:d, x1p in 1:d, x2p in 1:d, tp in 1:chi, bp in 1:chi
            Fn[tp,x1p,x2p,bp] += f * M[t,u,tp] * a1[u,x1,m,x1p] * a2[m,x2,dn,x2p] * M[bp,dn,b]
        end
    end
    Fn
end
function grow2R(F, M, a1, a2)
    chi = size(M,1); d = size(a1,1)
    Fn = zeros(chi, d, d, chi)
    @inbounds for tp in 1:chi, x1p in 1:d, x2p in 1:d, bp in 1:chi
        f = F[tp,x1p,x2p,bp]; f == 0 && continue
        # STEP 2: attach the column on the centre side, same cyclic sense as grow2L
        for t in 1:chi, x1 in 1:d, x2 in 1:d, b in 1:chi, u in 1:d, m in 1:d, dn in 1:d
            Fn[t,x1,x2,b] += M[t,u,tp] * a1[u,x1,m,x1p] * a2[m,x2,dn,x2p] * M[bp,dn,b] * f
        end
    end
    Fn
end
# ---------------------------------------------------------------------------
# diag_benchmark: corner-free DIAGONAL correlator via the two-row strip --
# the reference value for gate X2.  Agrees with the closed form
# (2/pi)[E(k)-(1-k^2)K(k)]/k, k=sinh^2(2beta), to 8 digits.
#
# in : M, a, am (+ iters cap)      out: <s(0,1) s(1,0)>
#
#   [F2L]--col(am@up)--col(am@low)--[F2R]
#   --------------------------------------  =  <s(0,up) s(1,low)>
#   [F2L]--col(a,a)----col(a,a)-----[F2R]
#
# STEP 1 -- INIT both two-row caps to normalized all-ones F[chi,d,d,chi]
#   (cap legs [t, x_upper, x_lower, b], see grow2L).
# STEP 2 -- LEFT CAP: power-iterate F2L <- grow2L(F2L,M,a,a), normalize,
#   sign-fix against the previous iterate (power iteration is sign-blind),
#   stop at 1e-14.
# STEP 3 -- RIGHT CAP: the same fixed-point loop with grow2R.
# STEP 4 -- NUM: insert am in the UPPER row of column 0 (inner grow2L
#   call, a1=am) and in the LOWER row of column 1 (outer call, a2=am),
#   then close on F2R by the elementwise overlap -> the diagonal pair.
# STEP 5 -- DEN + RATIO: identical all-plain geometry; the ratio cancels
#   both cap norms and lambda_col^2 (same-geometry principle, strip_nn).
# ---------------------------------------------------------------------------
function diag_benchmark(M, a, am; iters=30000)
    chi = size(M,1); d = size(a,1)
    # STEP 1: init both two-row caps to normalized ones
    F2L = ones(chi,d,d,chi); F2L ./= norm(F2L)
    F2R = ones(chi,d,d,chi); F2R ./= norm(F2R)
    for it in 1:iters
        # STEP 2: left-cap fixed point (normalize + sign-fix each sweep)
        Fn = grow2L(F2L, M, a, a); Fn ./= norm(Fn)
        sg = sign(sum(Fn .* F2L)); sg == 0 && (sg=1.0); Fn .*= sg
        cv = norm(Fn - F2L); F2L = Fn; cv < 1e-14 && break
    end
    for it in 1:iters
        # STEP 3: right-cap fixed point
        Fn = grow2R(F2R, M, a, a); Fn ./= norm(Fn)
        sg = sign(sum(Fn .* F2R)); sg == 0 && (sg=1.0); Fn .*= sg
        cv = norm(Fn - F2R); F2R = Fn; cv < 1e-14 && break
    end
    # STEP 4-5: am at (col0,upper) + (col1,lower), over the plain same-geometry ratio
    num = sum(grow2L(grow2L(F2L, M, am, a), M, a, am) .* F2R)
    den = sum(grow2L(grow2L(F2L, M, a,  a), M, a, a ) .* F2R)
    num / den
end

# ---------------------------------------------------------------------------
# corner_maps11: the USER'S corner equation as explicit matrices, A c = lam B c.
#
# in : Mt[chi,d2,chi]  canonical arm
#      Ep[chi, chi*d2] LEFT wall  (horizontal arm side)
#      Er[chi*d2, chi] RIGHT wall (vertical arm side)
#      a[u,l,d,r]      bulk tensor (the elbow)
# out: (A, B)  both (chi*d2)^2 x chi^2; feed to corner_branches
#
#   [E']---<c>---[E]            (Mh)---<c>---(Mv)
#      \    |   /       = lam     |           |       per open (g,dn,gp,rr)
#       \  [a] /                  dn          rr
#        (dn,rr open)
#
# LHS (A): the walls sit INSIDE the equation and absorb the thick tails;
#   the elbow a keeps its outer legs (dn, rr) OPEN.
# RHS (B): one arm tensor per side with bare legs matching (dn, rr) --
#   B is the M-sandwich = the user's original N.
#
# STEP 1 -- LAYOUT: column index cidx = (be-1)*chi + al over c[al,be];
#   row index ridx = ((rr-1)*chi + gp - 1)*(chi*d2) + (dn-1)*chi + g,
#   i.e. the open pair ((g,dn),(gp,rr)) with bare-slow/virtual-fast packing
#   on each factor -- the doubled_col convention.
# STEP 2 -- LHS entry: contract the elbow with BOTH walls,
#     A[ridx,cidx] = sum_{x,y} Ep[g, (x-1)*chi + al] * a[y,x,dn,rr]
#                              * Er[(y-1)*chi + be, gp]
#   x = a's left leg, packed with c's al into the LEFT wall's thick slot;
#   y = a's top leg, packed with c's be into the RIGHT wall's thick slot.
# STEP 3 -- RHS entry: B[ridx,cidx] = Mt[g,dn,al] * Mt[be,rr,gp]
#   (horizontal arm carries bare leg dn, vertical arm carries rr).
#
# WHY two DIFFERENT walls: E != E'^T (the canonical gauge is not reflection
# covariant) -- mixing them was the v8 wall bug.  The system is deliberately
# OVERDETERMINED (rows (chi*d2)^2 >> cols chi^2); corner_branches reduces it
# and the EXACT-ANSWER cross gates select the physical branch, never |lam|.
# ---------------------------------------------------------------------------
"A c = lam B c with the PROPER walls: Ep (left wall) horizontal, Er (right
 wall) vertical.  Row index ((g,dn),(gp,rr)); col index c[al,be]."
function corner_maps11(Mt, Ep, Er, a)
    chi = size(Mt,1); d2 = size(a,1)
    n = chi*chi
    rows = (chi*d2)*(chi*d2)
    # STEP 1: overdetermined layout, rows (chi*d2)^2, cols chi^2
    A = zeros(rows, n); B = zeros(rows, n)
    for g in 1:chi, dn in 1:d2, gp in 1:chi, rr in 1:d2
        # STEP 1b: row packing ((g,dn),(gp,rr)), bare slow / virtual fast per factor
        ridx = ((rr-1)*chi + gp - 1)*(chi*d2) + (dn-1)*chi + g
        for al in 1:chi, be in 1:chi
            cidx = (be-1)*chi + al
            sA = 0.0
            # STEP 2: LHS -- elbow a contracted with Ep (x with al) and Er (y with be)
            for x in 1:d2, y in 1:d2
                w = a[y, x, dn, rr]
                w == 0 && continue
                sA += Ep[g, (x-1)*chi + al] * w * Er[(y-1)*chi + be, gp]
            end
            A[ridx, cidx] = sA
            # STEP 3: RHS -- the M-sandwich (the user's original N)
            B[ridx, cidx] = Mt[g, dn, al] * Mt[be, rr, gp]
        end
    end
    (A, B)
end

# ---------------------------------------------------------------------------
# corner_branches: ALL branches of the corner equation, hardened.
#
# in : (A, B) from corner_maps11 (rows (chi*d2)^2, cols chi^2)
#      chi    corner dimension (c is chi x chi); nb = max candidates (12)
# out: up to nb tuples (lam_signed, c[chi,chi], fullres), scanned in
#      descending |lam|
#
# STEP 1 -- REDUCE (Petrov-Galerkin): project the overdetermined system to
#   a square pencil,  (B'A) c = lam (B'B) c,  with BB symmetrized.
#   WARNING (review finding): the reduction can create SPURIOUS pairs when
#   BB is ill-conditioned -- which is exactly why every candidate carries
#   the FULL-system residual, and why the physical branch is selected by
#   the cross gates, NEVER by |lam| or the reduced residual.
# STEP 2 -- ORDER: generalized eigen(BA, BB); candidates visited in
#   descending |lam|.
# STEP 3 -- FULL RESIDUAL: fullres(lam, c) = ||A c - lam B c|| / ||lam B c||
#   evaluated on the ORIGINAL stacked system (denominator floored at 1e-300
#   against zero modes).
# STEP 4 -- DEDUP + PUSH (pushc!): normalize c, drop near-zero vectors
#   (norm < 1e-12), drop duplicates (|overlap| > 1 - 1e-10 with any kept
#   c), keep the SIGNED lam (sign matters downstream).
# STEP 5 -- HARVEST: for each eigenpair until nb are kept:
#   - skip nonfinite lam or |lam| >= 1e8 (BB near-null junk);
#   - essentially real vector (max|Im v| <= 1e-8 max|Re v|): keep Re v;
#   - NEAR-REAL complex pair (|Im lam| < 0.1 |lam|): mine BOTH Re v and
#     Im v -- the real 2D invariant subspace may hide the physical corner;
#   - genuinely complex lam: discarded.
# ---------------------------------------------------------------------------
"branches of (B'A)c = lam (B'B)c (Petrov-Galerkin reduction; spurious pairs
 possible when BB is ill-conditioned -- review finding).  Hardened: keeps up
 to nb candidates |lam|-sorted, skips nonfinite lam, keeps the SIGNED lam,
 mines near-real complex pairs for their real 2D invariant subspace, and
 attaches the FULL-system residual ||Ac - lam Bc||/||lam Bc|| to each.
 Returns tuples (lam_signed, c, fullres)."
function corner_branches(A, B, chi; nb=12)
    # STEP 1: Petrov-Galerkin reduction to a square pencil (spurious pairs possible)
    BA = B' * A; BB = Symmetric((B'B + (B'B)')/2)
    F = eigen(BA, BB)
    # STEP 2: visit candidates in descending |lam|
    ord = sortperm(abs.(F.values), rev=true)
    # STEP 3: FULL-system residual ||Ac - lam Bc||/||lam Bc|| (not the reduced one)
    fullres(lam, c) = begin
        v = vec(c)
        r = A*v .- lam .* (B*v)
        norm(r) / max(norm(lam .* (B*v)), 1e-300)
    end
    out = Tuple{Float64,Matrix{Float64},Float64}[]
    seen = Matrix{Float64}[]
    # STEP 4: normalize, drop near-zero and duplicate candidates, keep SIGNED lam
    pushc!(lam, c) = begin
        nrm = norm(c); nrm < 1e-12 && return
        c = c ./ nrm
        for s0 in seen; abs(sum(s0 .* c)) > 1 - 1e-10 && return; end
        push!(seen, c)
        push!(out, (lam, c, fullres(lam, c)))
    end
    # STEP 5: harvest -- skip junk lam, mine near-real pairs for their real 2D subspace
    for idx in ord
        length(out) >= nb && break
        lam = F.values[idx]
        (isfinite(abs(lam)) && abs(lam) < 1e8) || continue
        v = F.vectors[:, idx]
        if maximum(abs.(imag.(v))) <= 1e-8 * maximum(abs.(real.(v)))
            pushc!(real(lam), reshape(real.(v), chi, chi))
        elseif abs(imag(lam)) < 0.1 * abs(lam)
            # near-real pair: examine the real 2D invariant subspace
            pushc!(real(lam), reshape(real.(v), chi, chi))
            pushc!(real(lam), reshape(imag.(v), chi, chi))
        end
    end
    out
end

# ---------------------------------------------------------------------------
# ring_Omega2: one replica sector as the explicit RING with the corner
# pair (cA, cB) -- equivalent to the octahedron assembly, proven equal to
# octa_* at 1e-16 (yvertex12 gate).  Use octa_* for large chi: the ring
# BUILDS the chi^k x chi^k full-seam matrices, octa_* never does.
#
# in : M, cA, cB (corner pair; the C side closes on a delta), beta, k,
#      replica perms pA, pB, pC, dirs (one per seam, forwarded to seam_w)
# out: Om (scalar sector value)
#
#        ,--- C-side delta ----.
#   [wCA]                      [wCB]   Om = tr[ wCA' cA^(x)k wAB' cB^(x)k
#        \<cA^k>--[wAB]--<cB^k>/            wCB ]   (last factor UNPRIMED)
#
#   w convention (also the octa_* reference): w[row = Xj side, col = Xi
#   side]; index wiring  Om = sum wCA[A1,C1] cA[A1,A2] wAB[B1,A2]
#   cB[B1,B2] wCB[B2,C1].  The code's closing sum(T3 .* wBC') is exactly
#   tr(T3 * wBC), i.e. the trace above ends in wCB[B2,C1] unprimed.
#
# STEP 1 -- SEAM CAPS from seam_w: wAC = seam_w(..,'C','A',..) is the
#   wCA[A,C] of the wiring above; wAB ('A','B') is wAB[B,A]; wBC
#   ('C','B') is wCB[B,C].
# STEP 2 -- REPLICA CORNERS: ckA = cA^(x)k, ckB = cB^(x)k by repeated
#   kron(cA, ckA) -- replica 1 fastest, matching seam_w's standard order.
# STEP 3 -- CONTRACT the ring: T3 = (wAC' * ckA) * (wAB' * ckB) = [C1,B2],
#   then Om = sum(T3 .* wBC') closes the C-side delta (wBC' = [C1,B2]).
# ---------------------------------------------------------------------------
function ring_Omega2(M, cA, cB, beta, k, pA, pB, pC; dirs=(:R,:R,:R))
    # STEP 1: three full-seam caps, w[row = Xj side, col = Xi side]
    wAC, lAC, _ = seam_w(M, beta, k, 'C','A', pA, pB, pC; dir=dirs[1])
    wAB, lAB, _ = seam_w(M, beta, k, 'A','B', pA, pB, pC; dir=dirs[2])
    wBC, lBC, _ = seam_w(M, beta, k, 'C','B', pA, pB, pC; dir=dirs[3])
    ckA = cA; ckB = cB
    # STEP 2: replica-power the corners (replica 1 fastest, seam_w order)
    for j in 2:k
        ckA = kron(cA, ckA); ckB = kron(cB, ckB)
    end
    # STEP 3: overlap sum(T3 .* wBC') = tr(T3 * wBC) -- the C legs close on a delta
    T3 = (wAC' * ckA) * (wAB' * ckB)
    sum(T3 .* wBC')
end

# ---------------------------------------------------------------------------
# yring_gates: one-call sector battery for the ring assembly.
#
# in : M, cA, cB, beta        out: (G4, mirror, tildeS)
#
# STEP 1 -- FIVE TWISTED SECTORS via ring_Omega2:
#   O1 (k=1, trivial), O2A/O2B/O2C (k=2, PSWAP2 on exactly one region),
#   O4 (k=4, the SA4/SB4/SC4 triple of the genuine multientropy).
# STEP 2 -- UNTWISTED-ZERO GATE: the SAME 5-sector combination rebuilt
#   with all-identity perms (U1, U2, U4; the three untwisted O2 sectors
#   coincide, hence the factor 3):
#     g4 = |ln U4 + 2 ln U1 - 3 ln U2|  -- exactly 0 by pure normalization
#   counting; any seam/norm leak shows up here first.
# STEP 3 -- MIRROR GATE: mir = |O2A - O2B|/|O2A|, machine zero by the
#   gamma_A = gamma_B symmetry of the junction.
# STEP 4 -- READOUT:
#     tildeS = -[ln O4 + 2 ln O1 - ln O2A - ln O2B - ln O2C]
#   the counting that cancels every extensive lam^N and block norm.
# ---------------------------------------------------------------------------
function yring_gates(M, cA, cB, beta)
    # STEP 1: the five twisted sectors (k = 1, 2, 2, 2, 4)
    O1  = ring_Omega2(M, cA, cB, beta, 1, [1],[1],[1])
    O2A = ring_Omega2(M, cA, cB, beta, 2, PSWAP2, [1,2], [1,2])
    O2B = ring_Omega2(M, cA, cB, beta, 2, [1,2], PSWAP2, [1,2])
    O2C = ring_Omega2(M, cA, cB, beta, 2, [1,2], [1,2], PSWAP2)
    O4  = ring_Omega2(M, cA, cB, beta, 4, SA4, SB4, SC4)
    # untwisted-zero gate (all identity perms, same sector structure)
    U1  = O1
    # STEP 2: untwisted controls -- same sector structure, must combine to exactly 0
    U2  = ring_Omega2(M, cA, cB, beta, 2, [1,2], [1,2], [1,2])
    U4  = ring_Omega2(M, cA, cB, beta, 4, [1,2,3,4], [1,2,3,4], [1,2,3,4])
    g4 = abs(log(abs(U4)) + 2*log(abs(U1)) - 3*log(abs(U2)))
    # STEP 3: mirror gate (gamma_A = gamma_B)
    mir = abs(O2A - O2B) / abs(O2A)
    # STEP 4: the 5-sector combination -- extensive parts cancel
    tS = -(log(abs(O4)) + 2*log(abs(O1)) - log(abs(O2A)) - log(abs(O2B)) - log(abs(O2C)))
    (g4, mir, tS)
end


# ---------------------------------------------------------------------------
# seam_blocks: the two UNIVERSAL cycle blocks every sector is made of --
# computed once per (Mt, beta) and consumed by all octa_* assemblies.
#
# in : Mt (canonical arm), beta
# out: (W1, W2, l1, l2)
#      W1 [chi,chi]          untwisted single zip cap  = the user's S = T.T
#      W2 [chi,chi,chi,chi]  swap-twisted double cap   = the user's R,
#                            legs [Xj_r, Xj_s, Xi_r, Xi_s]
#
#   --(Mt)---           --(MtxMt)---        MtxMt = the 2-replica kron
#      |                     ||             rung (rails chi^2, bare legs
#     {I}    ->  W1         {G2}    -> W2   2^2); NOT two adjacent
#      |                     ||             columns -- one rung carries
#   --(Mt)---           --(MtxMt)---        both replica copies.
#
# STEP 1 -- W1: zip_eig with G = Id_2, nrep = 1 (no twist).
# STEP 2 -- G2: the universal swap weight on the FIXED ('A','B') seam,
#   seam_G(2, beta, 'A', 'B', PSWAP2, [1,2], [1,2]) -- only the RELATIVE
#   swap matters (same lesson as the seam_w BC-seam bugfix: a swap pinned
#   to the wrong side leaves a seam silently untwisted).
# STEP 3 -- W2: zip_eig on the 2-replica ladder gives a chi^2 x chi^2
#   matrix [bot, top]; reshape to 4 legs [Xj_r, Xj_s, Xi_r, Xi_s]
#   (replica r fast, s slow on both sides -- zip_eig's fastest-first
#   replica packing).
# ---------------------------------------------------------------------------
"universal seam blocks: W1 (untwisted single) and W2 (swap-twisted double,
 reshaped to 4 legs [Xj_r, Xj_s, Xi_r, Xi_s]); their eigenvalues."
function seam_blocks(Mt, beta)
    # STEP 1: untwisted single block W1
    W1, l1 = zip_eig(Mt, Matrix{Float64}(I,2,2), 1; dir=:R)
    # STEP 2: universal relative-swap weight on the fixed ('A','B') seam
    G2 = Matrix(seam_G(2, beta, 'A', 'B', PSWAP2, [1,2], [1,2]))
    W2m, l2 = zip_eig(Mt, G2, 2; dir=:R)
    chi = size(Mt,1)
    # STEP 3: reshape W2 to legs [Xj_r, Xj_s, Xi_r, Xi_s]
    (W1, reshape(W2m, chi,chi,chi,chi), l1, l2)
end

# ---------------- octahedron sector contractions ----------------
# ring reference wiring (ring_Omega2):
#   Om = sum wCA[A1,C1] cA[A1,A2] wAB[B1,A2] cB[B1,B2] wCB[B2,C1]
# with w[row = Xj side, col = Xi side].

# ---------------------------------------------------------------------------
# octa_O1: k=1 normalization triangle (the user's Tr S^3, with corners).
# All three seams carry the SAME untwisted block W1.
#
# in : W1[chi,chi], cA, cB          out: O1 (scalar)
#
#   [W1_CA]--<cA>--[W1_AB]--<cB>--[W1_CB]--(close on the C-side delta)
#
# Reference wiring (ring_Omega2 at k=1, all w -> W1):
#   O1 = sum wCA[A1,C1] cA[A1,A2] wAB[B1,A2] cB[B1,B2] wCB[B2,C1]
#
# STEP 1 -- X = cA' * W1 : the CA seam pulled through the A corner,
#   X[A2,C1].
# STEP 2 -- Y = cB' * W1 : the AB seam pulled through the B corner,
#   Y[B2,A2].
# STEP 3 -- Z = Y * X = [B2,C1]; close with the CB seam:
#   O1 = sum(Z .* W1), with W1 sitting in the wCB[B2,C1] slot (the C legs
#   of wCA and wCB are identified directly -- the C region has no corner).
# ---------------------------------------------------------------------------
"k=1 triangle."
function octa_O1(W1, cA, cB)
    # STEP 1: CA seam through the A corner
    X = cA' * W1              # [A2, C1]
    # STEP 2: AB seam through the B corner
    Y = cB' * W1              # [B2, A2]
    # STEP 3: chain and close on the CB seam (C-side delta)
    Z = Y * X                 # [B2, C1]
    sum(Z .* W1)              # wCB[B2, C1]
end

# ---------------------------------------------------------------------------
# octa_O2A: k=2 sector gamma_A -- seams CA & AB swap-twisted (two W2
# blocks), CB untwisted (factorizes into two per-replica W1 chains).
# = the user's Tr( R (...) R (...) ) two-R diagram.  Mirror partner of
# octa_O2B; O2A == O2B is the mirror gate.
#
# in : W2 [Xj_r,Xj_s,Xi_r,Xi_s], W1, cA, cB          out: O2A (scalar)
#
#   [W2_CA]--<cA><cA>--[W2_AB]--<cB><cB>--[W1][W1]--(close C-side delta)
#
# STEP 1 -- K = kron(cA, cA): the replica pair of A corners.
# STEP 2 -- Qt = reshape(W2 * K', ...) = [B1,B2,A1,A2]: the AB block
#   (W2 read as wAB[Bpair,Apair]) pre-contracted with both cA's; the free
#   A legs face the CA seam.
# STEP 3 -- Y = Qt * W2 = [B1,B2,C1,C2]: attach the CA block (W2 read as
#   wCA[Apair,Cpair]) -- both twisted seams are now fused.
# STEP 4 -- V = cB * W1: the per-replica UNTWISTED CB chain,
#   V[B,C] = sum_b cB[B,b] W1[b,C]   (cB[B1 = Q side, b = CB side]).
# STEP 5 -- CLOSE: O2A = sum Y[b1,b2,c1,c2] V[b1,c1] V[b2,c2] -- the
#   untwisted CB seam acts replica-by-replica (one V per replica) and the
#   C legs close on the C-side delta.
# ---------------------------------------------------------------------------
"k=2, sector 2A: seams CA+AB swap-twisted, CB untwisted."
function octa_O2A(W2, W1, cA, cB)
    chi = size(cA,1)
    # STEP 1: replica pair of A corners
    K = kron(cA, cA)
    # STEP 2: AB block pre-dressed with both cA's
    Qt = reshape(reshape(W2, chi*chi, chi*chi) * K', chi,chi,chi,chi) # [B1,B2,A1,A2]
    # STEP 3: attach the CA block -> [B1,B2,C1,C2]
    Y = reshape(reshape(Qt, chi*chi, chi*chi) * reshape(W2, chi*chi, chi*chi),
                chi,chi,chi,chi)                                      # [B1,B2,C1,C2]
    # STEP 4: per-replica untwisted CB chain V[B,C]
    V = cB * W1                                                       # wait: see below
    # V[B,C] = sum_b cB[B,b] W1[b,C]  with cB[B1(Q side), b(CB side)]
    s = 0.0
    # STEP 5: close replica-by-replica on the C-side delta
    @inbounds for b1 in 1:chi, b2 in 1:chi, c1 in 1:chi, c2 in 1:chi
        s += Y[b1,b2,c1,c2] * V[b1,c1] * V[b2,c2]
    end
    s
end

# ---------------------------------------------------------------------------
# octa_O2B: k=2 sector gamma_B -- seams AB & CB swap-twisted, CA
# untwisted.  Mirror partner of octa_O2A (O2A == O2B is the mirror gate;
# it holds because only the RELATIVE twist per seam matters).
#
# in : W2 [Xj_r,Xj_s,Xi_r,Xi_s], W1, cA, cB          out: O2B (scalar)
#
#   [W1][W1]--<cA><cA>--[W2_AB]--<cB><cB>--[W2_CB]--(close C-side delta)
#
# STEP 1 -- Wt = cA' * W1 = [x, C]: the per-replica UNTWISTED CA chain
#   (x = AB-side A index) -- one chain per replica, the counterpart of V
#   in octa_O2A.
# STEP 2 -- Qh = reshape(kron(cB,cB)' * W2, ...):
#   Qh[y1,y2,x1,x2] = sum_B cB[B1,y1] cB[B2,y2] W2[B1,B2,x1,x2]
#   -- the twisted AB block dressed with both cB's (y = CB-side B index).
# STEP 3 -- INNER SUM: hook Qh's x pair to the two Wt chains,
#   z[y1,y2,c1,c2] = sum_{x1,x2} Qh[y1,y2,x1,x2] Wt[x1,c1] Wt[x2,c2].
# STEP 4 -- CLOSE with the twisted CB block: O2B += z * W2[y1,y2,c1,c2]
#   (the C legs of both twisted blocks meet on the C-side delta).
# ---------------------------------------------------------------------------
"k=2, sector 2B: seams AB+CB twisted, CA untwisted."
function octa_O2B(W2, W1, cA, cB)
    chi = size(cA,1)
    # STEP 1: per-replica untwisted CA chain
    Wt = cA' * W1                                # [x, C]  (x = AB-side A index)
    KB = kron(cB, cB)
    # STEP 2: dress the twisted AB block with both cB's
    Qh = reshape(KB' * reshape(W2, chi*chi, chi*chi), chi,chi,chi,chi)
    # Qh[y1,y2,x1,x2] = sum_B cB[B1,y1] cB[B2,y2] W2[B1,B2,x1,x2]
    s = 0.0
    @inbounds for y1 in 1:chi, y2 in 1:chi, c1 in 1:chi, c2 in 1:chi
        z = 0.0
        # STEP 3: contract onto the two CA chains; STEP 4 closes with the CB-side W2 below
        for x1 in 1:chi, x2 in 1:chi
            z += Qh[y1,y2,x1,x2] * Wt[x1,c1] * Wt[x2,c2]
        end
        s += z * W2[y1,y2,c1,c2]
    end
    s
end

# ---------------------------------------------------------------------------
# octa_O2C: k=2 sector gamma_C -- seams CA & CB swap-twisted, AB
# untwisted.  The untwisted AB seam collapses to one per-replica chain
# L = cA W1^T cB linking the two W2 blocks.
#
# in : W2 [Xj_r,Xj_s,Xi_r,Xi_s], W1, cA, cB          out: O2C (scalar)
#
#   [W2_CA]--<cA>--[W1]--<cB>--[W2_CB]      (two such middle chains,
#            <cA>--[W1]--<cB>               one per replica)
#
# STEP 1 -- L = cA * transpose(W1) * cB = [A, y]: the per-replica chain
#   through the untwisted AB seam -- cA[A1,A2] wAB[B1,A2] cB[B1,B2] with
#   W1 in the wAB slot (hence the transpose to [A,B] order).
# STEP 2 -- INNER SUM: hook the CA block to the two L chains,
#   z[y1,y2,c1,c2] = sum_{a1,a2} W2[a1,a2,c1,c2] L[a1,y1] L[a2,y2].
# STEP 3 -- CLOSE with the CB block: O2C += z * W2[y1,y2,c1,c2].  The two
#   blocks' C-leg pairs are identified directly = the C-side delta (the C
#   region carries no corner in this assembly).
# ---------------------------------------------------------------------------
"k=2, sector 2C: seams CA+CB twisted, AB untwisted."
function octa_O2C(W2, W1, cA, cB)
    chi = size(cA,1)
    # STEP 1: per-replica untwisted AB chain (W1 transposed into the wAB slot)
    L = cA * transpose(W1) * cB                  # [A, y]  per replica chain
    s = 0.0
    @inbounds for y1 in 1:chi, y2 in 1:chi, c1 in 1:chi, c2 in 1:chi
        z = 0.0
        # STEP 2: hook the CA block to both chains; STEP 3 closes with the CB block below
        for a1 in 1:chi, a2 in 1:chi
            z += W2[a1,a2,c1,c2] * L[a1,y1] * L[a2,y2]
        end
        s += z * W2[y1,y2,c1,c2]
    end
    s
end

# ---------------------------------------------------------------------------
# octa_O4: gamma_2^(3) numerator -- the K_{2,2,2} octahedron (user's page
# C).  Replica pairings: CA (1,3)(2,4); AB (1,2)(3,4); CB (1,4)(2,3).
#
# in : W2 [Xj_r,Xj_s,Xi_r,Xi_s] universal swap block, cA, cB
# out: O4 (scalar)
#
#            [AB(12)]
#      [CA(13)]    [CB(14)]     vertices: 6 x W2 (3 seams x 2 two-cycles),
#      [CB(23)]    [CA(24)]     opposite vertices = same seam (no edge);
#            [AB(34)]           12 chi-edges: mu_r -> cA, nu_r -> cB,
#                               lambda_r -> delta (C side, no corner).
#
# Largest stored intermediate chi^8 (X2/X3); the chi^4 x chi^4 full-seam
# matrices of the ring form are NEVER built -- that is the point of the
# octahedron.  Contraction order X1..X4 + final close (= note sec 6.2's
# X1..X5).
#
# STEP 1 -- PRE-DRESS the corner edges into vertex blocks:
#   Qt = W2 * kron(cA,cA)'  = [b_r,b_s,A_r,A_s]   (AB block + its 2 cA's)
#   Rt = kron(cB,cB) * W2   = [bQ_r,bQ_s,C_r,C_s] (CB block + its 2 cB's)
#   P  = W2                 = [A_r,A_s,C_r,C_s]   (CA block, bare)
# STEP 2 -- X1[a3,c1,c3,b1,b2,a2] = sum_a1 P[a1,a3,c1,c3] Qt[b1,b2,a1,a2]
#   fuse CA(13) with AB(12) over the replica-1 edge.      (chi^6 object)
# STEP 3 -- X2[c1,c3,b1,b2,a2,b3,b4,a4] = sum_a3 X1 Qt[b3,b4,a3,a4]
#   attach AB(34) over the replica-3 edge.                (chi^8 object)
# STEP 4 -- X3[c1,c3,b1,b2,b3,b4,c2,c4] = sum_{a2,a4} X2 P[a2,a4,c2,c4]
#   attach CA(24); all four A edges are now closed.       (chi^8 object)
# STEP 5 -- X4[c3,b2,b3,c2] = sum_{b1,b4,c1,c4} X3 Rt[b1,b4,c1,c4]
#   attach CB(14): four legs contract at once.            (chi^4 object)
# STEP 6 -- CLOSE: O4 = sum X4[c3,b2,b3,c2] Rt[b2,b3,c2,c3] -- the last
#   vertex CB(23) eats the remaining b edges and C-delta edges.
# ---------------------------------------------------------------------------
"k=4: the K_{2,2,2} octahedron. Cycles: CA (1,3)(2,4); AB (1,2)(3,4);
 CB (1,4)(2,3). Vertices = 6 copies of W2; edges carry cA/cB/delta."
function octa_O4(W2, cA, cB)
    chi = size(cA,1)
    # STEP 1: pre-dress corner edges -- Qt (AB+cA pair), Rt (CB+cB pair), P (bare CA)
    K  = kron(cA, cA)
    Qt = reshape(reshape(W2, chi*chi, chi*chi) * K', chi,chi,chi,chi)  # [b_r,b_s,A_r,A_s]
    KB = kron(cB, cB)
    Rt = reshape(KB * reshape(W2, chi*chi, chi*chi), chi,chi,chi,chi)  # [bQ_r,bQ_s,C_r,C_s]
    P  = W2                                                            # [A_r,A_s,C_r,C_s]
    # X1[a3,c1,c3,b1,b2,a2] = sum_a1 P[a1,a3,c1,c3] Qt[b1,b2,a1,a2]
    # STEP 2: fuse CA(13) with AB(12) over a1 (chi^6 object)
    X1 = zeros(chi,chi,chi,chi,chi,chi)
    @inbounds for a3 in 1:chi, c1 in 1:chi, c3 in 1:chi, b1 in 1:chi, b2 in 1:chi, a2 in 1:chi
        s = 0.0
        for a1 in 1:chi; s += P[a1,a3,c1,c3] * Qt[b1,b2,a1,a2]; end
        X1[a3,c1,c3,b1,b2,a2] = s
    end
    # X2[c1,c3,b1,b2,a2,b3,b4,a4] = sum_a3 X1[a3,...] Qt[b3,b4,a3,a4]
    # STEP 3: attach AB(34) over a3 -- chi^8, the largest stored object
    X2 = zeros(chi,chi,chi,chi,chi,chi,chi,chi)
    @inbounds for c1 in 1:chi, c3 in 1:chi, b1 in 1:chi, b2 in 1:chi, a2 in 1:chi,
                  b3 in 1:chi, b4 in 1:chi, a4 in 1:chi
        s = 0.0
        for a3 in 1:chi; s += X1[a3,c1,c3,b1,b2,a2] * Qt[b3,b4,a3,a4]; end
        X2[c1,c3,b1,b2,a2,b3,b4,a4] = s
    end
    # X3[c1,c3,b1,b2,b3,b4,c2,c4] = sum_{a2,a4} X2[...] P[a2,a4,c2,c4]
    # STEP 4: attach CA(24), closing all A edges (chi^8 object)
    X3 = zeros(chi,chi,chi,chi,chi,chi,chi,chi)
    @inbounds for c1 in 1:chi, c3 in 1:chi, b1 in 1:chi, b2 in 1:chi,
                  b3 in 1:chi, b4 in 1:chi, c2 in 1:chi, c4 in 1:chi
        s = 0.0
        for a2 in 1:chi, a4 in 1:chi
            s += X2[c1,c3,b1,b2,a2,b3,b4,a4] * P[a2,a4,c2,c4]
        end
        X3[c1,c3,b1,b2,b3,b4,c2,c4] = s
    end
    # X4[c3,b2,b3,c2] = sum_{b1,b4,c1,c4} X3[...] Rt[b1,b4,c1,c4]
    # STEP 5: attach CB(14), collapse to chi^4; STEP 6 closes with CB(23) below
    X4 = zeros(chi,chi,chi,chi)
    @inbounds for c3 in 1:chi, b2 in 1:chi, b3 in 1:chi, c2 in 1:chi
        s = 0.0
        for b1 in 1:chi, b4 in 1:chi, c1 in 1:chi, c4 in 1:chi
            s += X3[c1,c3,b1,b2,b3,b4,c2,c4] * Rt[b1,b4,c1,c4]
        end
        X4[c3,b2,b3,c2] = s
    end
    # Om = sum X4[c3,b2,b3,c2] Rt[b2,b3,c2,c3]
    s = 0.0
    @inbounds for c3 in 1:chi, b2 in 1:chi, b3 in 1:chi, c2 in 1:chi
        s += X4[c3,b2,b3,c2] * Rt[b2,b3,c2,c3]
    end
    s
end

# ---------------------------------------------------------------------------
# reldiff: symmetric relative difference |x-y| / max(|x|,|y|,1e-300) --
# the metric of every equality gate (ring vs octahedron, mirror, ...);
# the 1e-300 floor keeps 0-vs-0 comparisons finite.
# ---------------------------------------------------------------------------
reldiff(x, y) = abs(x - y) / max(abs(x), abs(y), 1e-300)

# ---------------------------------------------------------------------------
# pick_corner: the certified-corner selector (v11 selection, compact).
# Builds the full readout environment, solves the corner equation, and
# scores EVERY (branch, transpose pattern) against exact answers.  The
# physical branch is selected by the cross gates, NEVER by |lam| -- the
# refuted v3 corner is the standing proof that self-consistency lies.
#
# in : Mt (canonical arm), a, am, ss (exact Onsager NN), keff (chi_eff)
# out: (score, c, pat)   score = max(e1, e2, spread) of the winner;
#      score <~ 3e-3 = wall floor at beta=0.30  -> certified;
#      score ~  1e-2 (criticality)              -> NOT certified.
#
#   caps + benchmark -> walls -> corner branches -> gate scan -> best
#
# STEP 1 -- CAPS: the four channel fixed points FU/FR/FD/FL (cap_fp with
#   growU/R/D/L) -- the exact semi-infinite channel environments.
# STEP 2 -- BENCHMARK: corner-free diagonal correlator db
#   (diag_benchmark), the X2 reference; Onsager ss arrives as an argument.
# STEP 3 -- WALLS: Ep = left wall (intertwiner_left), Er = right wall
#   (intertwiner_right).  BOTH are needed: E != E'^T (the canonical gauge
#   is not reflection covariant) -- reusing E'^T for the vertical arm was
#   the v8 bug (cross gates 0.33 -> 3e-3 after the fix).
# STEP 4 -- CORNER EQUATION: (A11, B11) = corner_maps11(Mt, Ep, Er, a);
#   corner_branches returns ALL hardened candidates (up to nb=12,
#   |lam|-sorted, near-real complex pairs mined for their real 2D
#   subspace, full residual attached).
# STEP 5 -- GATE SCAN: for every branch x pattern 1..4 run cross_gates;
#   sc = max(e1, e2, spread); keep the minimum.  best starts at
#   (Inf, Id, 0), so if nothing scores finite the caller sees score=Inf.
# ---------------------------------------------------------------------------
"pick the certified corner (v11 selection, compact)."
function pick_corner(Mt, a, am, ss, keff)
    # STEP 1: four channel caps (exact semi-infinite environments)
    FU,_ = cap_fp(growU, Mt, a); FR,_ = cap_fp(growR, Mt, a)
    FD,_ = cap_fp(growD, Mt, a); FL,_ = cap_fp(growL, Mt, a)
    # STEP 2: corner-free diagonal benchmark for gate X2
    db = diag_benchmark(Mt, a, am)
    # STEP 3: BOTH walls -- E != E'^T (v8 bug: cross gates 0.33 -> 3e-3)
    Ep,_,_,_ = intertwiner_left(Mt, a)
    Er,_,_   = intertwiner_right(Mt, a)
    # STEP 4: corner generalized eigenproblem, all hardened branches
    A11, B11 = corner_maps11(Mt, Ep, Er, a)
    brs = corner_branches(A11, B11, keff)
    best = (Inf, Matrix{Float64}(I,keff,keff), 0)
    # STEP 5: exact-answer gate scan over branch x LEGAL pattern (1,4 only;
    # mixed 2/3 = negative controls, excluded since the 2026-08-10 review)
    for (lam, cb, _) in brs, pat in (1, 4)
        e1, e2, sp, _ = cross_gates(Mt, a, am, FU, FR, FD, FL, cb, pat, ss, db)
        sc = max(e1, e2, sp)
        sc < best[1] && (best = (sc, cb, pat))
    end
    best
end

