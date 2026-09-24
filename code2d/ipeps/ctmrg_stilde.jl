# ctmrg_stilde.jl -- THE Y-JUNCTION WINDOW: tildeS from CTMRG environments.
# =============================================================================
# Decomposition (cross-sector environment norms are incomparable, so tildeS
# is split into three WITHIN-sector ratio quantities):
#     tildeS_J = -[ kappa_Y - kappa_A - kappa_B ],   kappa_B = kappa_A (mirror)
#   kappa_A = ln(Z_bent/Z_straight)   (k=2 window, = the certified gamma run)
#   kappa_Y = ln W_Y - (1/2) [ln Ws_CA + ln Ws_CB + ln Ws_AB] + (1/2) ln W0
#             (k=4; counting cancels every C/T/Td/core-site factor exactly;
#              the same-form k=2 combo reduces to ln(Zb/Zs) -- validated to
#              0.06% in ctmrg_gamma v2)
# Geometry (k=4, sigma_A=1, sigma_B=(12)(34), sigma_C=(13)(24)):
#   horizontal seam between rows: west segment G_CA=(13)(24), east G_CB=(14)(23);
#   vertical ray between columns, northward: G_AB=(12)(34).
#   All three are double 2-cycles => straight tension = 2*s2 EXACTLY (gate).
# Implementation: d=16 forbids materializing 2x2-core windows -> column-sweep
# contraction, VALIDATED bitwise against the d=4 window1/window2 (which the
# gamma campaign already certified end-to-end).
# Env: CSBETA (default 0.30), CSCHI (colon list, default 24:32).
# =============================================================================
include(joinpath(@__DIR__, "yvx_core.jl"))

const CSBETA = parse(Float64, get(ENV, "CSBETA", "0.30"))
const CSCHIS = haskey(ENV, "CSCHI") ? parse.(Int, split(ENV["CSCHI"], ":")) : [24, 32]
const CSNMAX = parse(Int, get(ENV, "CSNMAX", "20000"))   # crit needs O(xi(chi)) steps
# multiplet-aware truncation (criticality): never cut inside a degenerate corner
# multiplet (replica-permutation multiplets are EXACT degeneracies of the k-replica
# spectrum; a mid-multiplet cut breaks replica symmetry -> three-seam gate failure).
# CSDEG = relative degeneracy tolerance (0 = off, exact v1 path); CSSLACK = max
# extension of chi above the target before we shrink to the gap below instead.
const CSDEG   = parse(Float64, get(ENV, "CSDEG", "0"))
const CSSLACK = parse(Int, get(ENV, "CSSLACK", "8"))

# ---------------- shared machinery (as ctmrg_gamma v2) ----------------
function build_a2(a)
    d = size(a, 1); D = d*d
    a2 = zeros(D, D, D, D)
    @inbounds for u1 in 1:d, l1 in 1:d, dn1 in 1:d, r1 in 1:d,
                  u2 in 1:d, l2 in 1:d, dn2 in 1:d, r2 in 1:d
        a2[(u2-1)*d+u1, (l2-1)*d+l1, (dn2-1)*d+dn1, (r2-1)*d+r1] =
            a[u1,l1,dn1,r1] * a[u2,l2,dn2,r2]
    end
    a2
end

function dress(aT, G, slot::Symbol)
    D = size(aT, 1); out = zeros(D, D, D, D)
    @inbounds for u in 1:D, l in 1:D, dn in 1:D, r in 1:D, x in 1:D
        if slot === :u
            out[u,l,dn,r] += G[u,x] * aT[x,l,dn,r]
        elseif slot === :l
            out[u,l,dn,r] += G[l,x] * aT[u,x,dn,r]
        else
            out[u,l,dn,r] += G[r,x] * aT[u,l,dn,x]
        end
    end
    out
end

function ctm_step_Z(C, T, a, χmax)
    χ = size(C,1); dd = size(a,1); cd = χ*dd
    CTl = reshape(C*reshape(T, χ, χ*dd), χ, χ, dd)
    res = reshape(reshape(permutedims(T,(2,1,3)), χ, χ*dd)' * reshape(CTl, χ, χ*dd),
                  χ, dd, χ, dd)
    Cb  = reshape(reshape(permutedims(res,(1,3,2,4)), χ*χ, dd*dd) * reshape(a, dd*dd, dd*dd),
                  χ, χ, dd, dd)
    Cbig = reshape(permutedims(Cb,(1,4,2,3)), cd, cd); Cbig = (Cbig + Cbig')/2
    F = eigen(Symmetric(Cbig)); ps = sortperm(abs.(F.values), rev=true)
    χcut = min(χmax, cd)
    if CSDEG > 0 && χcut < cd
        deg(i) = abs(F.values[ps[i+1]]) > (1-CSDEG)*abs(F.values[ps[i]])
        hi = min(χmax + CSSLACK, cd)
        c1 = χcut
        while c1 < hi && deg(c1); c1 += 1; end
        if c1 == hi && c1 < cd && deg(c1)
            c1 = χcut                       # slack exhausted: shrink to the gap below
            lo = max(χmax ÷ 2, 2)
            while c1 > lo && deg(c1); c1 -= 1; end
        end
        χcut = c1
    end
    rcut = χcut < cd ? abs(F.values[ps[χcut+1]])/abs(F.values[ps[χcut]]) : 0.0
    p = ps[1:χcut]
    Z = F.vectors[:, p]
    Cn = Z' * Cbig * Z; Cn = (Cn + Cn')/2; Cn ./= maximum(abs, Cn)
    (Cn, Z, rcut)
end

function grow_T(T, adr, Z)
    χ = size(T,1); dd = size(adr,1); cd = χ*dd
    res5 = reshape(reshape(permutedims(T,(3,1,2)), dd, χ*χ)' * reshape(adr, dd, dd*dd*dd),
                   χ, χ, dd, dd, dd)
    Tbig = reshape(permutedims(res5,(1,3,2,5,4)), cd, cd, dd)
    χn = size(Z,2)
    Tmid = reshape(Z' * reshape(Tbig, cd, cd*dd), χn, cd, dd)
    Tn = permutedims(reshape(Z' * reshape(permutedims(Tmid,(2,1,3)), cd, χn*dd), χn, χn, dd), (2,1,3))
    Tn ./= maximum(abs, Tn)
    Tn
end

function ctmrg_defect(aT, adrs::Vector; χmax, nmax=CSNMAX, tol=1e-12, tag="")
    C, T = init_CT(aT)
    Tds = [copy(T) for _ in adrs]
    sold = Float64[]; itdone = nmax; drift = NaN; rcut = NaN
    for it in 1:nmax
        Cn, Z, rcut = ctm_step_Z(C, T, aT, χmax)
        Tn  = grow_T(T, aT, Z)
        Tdn = [grow_T(Td, adr, Z) for (Td, adr) in zip(Tds, adrs)]
        C, T, Tds = Cn, Tn, Tdn
        sv = sort(abs.(eigvals(Symmetric((C+C')/2))), rev=true); sv ./= sv[1]
        if length(sv) == length(sold)
            drift = maximum(abs.(sv .- sold))
            if drift < tol
                itdone = it; break
            end
        end
        sold = sv
        if it % 2000 == 0
            @printf("    [%s it=%d drift=%.2e]\n", tag, it, drift); flush(stdout)
        end
    end
    if CSDEG > 0
        @printf("    [%s chi_fin=%d rcut=%.6f]\n", tag, size(C,1), rcut)
    end
    (C, T, Tds, itdone, drift)
end

# correlation length from the boundary-T channel  E = sum_p T[:,:,p] (x) T[:,:,p]
# (dense eigvals; N = chi^2 <= ~9e3 is minutes on 8 cores)
function xi_from_T(T)
    χ = size(T, 1); d = size(T, 3)
    E = zeros(χ*χ, χ*χ)
    for p in 1:d
        A = T[:, :, p]; E .+= kron(A, A)
    end
    mags = sort(abs.(eigvals(E)), rev=true)
    r = mags[2]/mags[1]
    (r < 1 ? -1/log(r) : Inf, mags[2:min(4, end)] ./ mags[1])
end

flipT(T) = permutedims(T, (2,1,3))

function window1(C, TN, TE, TS, TW, center)
    d = size(center, 1)
    AN = [C*TN[:,:,p] for p in 1:d]; AE = [C*TE[:,:,p] for p in 1:d]
    AS = [C*TS[:,:,p] for p in 1:d]; AW = [C*TW[:,:,p] for p in 1:d]
    z = 0.0
    @inbounds for u in 1:d, l in 1:d, dn in 1:d, r in 1:d
        c = center[u,l,dn,r]; c == 0.0 && continue
        z += c * tr(AN[u]*AE[r]*AS[dn]*AW[l])
    end
    z
end

# ---------------- column-sweep windows (any d; n = 1 or 2) ----------------
# ring convention (matches window1/window2 of the certified gamma run):
#   tr( C TN1(u1)..TNn(un) C TEn(rn)..TE1(r1) C TSn(dn)..TS1(d1) C TW1(l1)..TWn(ln) )
# sweep west->east with boundary vector v[iN, (l_n..l_1), iS]:
#   iN = north-chain bond, iS = south-chain bond (south chain pre-flipped).
"n-column window by west->east sweep (staged matmuls; any d; n = 1 or 2).
 TNs/TEs/TSs/TWs: edge tensors, index 1 = seam-adjacent position; cols[x] =
 core sites bottom(y=1)->top(y=n).  Ring convention identical to
 window1/window2 (gated below).  Boundary vector v[iS, l1..ln, iN]."
function window_sweep(C, TNs, TEs, TSs, TWs, cols)
    n = length(TNs); d = size(cols[1][1], 1); χ = size(C, 1)
    dn_ = d^n
    v = zeros(χ, ntuple(_ -> d, n)..., χ)
    for I in CartesianIndices(ntuple(_ -> d, n))
        B = C
        for y in 1:n; B = B * TWs[y][:, :, I[y]]; end
        B = B * C
        @inbounds @views v[:, I, :] .= B          # B[iS -> iN]
    end
    for x in 1:n
        TSf = flipT(TSs[x]); TNx = TNs[x]
        M = reshape(v, χ*dn_, χ) * reshape(TNx, χ, χ*d)          # (iS,L),(jN,u)
        t = reshape(M, χ, ntuple(_->d,n)..., χ, d)
        if n == 1
            s1 = cols[x][1]
            tp = reshape(permutedims(t, (1,3,2,4)), χ*χ, d*d)     # (iS,jN),(l1,u)
            S1 = reshape(permutedims(s1, (2,1,3,4)), d*d, d*d)    # (l1,u)->(dn,r1)
            q  = reshape(tp * S1, χ, χ, d, d)                     # iS,jN,dn,r1
            qp = reshape(permutedims(q, (2,4,1,3)), χ*d, χ*d)     # (jN,r1),(iS,dn)
            TSm = reshape(permutedims(TSf, (1,3,2)), χ*d, χ)      # (iS,dn)->jS
            r  = reshape(qp * TSm, χ, d, χ)                       # jN,r1,jS
            v  = permutedims(r, (3,2,1))                          # jS,r1,jN
        else
            s1 = cols[x][1]; s2 = cols[x][2]
            tp = reshape(permutedims(t, (1,2,4,3,5)), χ*d*χ, d*d) # (iS,l1,jN),(l2,u)
            S2 = reshape(permutedims(s2, (2,1,3,4)), d*d, d*d)    # (l2,u)->(vv,r2)
            q  = reshape(tp * S2, χ, d, χ, d, d)                  # iS,l1,jN,vv,r2
            qp = reshape(permutedims(q, (1,3,5,2,4)), χ*χ*d, d*d) # (iS,jN,r2),(l1,vv)
            S1 = reshape(permutedims(s1, (2,1,3,4)), d*d, d*d)    # (l1,vv)->(dn,r1)
            w  = reshape(qp * S1, χ, χ, d, d, d)                  # iS,jN,r2,dn,r1
            wp = reshape(permutedims(w, (2,3,5,1,4)), χ*d*d, χ*d) # (jN,r2,r1),(iS,dn)
            TSm = reshape(permutedims(TSf, (1,3,2)), χ*d, χ)
            r  = reshape(wp * TSm, χ, d, d, χ)                    # jN,r2,r1,jS
            v  = permutedims(r, (4,3,2,1))                        # jS,r1,r2,jN
        end
    end
    z = 0.0
    for I in CartesianIndices(ntuple(_ -> d, n))
        B = C
        for y in n:-1:1; B = B * TEs[y][:, :, I[y]]; end
        B = B * C
        z += sum(transpose(@view v[:, I, :]) .* B)  # v[jS,I,jN]*B[jN,jS]
    end
    z
end

# ---------------- main ----------------
const S2REF = Dict(0.30 => 0.05138867, 0.35 => 0.07242187)
const TSREF = Dict(0.30 => "jbench: 0.01104 (L=12, still +~1.5e-4/step) => ~0.0111-0.0115",
                   0.35 => "jbench: >=0.0343 (L=12, rising)",
                   0.44068679 => "MC crit campaign: tildeS(L) peaks ~L=12-14 then declines (L=64: -0.250); 0.103lnL NOT seen")

# float-tolerant dict lookup (env-parsed beta never equals a literal key exactly)
function refval(d::Dict, β)
    for (k, v) in d
        abs(k - β) < 1e-6 && return v
    end
    nothing
end

beta = CSBETA
@printf("CTMRG-STILDE  beta=%.4f\n", beta)
a, _ = ising_tensors(beta)

# ===== k=2 block: kappa_A (re-derive; also validates the sweep engine) =====
a2 = build_a2(a)
G2 = Matrix(seam_G(2, beta, 'C', 'A', [1,2], [1,2], PSWAP2))
a2Gu = dress(a2, G2, :u)
for χenv in CSCHIS
    println("-"^76)
    @printf("chi_env=%d\n", χenv)
    adr2 = dress(a2, G2, :r)
    C2, T2, Tds2, it2, dr2 = ctmrg_defect(a2, [adr2]; χmax=χenv, tag="k2")
    Td2 = Tds2[1]
    xi2, r2s = xi_from_T(T2)
    @printf("  conv(k2): it=%d drift=%.1e   xi2(chi)=%.4f  ratios=%s\n",
            it2, dr2, xi2, string(round.(r2s, digits=6))); flush(stdout)
    # sweep-engine validation against the certified ring windows (k=2, n=1,2)
    Z0_ring = window1(C2, T2, T2, T2, T2, a2)
    Z0_swp  = window_sweep(C2, [T2], [T2], [T2], [T2], [[a2]])
    Zb_ring = window1(C2, Td2, T2, T2, Td2, a2Gu)
    Zb_swp  = window_sweep(C2, [Td2], [T2], [T2], [Td2], [[a2Gu]])
    @printf("  gate(sweep,n=1): |Z0 ring-sweep| rel=%.2e  |Zb| rel=%.2e\n",
            abs(Z0_ring-Z0_swp)/abs(Z0_ring), abs(Zb_ring-Zb_swp)/abs(Zb_ring))
    # n=2 sweep vs known combo: straight tension gate through the sweep
    Zs1 = window_sweep(C2, [T2], [flipT(Td2)], [T2], [Td2], [[a2Gu]])
    Zs2 = window_sweep(C2, [T2,T2], [flipT(Td2),T2], [T2,T2], [Td2,T2],
                       [[a2Gu,a2], [a2Gu,a2]])
    Z02 = window_sweep(C2, [T2,T2], [T2,T2], [T2,T2], [T2,T2],
                       [[a2,a2], [a2,a2]])
    s2win = -(log(abs(Zs2/Z02)) - log(abs(Zs1/Z0_swp)))
    s2cert = refval(S2REF, beta)
    if s2cert !== nothing
        @printf("  gate(sweep,n=2 tension): s2=%.8f vs cert %.8f (rel %.1e)\n",
                s2win, s2cert, abs(s2win-s2cert)/s2cert)
    else
        @printf("  s2(chi) [sweep,n=2] = %.8f  (no cert ref at this beta; k=4 gate uses 2*s2win)\n", s2win)
    end
    Zb1 = Zb_swp
    kappaA = log(abs(Zb1/Zs1))
    @printf("  kappa_A = ln(Zb/Zs) = %+.8f   (gamma-v2 anchor: +0.0033612 @0.30)\n", kappaA)

    # ===== k=4 block: the Y junction =====
    a4 = build_a2(a2)                       # replica-1 fastest throughout
    GCA = Matrix(seam_G(4, beta, 'C', 'A', SA4, SB4, SC4))
    GCB = Matrix(seam_G(4, beta, 'C', 'B', SA4, SB4, SC4))
    GAB = Matrix(seam_G(4, beta, 'A', 'B', SA4, SB4, SC4))
    id4 = [1, 2, 3, 4]
    G4id = Matrix(seam_G(4, beta, 'C', 'A', id4, id4, id4))
    @printf("  gate(G4 untwisted)=%.1e\n", norm(G4id - I))
    aCAu = dress(a4, GCA, :u); aCBu = dress(a4, GCB, :u); aABr = dress(a4, GAB, :r)
    adrs4 = [dress(a4, GCA, :r), dress(a4, GCB, :r), dress(a4, GAB, :r)]
    C4, T4, Tds4, it4, dr4 = ctmrg_defect(a4, adrs4; χmax=χenv, tag="k4")
    TdCA, TdCB, TdAB = Tds4
    xi4, r4s = xi_from_T(T4)
    @printf("  conv(k4): it=%d drift=%.1e   xi4(chi)=%.4f  ratios=%s\n",
            it4, dr4, xi4, string(round.(r4s, digits=6))); flush(stdout)
    # per-seam straight windows n=1, n=2 (tension gate: each must give 2*s2)
    W0_1 = window_sweep(C4, [T4], [T4], [T4], [T4], [[a4]])
    W0_2 = window_sweep(C4, [T4,T4], [T4,T4], [T4,T4], [T4,T4], [[a4,a4],[a4,a4]])
    res = Dict{String,Tuple{Float64,Float64}}()
    for (nm, Td, aGu) in (("CA", TdCA, aCAu), ("CB", TdCB, aCBu))
        Ws1 = window_sweep(C4, [T4], [flipT(Td)], [T4], [Td], [[aGu]])
        Ws2 = window_sweep(C4, [T4,T4], [flipT(Td),T4], [T4,T4], [Td,T4],
                           [[aGu,a4],[aGu,a4]])
        t = -(log(abs(Ws2/W0_2)) - log(abs(Ws1/W0_1)))
        @printf("  seam %s: tension=%.8f vs 2*s2win=%.8f (rel %.1e)\n",
                nm, t, 2s2win, abs(t-2s2win)/(2s2win))
        res[nm] = (Ws1, Ws2)
    end
    # vertical AB straight: ray through center column; n=1: center r-dressed,
    # TN=TdAB, TS=flip(TdAB); n=2: core column x=1 r-dressed both rows
    WsAB1 = window_sweep(C4, [TdAB], [T4], [flipT(TdAB)], [T4], [[aABr]])
    WsAB2 = window_sweep(C4, [TdAB,T4], [T4,T4], [flipT(TdAB),T4], [T4,T4],
                         [[aABr,aABr], [a4,a4]])
    tAB = -(log(abs(WsAB2/W0_2)) - log(abs(WsAB1/W0_1)))
    @printf("  seam AB: tension=%.8f vs 2*s2win=%.8f (rel %.1e)\n",
            tAB, 2s2win, abs(tAB-2s2win)/(2s2win))
    # THE Y WINDOW (n=2 core; junction at the core's central bond-vertex):
    #   core: s11 = G_CA on u; s21 = G_CB on u; s12 = G_AB on r; s22 plain
    #   edges: TW1=TdCA, TE1=flip(TdCB), TN1=TdAB, rest plain
    WY = window_sweep(C4, [TdAB,T4], [flipT(TdCB),T4], [T4,T4], [TdCA,T4],
                      [[aCAu,aABr], [aCBu,a4]])
    kappaY = log(abs(WY)) - 0.5*(log(abs(res["CA"][2])) + log(abs(res["CB"][2])) +
                                  log(abs(WsAB2))) + 0.5*log(abs(W0_2))
    tSJ = -(kappaY - 2*kappaA)
    @printf("  kappa_Y = %+.8f   =>  tildeS_J = -(kappa_Y - 2 kappa_A) = %+.8f\n",
            kappaY, tSJ)
    tsr = refval(TSREF, beta)
    tsr !== nothing && @printf("  [target: %s]\n", tsr)
    @printf("  SUMMARY beta=%.8f chi=%d s2=%.8f xi2=%.4f xi4=%.4f kA=%+.8f kY=%+.8f tSJ=%+.8f it2=%d it4=%d dr2=%.1e dr4=%.1e\n",
            beta, χenv, s2win, xi2, xi4, kappaA, kappaY, tSJ, it2, it4, dr2, dr4)
    flush(stdout)
end
println("DONE CTMRG-STILDE")
