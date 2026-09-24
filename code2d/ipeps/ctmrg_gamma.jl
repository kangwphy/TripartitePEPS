# ctmrg_gamma.jl (v2) -- gamma_A from CTMRG defect environments.
# =============================================================================
# v2 upgrades over v1 (v1 archived in logs2d/ctmg_23715176-178.out):
#  * SYNCHRONIZED defect-CTMRG: T and Td co-evolve,每 step 用同一个 fresh
#    projector (v1 iterated Td with a frozen projector and drifted:
#    ||Td(untwisted)-T|| up to 0.9).  Now Td(untwisted) == T by construction.
#  * n=2 window: second point of the R-progression AND the chirality gate:
#      s2_win = -ln[(Zs(2)/Z0(2))/(Zs(1)/Z0(1))]  must hit the certified s2
#    for the correct dressed slot (l vs r); the wrong slot fails.
#  * beta=0.60 bmps truth extended to L=16 (ordered-phase slow transient).
# The L-shaped (point-hinge) route is UNTOUCHED and remains the baseline.
# Env: CGBETA (default 0.60), CGCHI (colon list, default 24:48:96).
# =============================================================================
include(joinpath(@__DIR__, "yvx_core.jl"))

const CGBETA = parse(Float64, get(ENV, "CGBETA", "0.60"))
const CGCHIS = haskey(ENV, "CGCHI") ? parse.(Int, split(ENV["CGCHI"], ":")) : [24, 48, 96]

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

function dress(a2, G, slot::Symbol)
    D = size(a2, 1); out = zeros(D, D, D, D)
    @inbounds for u in 1:D, l in 1:D, dn in 1:D, r in 1:D, x in 1:D
        if slot === :u
            out[u,l,dn,r] += G[u,x] * a2[x,l,dn,r]
        elseif slot === :l
            out[u,l,dn,r] += G[l,x] * a2[u,x,dn,r]
        else
            out[u,l,dn,r] += G[r,x] * a2[u,l,dn,x]
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
    F = eigen(Symmetric(Cbig)); p = sortperm(abs.(F.values), rev=true)[1:min(χmax, cd)]
    Z = F.vectors[:, p]
    Cn = Z' * Cbig * Z; Cn = (Cn + Cn')/2; Cn ./= maximum(abs, Cn)
    (Cn, Z)
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

"synchronized defect-CTMRG: plain (C,T) + a list of co-evolved defect edges."
function ctmrg_defect(a2, adrs::Vector; χmax, nmax=20000, tol=1e-12)
    C, T = init_CT(a2)
    Tds = [copy(T) for _ in adrs]
    sold = Float64[]; lastdT = Inf
    for it in 1:nmax
        Cn, Z = ctm_step_Z(C, T, a2, χmax)
        Tn  = grow_T(T, a2, Z)
        Tdn = [grow_T(Td, adr, Z) for (Td, adr) in zip(Tds, adrs)]
        lastdT = maximum(maximum(abs, Tdn[i] .- (size(Tdn[i]) == size(Tds[i]) ? Tds[i] : Tdn[i]))
                         for i in eachindex(Tds); init=0.0)
        C, T, Tds = Cn, Tn, Tdn
        sv = sort(abs.(eigvals(Symmetric((C+C')/2))), rev=true); sv ./= sv[1]
        if length(sv) == length(sold) && maximum(abs.(sv .- sold)) < tol && lastdT < 1e-9
            break
        end
        sold = sv
    end
    (C, T, Tds, lastdT)
end

flipT(T) = permutedims(T, (2,1,3))

"n=1 window."
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

"2x2 core block: core[u1,u2, r2,r1, d2,d1, l1,l2] with internal bonds summed.
 sites s[x][y], x=1:2 west->east, y=1:2 south->north; sxy[u,l,dn,r]."
function core22(s11, s21, s12, s22)
    d = size(s11, 1)
    out = zeros(d,d, d,d, d,d, d,d)
    @inbounds for u1 in 1:d, u2 in 1:d, r2 in 1:d, r1 in 1:d,
                  d2 in 1:d, d1 in 1:d, l1 in 1:d, l2 in 1:d
        acc = 0.0
        for h1 in 1:d, h2 in 1:d, v1 in 1:d, v2 in 1:d
            # h1: (1,1)-(2,1) bond;  h2: (1,2)-(2,2);  v1: (1,1)-(1,2); v2: (2,1)-(2,2)
            acc += s11[v1,l1,d1,h1] * s21[v2,h1,d2,r1] *
                   s12[u1,l2,v1,h2] * s22[u2,h2,v2,r2]
        end
        out[u1,u2, r2,r1, d2,d1, l1,l2] = acc
    end
    out
end

"n=2 window: edge strings of two T's; ring N(x1,x2) E(y2,y1) S(x2,x1) W(y1,y2)."
function window2(C, TN1, TN2, TE2, TE1, TS2, TS1, TW1, TW2, core)
    d = size(TN1, 3)
    z = 0.0
    @inbounds for u1 in 1:d, u2 in 1:d, r2 in 1:d, r1 in 1:d,
                  d2 in 1:d, d1 in 1:d, l1 in 1:d, l2 in 1:d
        c = core[u1,u2, r2,r1, d2,d1, l1,l2]
        abs(c) < 1e-300 && continue
        z += c * tr(C*TN1[:,:,u1]*TN2[:,:,u2] * C*TE2[:,:,r2]*TE1[:,:,r1] *
                    C*TS2[:,:,d2]*TS1[:,:,d1] * C*TW1[:,:,l1]*TW2[:,:,l2])
    end
    z
end

const S2REF = Dict(0.30 => 0.05138867, 0.35 => 0.07242187, 0.60 => 0.00164704)
const GTRUTH = Dict(0.30 => -0.003363, 0.35 => -0.0088)
const GHINGE = Dict(0.30 => -0.000135, 0.35 => +0.000276)

beta = CGBETA
@printf("CTMRG-GAMMA v2  beta=%.4f\n", beta)
a, _ = ising_tensors(beta)
a2 = build_a2(a)
G2 = Matrix(seam_G(2, beta, 'C', 'A', [1,2], [1,2], PSWAP2))
G2id = Matrix(seam_G(2, beta, 'C', 'A', [1,2], [1,2], [1,2]))
@printf("gate(0): ||G2(untwisted)-Id|| = %.2e\n", norm(G2id - I))
a2Gu = dress(a2, G2, :u)

if !haskey(GTRUTH, beta)
    for L in (8, 10, 12, 14, 16)
        lZ1  = logZ_network(L, L, 1, beta, PID, PID, PID, 256)
        lZ2A = logZ_network(L, L, 2, beta, PSWAP2, [1,2], [1,2], 256)
        lZ2C = logZ_network(L, L, 2, beta, [1,2], [1,2], PSWAP2, 256)
        @printf("bmps truth: L=%2d  cornerA = %+.8f\n", L, -(lZ2A - lZ2C))
        flush(stdout)
    end
end

for χenv in CGCHIS
    println("-"^76)
    @printf("chi_env=%d\n", χenv)
    adrs = [dress(a2, G2, :l), dress(a2, G2, :r), dress(a2, G2id, :l)]
    C, T, Tds, dT = ctmrg_defect(a2, adrs; χmax=χenv)
    Tdl, Tdr, Tid = Tds
    @printf("  gate(i): ||Td(untwisted)-T|| = %.2e   (Td co-evolution residual %.1e)\n",
            maximum(abs, Tid .- T), dT)
    for (nm, Td) in (("l", Tdl), ("r", Tdr))
        # n=1 windows
        Z0_1 = window1(C, T, T, T, T, a2)
        Zs_1 = window1(C, T, flipT(Td), T, Td, a2Gu)
        Zb_1 = window1(C, Td, T, T, Td, a2Gu)
        # n=2 windows
        cP  = core22(a2, a2, a2, a2)
        # straight: seam between the two core rows: dress u of (1,1),(2,1)
        a2u = a2Gu
        cS  = core22(a2u, a2u, a2, a2)
        # bent: dress u of (1,1) + r of (1,2) [turn]; then north col-1 edge = Td
        a2r = dress(a2, G2, :r)
        cB  = core22(a2u, a2, a2r, a2)
        Z0_2 = window2(C, T,T, T,T, T,T, T,T, cP)
        # straight: only the seam-adjacent y=1 strings are dressed
        Zs_2 = window2(C, T,T, T,flipT(Td), T,T, Td,T, cS)
        # bent: west y=1 and north x=1 strings dressed
        Zb_2 = window2(C, Td,T, T,T, T,T, Td,T, cB)
        # chirality/wiring gate: one extra seam column between n=2 and n=1
        s2win = -(log(abs(Zs_2/Z0_2)) - log(abs(Zs_1/Z0_1)))
        g1 = -log(abs(Zb_1/Zs_1)); g2 = -log(abs(Zb_2/Zs_2))
        @printf("  slot=%s : s2_win=%.8f (cert %.8f, rel %.1e) | -ln(gA): R=1 %+.8f  R=2 %+.8f\n",
                nm, s2win, S2REF[beta], abs(s2win-S2REF[beta])/S2REF[beta], g1, g2)
    end
    if haskey(GTRUTH, beta)
        @printf("  [truth %+.6f | point-hinge %+.6f]\n", GTRUTH[beta], GHINGE[beta])
    end
    flush(stdout)
end
println("DONE CTMRG-GAMMA")
