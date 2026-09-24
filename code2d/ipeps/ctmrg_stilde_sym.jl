# ctmrg_stilde_sym.jl -- REPLICA-SYMMETRIZED defect CTMRG for tildeS.
# =============================================================================
# Root fix for the critical-point branch instability of ctmrg_stilde.jl:
# the truncation there picks chi eigenvectors with no regard for the layer-
# permutation symmetry S_k of the k-replica bond space -> at beta_c the cut
# lands inside exactly-degenerate permutation multiplets, replica symmetry
# breaks spontaneously, and the iteration self-consistently locks into broken
# fixed points (three-seam tensions unequal, kappa_Y O(1) wrong).
# Fix (this script), mirroring how SPATIAL C4v symmetry is already hardwired
# (single C, single T, forced C=C'):
#   (1) group-average the grown corner over U_g = R_g (x) P_g each step
#       (projects out accumulated symmetry-breaking noise; multiplets stay
#        EXACTLY degenerate),
#   (2) never cut inside a degenerate multiplet (whole-block truncation),
#   (3) track the group action R_g on the compressed basis: R_g' = Z' U_g Z,
#       polar-orthogonalized; leakage |R'R'^T - 1| gates subspace invariance.
# Defect edges Td are NOT group-averaged (each is covariant only under the
# stabilizer of its pairing); they are grown with the same projector as before.
# Plain T IS averaged (it is fully symmetric in the exact theory).
# Env: CSBETA, CSCHI (colon list), CSNMAX, CSDEG (multiplet tol, default 1e-8),
#      CSSLACK (default 8).
# =============================================================================
include(joinpath(@__DIR__, "yvx_core.jl"))

const CSBETA  = parse(Float64, get(ENV, "CSBETA", "0.30"))
const CSCHIS  = haskey(ENV, "CSCHI") ? parse.(Int, split(ENV["CSCHI"], ":")) : [24]
const CSNMAX  = parse(Int, get(ENV, "CSNMAX", "20000"))
const CSDEG   = parse(Float64, get(ENV, "CSDEG", "1e-8"))
const CSSLACK = parse(Int, get(ENV, "CSSLACK", "8"))
# CSFLIP=1: extend the twirl group S_k -> Z2 x S_k (global spin flip = invert all
# k bits of the bond factor). Kills Z2-broken (boundary-pinned, ferro-like)
# symmetric fixed points that S_k alone cannot exclude.
const CSFLIP  = parse(Int, get(ENV, "CSFLIP", "0"))

# ---------------- group machinery ----------------
"all permutations of 1:n (image lists)."
function all_perms(n)
    n == 1 && return [[1]]
    out = Vector{Vector{Int}}()
    for p in all_perms(n - 1), pos in 1:n
        q = copy(p); insert!(q, pos, n)
        push!(out, q)
    end
    out
end

"index map p of U_sigma on the 2^k bond factor (layer j = bit j-1; sigma sends
 layer j's spin to layer sigma[j]):  (U_sigma v)[p[i]] = v[i]."
function perm_p(sigma::Vector{Int})
    k = length(sigma); dd = 2^k
    p = Vector{Int}(undef, dd)
    for idx in 0:dd-1
        nidx = 0
        for j in 1:k
            nidx |= ((idx >> (j - 1)) & 1) << (sigma[j] - 1)
        end
        p[idx+1] = nidx + 1
    end
    p
end

"rows of M live on (chi_o fast) x (dd slow); return (R (x) P_sigma) * M.
 ip must be invperm(perm_p(sigma)):  (Uv)[j] = v[ip[j]] on the dd factor."
function applyU_rows(M::AbstractMatrix, R::AbstractMatrix, ip::Vector{Int},
                     χo::Int, dd::Int)
    n = size(M, 2)
    B = reshape(R * reshape(M, χo, dd * n), χo, dd, n)
    reshape(B[:, ip, :], χo * dd, n)
end

"group-average of Cbig over U_g . Cbig . U_g' (exact symmetrization)."
function sym_Cbig(Cbig, Rs, ips, χo, dd)
    S = zeros(size(Cbig))
    for (R, ip) in zip(Rs, ips)
        UC = applyU_rows(Cbig, R, ip, χo, dd)
        S .+= applyU_rows(Matrix(UC'), R, ip, χo, dd)'
    end
    S ./= length(Rs)
    (S + S') / 2
end

"group-average of the plain edge T[w,e,p] over (R,R,P)."
function sym_T(T, Rs, ips)
    χ = size(T, 1); dd = size(T, 3)
    S = zeros(size(T))
    for (R, ip) in zip(Rs, ips)
        A = reshape(R * reshape(T, χ, χ * dd), χ, χ, dd)                 # w
        B = permutedims(A, (2, 1, 3))
        B = reshape(R * reshape(B, χ, χ * dd), χ, χ, dd)                 # e
        B = permutedims(B, (2, 1, 3))
        S .+= B[:, :, ip]                                                # p
    end
    S ./ length(Rs)
end

# ---------------- shared machinery (verbatim from ctmrg_stilde.jl) ----------
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

function window_sweep(C, TNs, TEs, TSs, TWs, cols)
    n = length(TNs); d = size(cols[1][1], 1); χ = size(C, 1)
    dn_ = d^n
    v = zeros(χ, ntuple(_ -> d, n)..., χ)
    for I in CartesianIndices(ntuple(_ -> d, n))
        B = C
        for y in 1:n; B = B * TWs[y][:, :, I[y]]; end
        B = B * C
        @inbounds @views v[:, I, :] .= B
    end
    for x in 1:n
        TSf = flipT(TSs[x]); TNx = TNs[x]
        M = reshape(v, χ*dn_, χ) * reshape(TNx, χ, χ*d)
        t = reshape(M, χ, ntuple(_->d,n)..., χ, d)
        if n == 1
            s1 = cols[x][1]
            tp = reshape(permutedims(t, (1,3,2,4)), χ*χ, d*d)
            S1 = reshape(permutedims(s1, (2,1,3,4)), d*d, d*d)
            q  = reshape(tp * S1, χ, χ, d, d)
            qp = reshape(permutedims(q, (2,4,1,3)), χ*d, χ*d)
            TSm = reshape(permutedims(TSf, (1,3,2)), χ*d, χ)
            r  = reshape(qp * TSm, χ, d, χ)
            v  = permutedims(r, (3,2,1))
        else
            s1 = cols[x][1]; s2 = cols[x][2]
            tp = reshape(permutedims(t, (1,2,4,3,5)), χ*d*χ, d*d)
            S2 = reshape(permutedims(s2, (2,1,3,4)), d*d, d*d)
            q  = reshape(tp * S2, χ, d, χ, d, d)
            qp = reshape(permutedims(q, (1,3,5,2,4)), χ*χ*d, d*d)
            S1 = reshape(permutedims(s1, (2,1,3,4)), d*d, d*d)
            w  = reshape(qp * S1, χ, χ, d, d, d)
            wp = reshape(permutedims(w, (2,3,5,1,4)), χ*d*d, χ*d)
            TSm = reshape(permutedims(TSf, (1,3,2)), χ*d, χ)
            r  = reshape(wp * TSm, χ, d, d, χ)
            v  = permutedims(r, (4,3,2,1))
        end
    end
    z = 0.0
    for I in CartesianIndices(ntuple(_ -> d, n))
        B = C
        for y in n:-1:1; B = B * TEs[y][:, :, I[y]]; end
        B = B * C
        z += sum(transpose(@view v[:, I, :]) .* B)
    end
    z
end

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

# ---------------- symmetrized CTMRG step ----------------
"grow corner, group-symmetrize, whole-multiplet truncate, update R_g."
function ctm_step_Zsym(C, T, a, χmax, Rs, ips)
    χ = size(C,1); dd = size(a,1); cd = χ*dd
    CTl = reshape(C*reshape(T, χ, χ*dd), χ, χ, dd)
    res = reshape(reshape(permutedims(T,(2,1,3)), χ, χ*dd)' * reshape(CTl, χ, χ*dd),
                  χ, dd, χ, dd)
    Cb  = reshape(reshape(permutedims(res,(1,3,2,4)), χ*χ, dd*dd) * reshape(a, dd*dd, dd*dd),
                  χ, χ, dd, dd)
    Cbig = reshape(permutedims(Cb,(1,4,2,3)), cd, cd)
    Cbig = sym_Cbig(Cbig, Rs, ips, χ, dd)            # exact group projection
    F = eigen(Symmetric(Cbig)); ps = sortperm(abs.(F.values), rev=true)
    # whole-multiplet cut: multiplets are exactly degenerate after projection,
    # so a tight CSDEG suffices; extend to the gap, shrink if slack exhausted.
    χcut = min(χmax, cd)
    if χcut < cd
        deg(i) = abs(F.values[ps[i+1]]) > (1-CSDEG)*abs(F.values[ps[i]])
        hi = min(χmax + CSSLACK, cd)
        c1 = χcut
        while c1 < hi && deg(c1); c1 += 1; end
        if c1 == hi && c1 < cd && deg(c1)
            c1 = χcut; lo = max(χmax ÷ 2, 2)
            while c1 > lo && deg(c1); c1 -= 1; end
        end
        χcut = c1
    end
    rcut = χcut < cd ? abs(F.values[ps[χcut+1]])/abs(F.values[ps[χcut]]) : 0.0
    # mid=true: the final cut STILL splits a degenerate pair (slack exhausted both
    # ways) -- the one remaining silent-symmetry-breaking path; counted and reported.
    mid = χcut < cd && abs(F.values[ps[χcut+1]]) > (1-CSDEG)*abs(F.values[ps[χcut]])
    Z = F.vectors[:, ps[1:χcut]]
    Cn = Z' * Cbig * Z; Cn = (Cn + Cn')/2; Cn ./= maximum(abs, Cn)
    # update the group action on the kept basis; leakage gates invariance
    Rs2 = Matrix{Float64}[]; leak = 0.0
    for (R, ip) in zip(Rs, ips)
        Rp = Z' * applyU_rows(Z, R, ip, χ, dd)
        leak = max(leak, norm(Rp*Rp' - I))
        S = svd(Rp); push!(Rs2, S.U * S.Vt)          # polar re-orthogonalization
    end
    (Cn, Z, Rs2, leak, rcut, mid)
end

"index maps of the twirl group on the dd=2^k bond factor: S_k perms, and with
 CSFLIP=1 also each composed with the global bit-flip (idx -> dd-1-idx)."
function group_maps(perms::Vector{Vector{Int}}, dd::Int)
    maps = [perm_p(σ) for σ in perms]
    if CSFLIP == 1
        fl = collect(dd:-1:1)                       # flip all bits
        append!(maps, [fl[p] for p in maps])        # (flip o U_sigma) index maps
    end
    maps
end

function ctmrg_defect_sym(aT, adrs::Vector, perms::Vector{Vector{Int}};
                          χmax, nmax=CSNMAX, tol=1e-12, tag="")
    dd = size(aT, 1)
    C, T = init_CT(aT)
    maps = group_maps(perms, dd)
    ips = [invperm(p) for p in maps]
    # R0 = the map's permutation matrix on the init env index (chi0 = dd). NOTE
    # column selection: I[:, p] has W[p[i], i] = 1, i.e. (Wv)[p[i]] = v[i] = our
    # U convention; row selection I[p, :] would be U^{-1} and mispair the factors.
    Rs  = [Matrix{Float64}(I, dd, dd)[:, p] for p in maps]
    Tds = [copy(T) for _ in adrs]
    sold = Float64[]; itdone = nmax; drift = NaN; leak = NaN; rcut = NaN; nmid = 0
    for it in 1:nmax
        Cn, Z, Rs2, leak, rcut, mid = ctm_step_Zsym(C, T, aT, χmax, Rs, ips)
        nmid += mid
        Tn  = grow_T(T, aT, Z)
        Tn  = sym_T(Tn, Rs2, ips)                    # keep plain edge exactly symmetric
        Tdn = [grow_T(Td, adr, Z) for (Td, adr) in zip(Tds, adrs)]
        C, T, Tds, Rs = Cn, Tn, Tdn, Rs2
        sv = sort(abs.(eigvals(Symmetric((C+C')/2))), rev=true); sv ./= sv[1]
        if length(sv) == length(sold)
            drift = maximum(abs.(sv .- sold))
            if drift < tol
                itdone = it; break
            end
        end
        sold = sv
        if it % 2000 == 0
            @printf("    [%s it=%d drift=%.2e leak=%.1e]\n", tag, it, drift, leak); flush(stdout)
        end
    end
    # representation-closure gate: R_g R_h vs R_{gh} for a non-trivial pair
    i = 2; j = min(3, length(perms))
    στ = [perms[i][perms[j][x]] for x in 1:length(perms[i])]
    kk = findfirst(==(στ), perms)
    reperr = norm(Rs[i] * Rs[j] - Rs[kk])
    nmid > 0 && @printf("    [%s WARNING: %d mid-multiplet cuts (slack exhausted)]\n", tag, nmid)
    @printf("    [%s chi_fin=%d rcut=%.6f leak=%.2e reperr=%.2e nmid=%d]\n",
            tag, size(C,1), rcut, leak, reperr, nmid)
    (C, T, Tds, itdone, drift, leak)
end

# ---------------- convention self-gates ----------------
"verify P index conventions: representation property + a-invariance + G-conjugation."
function convention_gates(a4, GCA, GCB, GAB)
    P4 = all_perms(4)
    # representation property on the 16-dim factor
    e1 = 0.0
    for σ in P4[1:6], τ in P4[1:6]
        pσ, pτ = perm_p(σ), perm_p(τ)
        στ = [σ[τ[j]] for j in 1:4]                  # (sigma.tau)[j] = sigma[tau[j]]
        v = randn(16)
        w1 = zeros(16); w1[pτ] = v; w2 = zeros(16); w2[pσ] = w1
        w3 = zeros(16); w3[perm_p(στ)] = v
        e1 = max(e1, norm(w2 - w3))
    end
    # a4 invariance: permute all four legs' dd factors
    e2 = 0.0
    for σ in (P4[2], P4[8], P4[15])
        ip = invperm(perm_p(σ))
        e2 = max(e2, norm(a4[ip, ip, ip, ip] - a4))
    end
    # seam conjugation: U_(34) G_CA U_(34)' = G_CB ; U_(23) G_AB U_(23)' = G_CA
    U(σ) = Matrix{Float64}(I, 16, 16)[:, perm_p(σ)]
    e3 = max(norm(U([1,2,4,3]) * GCA * U([1,2,4,3])' - GCB),
             norm(U([1,3,2,4]) * GAB * U([1,3,2,4])' - GCA))
    (e1, e2, e3)
end

# ---------------- main ----------------
beta = CSBETA
@printf("CTMRG-STILDE-SYM  beta=%.8f  (group-projected defect CTMRG)\n", beta)
a, _ = ising_tensors(beta)
a2 = build_a2(a)
G2 = Matrix(seam_G(2, beta, 'C', 'A', [1,2], [1,2], PSWAP2))
a2Gu = dress(a2, G2, :u)
a4 = build_a2(a2)
GCA = Matrix(seam_G(4, beta, 'C', 'A', SA4, SB4, SC4))
GCB = Matrix(seam_G(4, beta, 'C', 'B', SA4, SB4, SC4))
GAB = Matrix(seam_G(4, beta, 'A', 'B', SA4, SB4, SC4))
g1, g2, g3 = convention_gates(a4, GCA, GCB, GAB)
@printf("gate(P rep)=%.1e  gate(a4 inv)=%.1e  gate(G conj)=%.1e\n", g1, g2, g3)
PERM2 = all_perms(2)
PERM4 = all_perms(4)

for χenv in CSCHIS
    println("-"^76)
    @printf("chi_env=%d\n", χenv)
    # ===== k=2 block (S2-symmetrized) =====
    adr2 = dress(a2, G2, :r)
    C2, T2, Tds2, it2, dr2, lk2 = ctmrg_defect_sym(a2, [adr2], PERM2; χmax=χenv, tag="k2")
    Td2 = Tds2[1]
    xi2, _ = xi_from_T(T2)
    @printf("  conv(k2): it=%d drift=%.1e leak=%.1e xi2=%.4f\n", it2, dr2, lk2, xi2)
    Z0 = window_sweep(C2, [T2], [T2], [T2], [T2], [[a2]])
    Zb1 = window_sweep(C2, [Td2], [T2], [T2], [Td2], [[a2Gu]])
    Zs1 = window_sweep(C2, [T2], [flipT(Td2)], [T2], [Td2], [[a2Gu]])
    Zs2 = window_sweep(C2, [T2,T2], [flipT(Td2),T2], [T2,T2], [Td2,T2],
                       [[a2Gu,a2], [a2Gu,a2]])
    Z02 = window_sweep(C2, [T2,T2], [T2,T2], [T2,T2], [T2,T2], [[a2,a2],[a2,a2]])
    s2win = -(log(abs(Zs2/Z02)) - log(abs(Zs1/Z0)))
    kappaA = log(abs(Zb1/Zs1))
    @printf("  s2(chi)=%.8f   kappa_A=%+.8f\n", s2win, kappaA)
    # ===== k=4 block (S4-symmetrized) =====
    aCAu = dress(a4, GCA, :u); aCBu = dress(a4, GCB, :u); aABr = dress(a4, GAB, :r)
    adrs4 = [dress(a4, GCA, :r), dress(a4, GCB, :r), dress(a4, GAB, :r)]
    C4, T4, Tds4, it4, dr4, lk4 = ctmrg_defect_sym(a4, adrs4, PERM4; χmax=χenv, tag="k4")
    TdCA, TdCB, TdAB = Tds4
    xi4, _ = xi_from_T(T4)
    @printf("  conv(k4): it=%d drift=%.1e leak=%.1e xi4=%.4f\n", it4, dr4, lk4, xi4)
    W0_1 = window_sweep(C4, [T4], [T4], [T4], [T4], [[a4]])
    W0_2 = window_sweep(C4, [T4,T4], [T4,T4], [T4,T4], [T4,T4], [[a4,a4],[a4,a4]])
    res = Dict{String,Tuple{Float64,Float64}}()
    ts = Float64[]
    for (nm, Td, aGu) in (("CA", TdCA, aCAu), ("CB", TdCB, aCBu))
        Ws1 = window_sweep(C4, [T4], [flipT(Td)], [T4], [Td], [[aGu]])
        Ws2 = window_sweep(C4, [T4,T4], [flipT(Td),T4], [T4,T4], [Td,T4],
                           [[aGu,a4],[aGu,a4]])
        t = -(log(abs(Ws2/W0_2)) - log(abs(Ws1/W0_1)))
        @printf("  seam %s: tension=%.8f vs 2*s2win=%.8f (rel %.1e)\n",
                nm, t, 2s2win, abs(t-2s2win)/(2s2win))
        res[nm] = (Ws1, Ws2); push!(ts, t)
    end
    WsAB1 = window_sweep(C4, [TdAB], [T4], [flipT(TdAB)], [T4], [[aABr]])
    WsAB2 = window_sweep(C4, [TdAB,T4], [T4,T4], [flipT(TdAB),T4], [T4,T4],
                         [[aABr,aABr], [a4,a4]])
    tAB = -(log(abs(WsAB2/W0_2)) - log(abs(WsAB1/W0_1)))
    @printf("  seam AB: tension=%.8f vs 2*s2win=%.8f (rel %.1e)\n",
            tAB, 2s2win, abs(tAB-2s2win)/(2s2win))
    push!(ts, tAB)
    asym = (maximum(ts) - minimum(ts)) / maximum(ts)
    WY = window_sweep(C4, [TdAB,T4], [flipT(TdCB),T4], [T4,T4], [TdCA,T4],
                      [[aCAu,aABr], [aCBu,a4]])
    kappaY = log(abs(WY)) - 0.5*(log(abs(res["CA"][2])) + log(abs(res["CB"][2])) +
                                  log(abs(WsAB2))) + 0.5*log(abs(W0_2))
    tSJ = -(kappaY - 2*kappaA)
    @printf("  kappa_Y = %+.8f   =>  tildeS_J = %+.8f\n", kappaY, tSJ)
    @printf("  SUMMARY-SYM beta=%.8f chi=%d s2=%.8f xi2=%.4f xi4=%.4f kA=%+.8f kY=%+.8f tSJ=%+.8f it2=%d it4=%d dr2=%.1e dr4=%.1e lk2=%.1e lk4=%.1e asym=%.2e\n",
            beta, χenv, s2win, xi2, xi4, kappaA, kappaY, tSJ, it2, it4, dr2, dr4, lk2, lk4, asym)
    flush(stdout)
end
println("DONE CTMRG-STILDE-SYM")
