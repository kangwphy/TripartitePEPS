# rk_ctmrg.jl — symmetric (C4v) CTMRG for the 2D classical Ising model (= the RK-state
# single-copy environment, D=2 exact). Gives the fixed-point corner C(χ,χ) and edge
# T(χ,χ,2), the per-site free energy (Baxter ratio), spontaneous magnetization,
# finite-entanglement ξ(χ), and the CTM entanglement entropy. Validated against Onsager.
# STATUS: C and T themselves are Onsager-certified and remain the single-copy
# environment for all routes. (The CTM-ring junction CLOSURE that consumed them
# was withdrawn 2026-07-28 — see trash/rk_junction_ctm.jl and the REPORT — but
# that defect was in the closure geometry, not in this file.)
using LinearAlgebra

"Ising bulk tensor a[u,l,d,r] and σ-inserted am[...] via the symmetric √W split; optional field h."
function ising_tensors(β; h=0.0)
    c = sqrt(2cosh(β)); s = sqrt(2sinh(β))
    Q = [ (c+s)/2  (c-s)/2 ;
          (c-s)/2  (c+s)/2 ]              # symmetric √W, Q[k,σ], Q*Q = W
    σ = (1.0, -1.0)
    a  = zeros(2,2,2,2); am = zeros(2,2,2,2)
    @inbounds for u in 1:2, l in 1:2, dn in 1:2, r in 1:2
        sa = 0.0; sm = 0.0
        for k in 1:2
            w = Q[k,u]*Q[k,l]*Q[k,dn]*Q[k,r]*exp(β*h*σ[k])
            sa += w; sm += σ[k]*w
        end
        a[u,l,dn,r] = sa; am[u,l,dn,r] = sm
    end
    return a, am
end

"init C[e,s], T[w,e,p] from a (χ=d=2 start with correct symmetry)."
function init_CT(a)
    C = permutedims(dropdims(sum(a, dims=(1,2)), dims=(1,2)), (2,1))   # [r,dn]->[e,s]
    T = permutedims(dropdims(sum(a, dims=1),     dims=1),     (1,3,2)) # [l,dn,r]->[w,e,p]
    return (C+C')/2, (T+permutedims(T,(2,1,3)))/2
end

"one symmetric CTMRG move: grow C,T by absorbing one a, truncate to χmax."
function ctm_step(C, T, a, χmax)
    χ = size(C,1); dd = size(a,1); cd = χ*dd
    # ---- grow corner  Cbig[(w,rb),(s,db)] ----
    CTl = reshape(C*reshape(T, χ, χ*dd), χ, χ, dd)                              # [i,s,n]
    res = reshape(reshape(permutedims(T,(2,1,3)), χ, χ*dd)' * reshape(CTl, χ, χ*dd),
                  χ, dd, χ, dd)                                                 # [w,m,s,n]
    Cb  = reshape(reshape(permutedims(res,(1,3,2,4)), χ*χ, dd*dd) * reshape(a, dd*dd, dd*dd),
                  χ, χ, dd, dd)                                                 # [w,s,db,rb]
    Cbig = reshape(permutedims(Cb,(1,4,2,3)), cd, cd); Cbig = (Cbig + Cbig')/2
    # ---- grow edge  Tbig[(w,l),(e,r),db] ----
    res5 = reshape(reshape(permutedims(T,(3,1,2)), dd, χ*χ)' * reshape(a, dd, dd*dd*dd),
                   χ, χ, dd, dd, dd)                                            # [w,e,l,db,r]
    Tbig = reshape(permutedims(res5,(1,3,2,5,4)), cd, cd, dd)
    # ---- projector = leading-χ eigenvectors of the symmetric enlarged corner ----
    F = eigen(Symmetric(Cbig)); p = sortperm(abs.(F.values), rev=true)[1:min(χmax, cd)]
    Z = F.vectors[:, p]; χn = size(Z,2)
    # ---- renormalize ----
    Cn = Z' * Cbig * Z
    Tmid = reshape(Z' * reshape(Tbig, cd, cd*dd), χn, cd, dd)
    Tn = permutedims(reshape(Z' * reshape(permutedims(Tmid,(2,1,3)), cd, χn*dd), χn, χn, dd), (2,1,3))
    Cn = (Cn + Cn')/2;                     Cn ./= maximum(abs, Cn)
    Tn = (Tn + permutedims(Tn,(2,1,3)))/2; Tn ./= maximum(abs, Tn)
    return Cn, Tn
end

"run CTMRG to a fixed point (corner-spectrum convergence). Returns (C,T,iters)."
function ctmrg(a; χmax=32, nmax=20000, tol=1e-12)
    C, T = init_CT(a); sold = Float64[]
    it = 0
    for outer it in 1:nmax
        C, T = ctm_step(C, T, a, χmax)
        sv = sort(abs.(eigvals(Symmetric((C+C')/2))), rev=true); sv ./= sv[1]
        if length(sv) == length(sold) && maximum(abs.(sv .- sold)) < tol
            break
        end
        sold = sv
    end
    return C, T, it
end

"log Z per site via the balanced Baxter ratio κ = z1·ξC/Z2² (= −βf)."
function logZ_per_site(C, T, a)
    d = size(a,1)
    A = [C*T[:,:,p] for p in 1:d]
    z1 = 0.0
    @inbounds for u in 1:d, l in 1:d, dn in 1:d, r in 1:d
        z1 += a[u,l,dn,r]*tr(A[u]*A[r]*A[dn]*A[l])
    end
    ξC = tr(C*C*C*C)
    Z2 = sum(tr((C*T[:,:,p]*C)^2) for p in 1:d)          # 4 corners + 2 opposite edges
    return log(z1) + log(ξC) - 2*log(Z2)
end

"⟨σ⟩ = z1m/z1 (build a,am with a small field h to break Z2 for β>β_c)."
function magnetization(C, T, a, am)
    d = size(a,1)
    A = [C*T[:,:,p] for p in 1:d]
    z1 = 0.0; z1m = 0.0
    @inbounds for u in 1:d, l in 1:d, dn in 1:d, r in 1:d
        w = tr(A[u]*A[r]*A[dn]*A[l])
        z1 += a[u,l,dn,r]*w; z1m += am[u,l,dn,r]*w
    end
    return abs(z1m/z1)
end

"finite-entanglement correlation length from the edge (boundary-MPS) transfer spectrum."
function corr_length(T)
    χ = size(T,1); d = size(T,3)
    E = zeros(χ*χ, χ*χ)
    @inbounds for a1 in 1:χ, ap in 1:χ, b in 1:χ, bp in 1:χ
        s = 0.0
        for p in 1:d; s += T[a1,b,p]*T[ap,bp,p]; end
        E[(a1-1)*χ+ap, (b-1)*χ+bp] = s
    end
    ε = sort(abs.(eigvals(E)), rev=true)
    return 1/log(ε[1]/ε[2])
end

"CTM (half-plane) entanglement entropy, ρ ∝ C⁴."
function ctm_entropy(C)
    λ = abs.(eigvals(Symmetric((C+C')/2)))
    p = λ.^4; p ./= sum(p); p = filter(>(0), p)
    return -sum(p .* log.(p))
end

const BETA_C = 0.5*log(1+sqrt(2))

"exact Onsager −βf per site via double-integral (midpoint grid). ∫∫dθdφ≈(2π/n)²Σ ⇒ +Σ/(2n²)."
function onsager_logZ(β; n=2000)
    acc = 0.0; c2 = cosh(2β)^2; s2 = sinh(2β)
    for i in 0:n-1, j in 0:n-1
        θ = 2π*(i+0.5)/n; φ = 2π*(j+0.5)/n
        acc += log(c2 - s2*(cos(θ)+cos(φ)))
    end
    return log(2) + acc/(2*n^2)     # ln2 + (1/8π²)·(2π/n)²·Σ = ln2 + Σ/(2n²)
end

"exact spontaneous magnetization (β>β_c), else 0."
onsager_m(β) = β > BETA_C ? (1 - sinh(2β)^(-4))^(1/8) : 0.0
