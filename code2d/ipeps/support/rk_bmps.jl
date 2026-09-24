# rk_bmps.jl — boundary-MPS contraction of the k-replica twisted 2D Ising network
# for the RK-state tripartite multi-entropy. Dense-array reference implementation
# (validated against exact enumeration; the ITensor/GPU version cross-checks this).
#
# For the RK state |ψ⟩ ∝ Σ_s √w(s)|s⟩ of the 2D Ising model, all replica quantities
# are classical partition-function ratios (note §RK):
#   S_2(X)   = -(log Z2[X] - 2 log Z)
#   log Zcube(normalized) = log Z4[cube] - 4 log Z
#   tildeS   = -log Z4 + (log Z2A + log Z2B + log Z2C) - 2 log Z
# where Zk[...] is the k-replica Ising network with the KET spins σ_r(i) as site
# variables and the BRA spins fixed by region permutations p_A,p_B,p_C (σ_X gluing).
# Each network is contracted by a boundary MPS (bond χ), tracking log-norms.
#
# Bond weight (ket σ_r + bra σ_{p_X(r)}), i∈X_i, j∈X_j:
#   M[τ_i,τ_j] = Π_r e^{β σ_r(i)σ_r(j)/2} · Π_r e^{β σ_{p_{X_i}(r)}(i) σ_{p_{X_j}(r)}(j)/2}

using LinearAlgebra

# ---- spin decode: τ ∈ 1..2^k  →  (σ_1,…,σ_k) ∈ {±1}^k -----------------------
@inline spin(tau1, r) = ((tau1 >> (r-1)) & 1) == 1 ? 1 : -1   # tau1 is 0-based

"bond weight matrix M (q×q), q=2^k, for a bond between regions Xi,Xj."
function bond_M(k, beta, Xi::Char, Xj::Char, pA, pB, pC)
    q = 1 << k
    permof(X) = X=='A' ? pA : (X=='B' ? pB : pC)
    pi_ = permof(Xi); pj_ = permof(Xj)
    M = Matrix{Float64}(undef, q, q)
    @inbounds for tj0 in 0:q-1, ti0 in 0:q-1
        e = 0.0
        for r in 1:k
            # ket factor
            e += spin(ti0, r)*spin(tj0, r)
            # bra factor: replica r's bra uses ket p_X(r)
            e += spin(ti0, pi_[r])*spin(tj0, pj_[r])
        end
        M[ti0+1, tj0+1] = exp(beta*e/2)
    end
    return M
end

"symmetric-ish split of a bond weight M into two half-matrices (for the two sites)."
# NOTE: no truncation -- all q singular values are kept. Seam bond weights are
# rank-deficient: rank 3 per 2-cycle factor (k=2 swap: rank 3 of 4; k=4 double
# 2-cycle: rank 9 of 16). Halves stay q x q regardless. See job 23560796.
function split_M(M)
    F = svd(M)
    s = sqrt.(F.S)
    P = F.U .* s'          # [τ_low, bond]   (τ index rows)
    Q = (F.Vt' .* s')      # [τ_high, bond]  (τ index rows)  (Q[τ,b] = V[τ,b]√s_b)
    return P, Q            # M[τi,τj] = Σ_b P[τi,b] Q[τj,b]
end

# region of a site (x,y), T-junction on Lx×Ly (even). '1'-replica network: all 'A'
# (regions irrelevant when perms are identity) — caller passes a region function.
region_Tjunction(x, y, Lx, Ly) = (y >= Ly÷2) ? 'C' : (x < Lx÷2 ? 'A' : 'B')

"""
    build_site_tensors(Lx,Ly,k,beta,pA,pB,pC; regfun) -> A[x,y] rank-4 (l,u,r,d)

Each site: copy-δ over τ (dim q) contracted with the half-matrix of each present bond.
Open BC: missing bonds have leg dim 1.
NOTE (verified, job 23560796): bond index dims = q = 2^k on EVERY bond.
split_M keeps all q SVD columns (no truncation, no rank detection), so the
seam rank deficiency (3 per 2-cycle of the relative twist: 3 at k=2, 9 at k=4)
is real but NOT exploited. Truncating split_M to the numerical rank would be
exact; the gain is bounded because seam sites are O(L) of L^2 and carry at
most two seam legs.
"""
function build_site_tensors(Lx, Ly, k, beta, pA, pB, pC; regfun=region_Tjunction)
    q = 1 << k
    reg(x,y) = regfun(x,y,Lx,Ly)
    # horizontal bond (x,y)-(x+1,y): half P on left site's right-leg, Q on right site's left-leg
    Hleft  = Array{Matrix{Float64}}(undef, Lx, Ly)   # right-leg half for site (x,y)
    Hright = Array{Matrix{Float64}}(undef, Lx, Ly)   # left-leg  half for site (x,y)
    Vdown  = Array{Matrix{Float64}}(undef, Lx, Ly)   # up-leg   half for site (x,y)  (bond to y+1)
    Vup    = Array{Matrix{Float64}}(undef, Lx, Ly)   # down-leg half for site (x,y)  (bond to y-1)
    for y in 0:Ly-1, x in 0:Lx-1
        if x+1 <= Lx-1
            M = bond_M(k, beta, reg(x,y), reg(x+1,y), pA,pB,pC)
            P,Q = split_M(M)
            Hleft[x+1,y+1]  = P      # site (x,y) right leg
            Hright[x+2,y+1] = Q      # site (x+1,y) left leg
        end
        if y+1 <= Ly-1
            M = bond_M(k, beta, reg(x,y), reg(x,y+1), pA,pB,pC)
            P,Q = split_M(M)
            Vdown[x+1,y+1] = P       # site (x,y) up leg (toward y+1)
            Vup[x+1,y+2]   = Q       # site (x,y+1) down leg (toward y)
        end
    end
    A = Array{Array{Float64,4}}(undef, Lx, Ly)
    for y in 0:Ly-1, x in 0:Lx-1
        # legs (l,u,r,d); half-matrices are [τ,bond]; missing → [τ,1] ones
        L = (x>=1)     ? Hright[x+1,y+1] : ones(q,1)
        U = (y<=Ly-2)  ? Vdown[x+1,y+1]  : ones(q,1)
        R = (x<=Lx-2)  ? Hleft[x+1,y+1]  : ones(q,1)
        Dn= (y>=1)     ? Vup[x+1,y+1]    : ones(q,1)
        dl=size(L,2); du=size(U,2); dr=size(R,2); dd=size(Dn,2)
        T = zeros(dl,du,dr,dd)
        @inbounds for t in 1:q
            for d in 1:dd, r in 1:dr, u in 1:du, l in 1:dl
                T[l,u,r,d] += L[t,l]*U[t,u]*R[t,r]*Dn[t,d]
            end
        end
        A[x+1,y+1] = T
    end
    return A
end

# ---- boundary MPS: absorb rows bottom→top, truncate to χ, track log-norm --------

"compress an MPS (Vector of (χL,p,χR)) to maxdim χ via one L→R SVD sweep; returns log-norm."
function compress!(mps::Vector{Array{Float64,3}}, χ::Int; cutoff=1e-14)
    n = length(mps); lognorm = 0.0
    # left-to-right: carry R factor forward, truncate
    for i in 1:n-1
        χL,p,χR = size(mps[i])
        Mmat = reshape(mps[i], χL*p, χR)
        F = svd(Mmat)
        keep = min(χ, count(>(cutoff*maximum(F.S; init=1.0)), F.S))
        keep = max(keep,1)
        U = F.U[:,1:keep]; S = F.S[1:keep]; Vt = F.Vt[1:keep,:]
        nrm = norm(S); S ./= nrm; lognorm += log(nrm)
        mps[i] = reshape(U, χL, p, keep)
        # push S*Vt into next
        SV = Diagonal(S)*Vt                       # (keep, χR)
        nl,pl,nr = size(mps[i+1])
        mps[i+1] = reshape(SV*reshape(mps[i+1], nl, pl*nr), keep, pl, nr)
    end
    # normalize last
    nrm = norm(mps[n]); mps[n] ./= nrm; lognorm += log(nrm)
    return lognorm
end

"contract the site-tensor network by boundary MPS bottom→top; returns log Z."
function contract_bmps(A::Array{Array{Float64,4}}, Lx, Ly, χ::Int)
    # boundary MPS below row 0: each site (χL=1, physical=down-leg dim=1, χR=1)
    mps = [reshape([1.0], 1,1,1) for _ in 1:Lx]
    logZ = 0.0
    for y in 1:Ly
        # absorb row y: contract each site's down-leg (d) with mps physical leg
        newmps = Vector{Array{Float64,3}}(undef, Lx)
        for x in 1:Lx
            T = A[x,y]                     # (l,u,r,d)
            dl,du,dr,dd = size(T)
            B = mps[x]                     # (χL, p, χR), p == dd
            χL,p,χR = size(B)
            @assert p == dd "physical mismatch row $y site $x: $p vs $dd"
            # contract d with p:  C[χL,l, u, χR,r] = Σ_d B[χL,d,χR] T[l,u,r,d]
            Bp = reshape(permutedims(B,(1,3,2)), χL*χR, p)     # (χL*χR, d)
            Tp = reshape(permutedims(T,(4,1,2,3)), dd, dl*du*dr) # (d, l*u*r)
            C = reshape(Bp*Tp, χL, χR, dl, du, dr)
            # new MPS tensor: left bond (χL,l), physical u, right bond (χR,r)
            C = permutedims(C, (1,3,4,2,5))                    # (χL,l,u,χR,r)
            newmps[x] = reshape(C, χL*dl, du, χR*dr)
        end
        mps = newmps
        logZ += compress!(mps, χ)
    end
    # top row up-legs are dim-1; contract the final chain (physical dim 1) to scalar
    v = reshape(mps[1], size(mps[1],1), size(mps[1],3))  # (1,χR) since χL=1,p=1
    for x in 2:Lx
        w = reshape(mps[x], size(mps[x],1), size(mps[x],3))
        v = v*w
    end
    logZ += log(abs(v[1,1]))
    return logZ
end

# ---- top-level: tripartite multi-entropy via boundary MPS ----------------------

const PID = [1]; const PSWAP2 = [2,1]
# cube perms in the σ_A=1 gauge of the user's multientropy_PEPS.pdf:
#   σ_A = 1, σ_B = (12)(34), σ_C = (13)(24)   (relative twists q_AB=(12)(34),
#   q_AC=(13)(24), q_BC=(14)(23); Z4 is gauge invariant — certified job 23550018).
# The old symmetric gauge was SA4=[3,4,1,2], SB4=[2,1,4,3], SC4=[4,3,2,1].
const SA4=[1,2,3,4]; const SB4=[2,1,4,3]; const SC4=[3,4,1,2]  # σ_A,σ_B,σ_C

"log Z of the k-replica network with region perms (pA,pB,pC)."
function logZ_network(Lx,Ly,k,beta,pA,pB,pC,χ; regfun=region_Tjunction)
    A = build_site_tensors(Lx,Ly,k,beta,pA,pB,pC; regfun=regfun)
    return contract_bmps(A, Lx, Ly, χ)
end

"boundary-MPS tildeS for the RK-Ising T-junction at bond χ."
function tildeS_bmps(Lx, Ly, beta, χ)
    lZ  = logZ_network(Lx,Ly,1,beta,PID,PID,PID,χ)                    # single Ising
    lZ2A= logZ_network(Lx,Ly,2,beta,PSWAP2,[1,2],[1,2],χ)            # swap in A
    lZ2B= logZ_network(Lx,Ly,2,beta,[1,2],PSWAP2,[1,2],χ)
    lZ2C= logZ_network(Lx,Ly,2,beta,[1,2],[1,2],PSWAP2,χ)
    lZ4 = logZ_network(Lx,Ly,4,beta,SA4,SB4,SC4,χ)                   # cube
    S2A = -(lZ2A - 2lZ); S2B = -(lZ2B - 2lZ); S2C = -(lZ2C - 2lZ)
    logZcube = lZ4 - 4lZ
    tildeS = -logZcube - S2A - S2B - S2C            # = -lZ4 + lZ2A+lZ2B+lZ2C - 2lZ
    S3 = -0.5*logZcube
    return (Lx=Lx,Ly=Ly,beta=beta,chi=χ,tildeS=tildeS,S3=S3,S2A=S2A,S2B=S2B,S2C=S2C,
            logZ=lZ,logZ4=lZ4)
end
