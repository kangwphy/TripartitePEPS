# cube.jl — tripartite multi-entropy S_2^{(3)} ("replica cube") from an MPS or dense state.
#
# Definition (note/multientropy_QBT.tex):
#   Z = Tr[ ρ⊗4 Π_A Π_B Π_C ],  replicas r ∈ {1,2,3,4} ≙ {00,01,10,11} (Z2×Z2),
#   σ_A: 1↔3, 2↔4 (flip bit 1);  σ_B: 1↔2, 3↔4 (flip bit 2);  σ_C: 1↔4, 2↔3 (flip both).
#   S_2^{(3)} = -(1/2) log Z,   S_2(X) = -log Tr ρ_X²,
#   tildeS   = -log Z - S_2(A) - S_2(B) - S_2(C).
#
# Index form (bra r carries ket indices of the σ-partners):
#   Z = Σ ψ_{a1b1c1}ψ_{a2b2c2}ψ_{a3b3c3}ψ_{a4b4c4} ·
#       ψ*_{a3b2c4} ψ*_{a4b1c3} ψ*_{a1b4c2} ψ*_{a2b3c1}
#
# Everything reduces to the Gram tensor of the middle region:
#   g4[α',α,γ',γ] = Σ_B  X*_{α'γ'}(B) X_{αγ}(B)
# with X_{αγ}(B) the middle-arc amplitudes between orthonormal bases of A (index α)
# and C (index γ). Then with Gm = reshape(g4, a², c²) (rows (α' fast, α)),
# Gsw = reshape(permutedims(g4,(2,1,3,4)), a², c²):
#   K = Gmᵀ * Gsw,  k = reshape(K, c,c,c,c),  Z = Σ k .* permutedims(k,(4,3,2,1))
# (derivation in ../report; validated against the literal Tr[ρ⊗4 Π] in the self-test).
#
# All arrays real Float64 (GS of TFIM/XX real). Cost O(a²c⁴ + ℓ_B χ⁶), memory O(χ⁴).

using LinearAlgebra

"""results container"""
struct CubeResult
    Z::Float64          # normalized Z
    S3::Float64         # -(1/2) log Z
    S2A::Float64
    S2B::Float64
    S2C::Float64
    tildeS::Float64
    nrm2::Float64       # ⟨ψ|ψ⟩ as computed (should be ≈1)
end

"""core evaluation from the Gram 4-tensor g4[α',α,γ',γ] (dims a,a,c,c), unnormalized."""
function cube_eval(g4::Array{Float64,4})
    a = size(g4,1); c = size(g4,3)
    @assert size(g4,2) == a && size(g4,4) == c
    nrm2 = 0.0
    @inbounds for γ in 1:c, α in 1:a
        nrm2 += g4[α,α,γ,γ]
    end
    Gm  = reshape(g4, a*a, c*c)
    Gsw = reshape(permutedims(g4, (2,1,3,4)), a*a, c*c)
    K = transpose(Gm) * Gsw                       # (c², c²): K[(γ'1 γ1),(γ'2 γ2)]
    k = reshape(K, c, c, c, c)
    kp = permutedims(k, (4,3,2,1))
    Z = dot(vec(k), vec(kp))
    # Renyi-2 pieces
    rhoA = zeros(a, a)                             # rhoA[m1=α', m2=α] = Σ_γ g4[α',α,γ,γ]
    @inbounds for γ in 1:c, m2 in 1:a, m1 in 1:a
        rhoA[m1,m2] += g4[m1,m2,γ,γ]
    end
    T2A = sum(rhoA .* transpose(rhoA))
    rhoC = zeros(c, c)
    @inbounds for m2 in 1:c, m1 in 1:c
        s = 0.0
        for α in 1:a
            s += g4[α,α,m1,m2]
        end
        rhoC[m1,m2] = s
    end
    T2C = sum(rhoC .* transpose(rhoC))
    T2B = dot(vec(g4), vec(permutedims(g4, (2,1,4,3))))
    # normalize
    Zn   = Z   / nrm2^4
    T2An = T2A / nrm2^2
    T2Bn = T2B / nrm2^2
    T2Cn = T2C / nrm2^2
    S2A = -log(T2An); S2B = -log(T2Bn); S2C = -log(T2Cn)
    S3  = -0.5*log(Zn)
    tS  = -log(Zn) - S2A - S2B - S2C
    return CubeResult(Zn, S3, S2A, S2B, S2C, tS, nrm2)
end

# ---------------------------------------------------------------- MPS pathway

"""
    gauge_for_cut!(A, x, y)

A = [1..x], B = [x+1..y], C = [y+1..N]. Requires 1 ≤ x < y < N.
After this call sites 1..x are left-canonical, sites y+1..N right-canonical,
the orthogonality center sits inside B. (Full left-canonicalization first, then
a partial right sweep — sites ≤ x are untouched by the second pass.)
"""
function gauge_for_cut!(A::Vector{Array{Float64,3}}, x::Int, y::Int)
    N = length(A)
    @assert 1 <= x < y < N
    left_canonicalize!(A)
    right_canonicalize_range!(A, y+1)
    return A
end

"""
    gram_tensor(A, x, y) -> g4[α',α,γ',γ]

Transfer-matrix product over the middle arc. Bond at cut x = right bond of site x
(dim a); bond at cut y = right bond of site y (dim c).
"""
function gram_tensor(A::Vector{Array{Float64,3}}, x::Int, y::Int)
    a = size(A[x], 3)
    G = Matrix{Float64}(I, a*a, a*a)               # accumulates (l'l) × (r'r)
    for i in x+1:y
        l, d, r = size(A[i])
        A1 = reshape(permutedims(A[i], (1,3,2)), l*r, d)
        T = A1 * transpose(A1)                     # [(l',r'),(l,r)]
        Em = reshape(permutedims(reshape(T, l, r, l, r), (1,3,2,4)), l*l, r*r)
        G = G * Em
    end
    c = size(A[y], 3)
    g4 = permutedims(reshape(G, a, a, c, c), (1,2,3,4))  # already (α',α,γ',γ)
    return g4
end

"""full MPS cube evaluation for cut (x, y). Mutates gauge of A."""
function cube_mps!(A::Vector{Array{Float64,3}}, x::Int, y::Int)
    gauge_for_cut!(A, x, y)
    g4 = gram_tensor(A, x, y)
    return cube_eval(g4)
end

# ---------------------------------------------------------------- dense pathway

"""g4 from a dense state vector, little-endian blocks (dA fastest)."""
function gram_dense(psi::Vector{Float64}, dA::Int, dB::Int, dC::Int)
    @assert length(psi) == dA*dB*dC
    X = reshape(psi, dA, dB, dC)
    Xp = reshape(permutedims(X, (1,3,2)), dA*dC, dB)
    M = Xp * transpose(Xp)                         # [(a'c'),(a c)]
    g4 = permutedims(reshape(M, dA, dC, dA, dC), (1,3,2,4))  # (a',a,c',c)
    return g4
end

"""cube from dense state via the same g4 route."""
cube_dense(psi::Vector{Float64}, dA::Int, dB::Int, dC::Int) =
    cube_eval(gram_dense(psi, dA, dB, dC))

"""
    cube_literal(psi, dA, dB, dC) -> Z (unnormalized)

Literal Tr[ρ⊗4 Π_A Π_B Π_C]: quadruple sum over composite indices. O(D⁴) — only for
tiny D. Independent of the g4 machinery; used to pin the definition in the self-test.
"""
function cube_literal(psi::Vector{Float64}, dA::Int, dB::Int, dC::Int)
    D = dA*dB*dC
    @assert length(psi) == D
    rho = psi * transpose(psi)
    # composite index (little-endian): n = a + dA*(b-1) + dA*dB*(c-1), 1-based
    idx(a,b,c) = a + dA*(b-1) + dA*dB*(c-1)
    dec = Vector{NTuple{3,Int}}(undef, D)
    for c in 1:dC, b in 1:dB, a in 1:dA
        dec[idx(a,b,c)] = (a,b,c)
    end
    Z = 0.0
    for n1 in 1:D, n2 in 1:D, n3 in 1:D, n4 in 1:D
        a1,b1,c1 = dec[n1]; a2,b2,c2 = dec[n2]; a3,b3,c3 = dec[n3]; a4,b4,c4 = dec[n4]
        # bra r = (a_{σA(r)}, b_{σB(r)}, c_{σC(r)}); σA:1↔3,2↔4  σB:1↔2,3↔4  σC:1↔4,2↔3
        m1 = idx(a3,b2,c4); m2 = idx(a4,b1,c3); m3 = idx(a1,b4,c2); m4 = idx(a2,b3,c1)
        Z += rho[n1,m1]*rho[n2,m2]*rho[n3,m3]*rho[n4,m4]
    end
    return Z
end
