# validate_split_bridge_ranks_ising.jl
# -----------------------------------------------------------------------------
# Measure the region--replica bridge ranks that arise when the current 2x2
# split-replica CTMRG window is regrouped into six dense octahedron vertices.
# Only the certified one-copy Ising C,T,a tensors enter these bridges; the pair
# defect tensors live at the six vertices and do not affect a bridge rank.
# -----------------------------------------------------------------------------
include(joinpath(@__DIR__, "rk_ctmrg.jl"))
using LinearAlgebra, Printf

function numerical_rank(s; rtol=1e-12)
    isempty(s) && return 0
    count(x -> x > rtol * first(s), s)
end

function discarded_fraction(s, keep)
    keep >= length(s) && return 0.0
    sqrt(sum(abs2, @view(s[keep+1:end])) / sum(abs2, s))
end

"A-region outer bridge; a direct dimension-D identity multiplies its spectrum."
function bridge_A_spectrum(C, T)
    chi, D = size(T, 1), size(T, 3)
    matrix = zeros(chi * D, chi)
    for n0 in 1:chi, ltl in 1:D, w1 in 1:chi
        row = n0 + chi * (ltl - 1)
        matrix[row, w1] = sum(C[w2, n0] * T[w1, w2, ltl] for w2 in 1:chi)
    end
    sort(repeat(svdvals(matrix), inner=D), rev=true)
end

"B-region bridge through the plain top-right core site."
function bridge_B_spectrum(C, T, a)
    chi, D = size(T, 1), size(T, 3)
    matrix = zeros(chi * D, chi * D)
    for n1 in 1:chi, htop in 1:D, e1 in 1:chi, vright in 1:D
        row = n1 + chi * (htop - 1)
        col = e1 + chi * (vright - 1)
        value = 0.0
        for n2 in 1:chi, e0 in 1:chi, utr in 1:D, rtr in 1:D
            value += C[n2, e0] * T[n1, n2, utr] * T[e0, e1, rtr] *
                     a[utr, htop, vright, rtr]
        end
        matrix[row, col] = value
    end
    svdvals(matrix)
end

"C-region outer bridge; the direct hbot identity multiplies its spectrum."
function bridge_C_spectrum(C, T)
    chi, D = size(T, 1), size(T, 3)
    matrix = zeros(chi * D, chi * D)
    for w0 in 1:chi, dbl in 1:D, e2 in 1:chi, dbr in 1:D
        row = w0 + chi * (dbl - 1)
        col = e2 + chi * (dbr - 1)
        value = 0.0
        for s0 in 1:chi, s1 in 1:chi, s2 in 1:chi
            value += C[e2, s2] * C[s0, w0] * T[s0, s1, dbl] * T[s1, s2, dbr]
        end
        matrix[row, col] = value
    end
    # The outer bridge has rank chi; the direct hbot identity supplies the
    # second factor D, giving the dense octahedron edge rank D*chi.
    sort(repeat(svdvals(matrix), inner=D), rev=true)
end

betas = (0.30, 0.5 * log(1 + sqrt(2.0)), 0.48)
chis = (2, 3, 4, 6)
println("beta, requested_chi, actual_chi, rank_A, rank_B, rank_C, " *
        "discard_to_chi_A, discard_to_chi_B, discard_to_chi_C")
for beta in betas
    a, _ = ising_tensors(beta)
    for requested in chis
        C, T, _ = ctmrg(a; χmax=requested, nmax=20000, tol=1e-13)
        chi = size(C, 1)
        sA = bridge_A_spectrum(C, T)
        sB = bridge_B_spectrum(C, T, a)
        sC = bridge_C_spectrum(C, T)
        @printf("%.12f %d %d %d %d %d %.6e %.6e %.6e\n",
                beta, requested, chi,
                numerical_rank(sA), numerical_rank(sB), numerical_rank(sC),
                discarded_fraction(sA, chi), discarded_fraction(sB, chi),
                discarded_fraction(sC, chi))
    end
end
