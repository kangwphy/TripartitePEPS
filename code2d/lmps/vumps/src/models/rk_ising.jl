# ising.jl -- the benchmark state: the RK state of the 2D classical Ising
# model, as a genuine quantum state with an explicit bra.
# =============================================================================
#   |psi> = sum_s exp(-beta E(s)/2) |s>,     E(s) = -sum_<ij> s_i s_j
# so that  <psi|psi> = sum_s exp(-beta E(s)) = Z_classical(beta).
#
# As a PEPS this is the copy (delta) tensor with M = W^{1/4} on every virtual
# leg, where W[s,s'] = exp(beta s s').  Each bond then receives M from both ends,
# giving W^{1/2} per bond in the ket, hence amplitude exp(-beta E/2); the ket
# times the bra restores the full W per bond, i.e. the classical Boltzmann
# weight.  This is the benchmark precisely because the answer is known exactly:
# the free energy per site must reproduce Onsager.
# =============================================================================

using LinearAlgebra

const SPINS = (1.0, -1.0)

"""
    ising_bond(beta) -> (W, M)

`W[s,s'] = exp(beta s s')` is the classical Boltzmann bond weight.  `M` is the
matrix that a PEPS site puts on each of its virtual legs so that the KET network
carries the RK amplitude.

Two different square roots meet here and must not be confused, which is the one
bug this file has already had:

  * the RK amplitude needs HALF the Boltzmann weight per bond, and "half" there
    is ELEMENTWISE: Whalf[s,s'] = exp(beta s s' / 2).  This defines the state.
  * splitting that bond weight between its two end sites is a MATRIX square
    root: Whalf = M M, so each site contributes one M.

Taking the matrix fourth root of W instead gives M M = W^{1/2} in the matrix
sense, which is NOT exp(beta s s'/2), and the resulting network computes a
different function of the spins altogether.
"""
function ising_bond(beta::Real)
    isfinite(beta) && beta >= 0 ||
        throw(ArgumentError("the ferromagnetic RK-Ising benchmark needs finite beta >= 0"))
    W     = [exp(beta * s * t)     for s in SPINS, t in SPINS]
    Whalf = [exp(beta * s * t / 2) for s in SPINS, t in SPINS]   # ELEMENTWISE half
    F = eigen(Symmetric(Whalf))
    tol = 100eps(Float64) * max(maximum(abs, F.values), 1.0)
    minimum(F.values) >= -tol || error("ising_bond: Whalf is not positive semidefinite")
    M = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0))) * F.vectors'
    W, M
end

"""
    rk_ising(beta) -> PEPS

The RK PEPS: A[s;u,l,d,r] = M[s,u] M[s,l] M[s,d] M[s,r], with M M = Whalf so
that each bond of the ket network carries exp(beta s s' / 2).
Physical and virtual dimensions are both 2.
"""
function rk_ising(beta::Real)
    _, M = ising_bond(beta)
    A = zeros(ComplexF64, 2, 2, 2, 2, 2)
    for s in 1:2, u in 1:2, l in 1:2, dn in 1:2, r in 1:2
        A[s, u, l, dn, r] = M[s, u] * M[s, l] * M[s, dn] * M[s, r]
    end
    PEPS(A)
end

"Exact Onsager free energy per site of the 2D square-lattice Ising model,
 f = -ln(Z)/(N beta), by numerical quadrature of the standard integral."
function onsager_f(beta::Real; n::Int = 4000)
    isfinite(beta) && beta > 0 || throw(ArgumentError("Onsager beta must be finite and positive"))
    n > 0 || throw(ArgumentError("quadrature size n must be positive"))
    k = 1 / (sinh(2beta)^2)
    integ = 0.0
    for i in 1:n                      # midpoint rule over theta in [0, pi]
        th = (i - 0.5) * pi / n
        integ += log(cosh(2beta)^2 + sqrt(1 + k^2 - 2k * cos(2th)) / k) * (pi / n)
    end
    lnZ = log(2) / 2 + integ / (2pi)
    -lnZ / beta
end
