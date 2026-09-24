function vumps_free_energy(boundary::VUMPSBoundary, beta::Real)
    beta > 0 || throw(ArgumentError("beta must be positive"))
    -vumps_logz(boundary) / beta
end

"Exact spontaneous magnetization of the infinite square-lattice Ising model."
function ising_exact_magnetization(beta::Real)
    betac = 0.5 * log(1 + sqrt(2))
    beta <= betac && return 0.0
    (1 - sinh(2beta)^(-4))^(1 / 8)
end

"Exact connected-spin exponential correlation length along a lattice axis."
function ising_exact_correlation_length(beta::Real)
    betac = 0.5 * log(1 + sqrt(2))
    beta == betac && return Inf
    beta_dual = -0.5 * log(tanh(beta))
    beta < betac ? 1 / (2 * (beta_dual - beta)) :
                    1 / (4 * (beta - beta_dual))
end

"""
    vumps_ising_magnetization(boundary)

Measure the diagonal RK-Ising spin operator `diag(+1,-1)` using the same
converged VUMPS state and the same left/right MPO environments as the norm.
The local impurity is inserted on the PEPS ket physical leg; the denominator
is the unmodified norm-transfer expectation value.  No new environment is
solved and no exact Ising formula enters this measurement.
"""
function vumps_ising_magnetization(boundary::VUMPSBoundary)
    boundary.source.d == 2 || throw(DimensionMismatch(
        "the Ising magnetization helper requires physical dimension 2"
    ))
    boundary.backend === :cpu || error(
        "vumps_ising_magnetization currently requires a CPU boundary"
    )
    sandwich = boundary.transfer[1]
    top, bottom = sandwich
    physical = codomain(top)
    spin = TensorKit.TensorMap(
        ComplexF64[1 0; 0 -1], physical ← physical,
    )
    impurity = (spin * top, bottom)
    center = boundary.state.AC[1]
    left = boundary.environments.GLs[1]
    right = boundary.environments.GRs[1]
    numerator = MPSKit.contract_mpo_expval(center, left, impurity, right)
    denominator = MPSKit.contract_mpo_expval(
        center, left, boundary.transfer[1], right,
    )
    value = numerator / denominator
    abs(imag(value)) <= 1e-9 * max(abs(real(value)), 1.0) ||
        @warn "magnetization has a non-negligible imaginary part" value
    Float64(real(value))
end

"""
    vumps_boundary_observables(boundary, beta; spectrum_num=8, spectrum_tol=1e-11)

All inexpensive one-copy diagnostics derived from one saved boundary state.
The returned Schmidt probabilities are normalized eigenvalues of the boundary
cut density matrix.  We retain the full `chi`-component Schmidt vector and the
requested leading transfer spectrum, then derive
`SvN, S2, S3, S4, Sinf`, correlation length, magnetization, and free energy.
"""
function vumps_boundary_observables(
        boundary::VUMPSBoundary, beta::Real;
        spectrum_num::Integer=8,
        spectrum_tol::Real=1e-11,
    )
    beta > 0 || throw(ArgumentError("beta must be positive"))
    spectrum_num >= 2 || throw(ArgumentError("spectrum_num must be at least 2"))
    chi = size(_host_array(boundary.state.C[1]), 1)
    spectrum = MPSKit.transfer_spectrum(
        boundary.state; num_vals=min(Int(spectrum_num), chi^2), tol=spectrum_tol,
    )
    order = sortperm(abs.(spectrum); rev=true)
    spectrum = ComplexF64.(spectrum[order])
    length(spectrum) >= 2 || error("fewer than two transfer eigenvalues")
    magnitudes = abs.(spectrum)
    ratio = magnitudes[2] / magnitudes[1]
    0 < ratio < 1 || error("invalid transfer ratio $ratio")
    xi = -1 / log(ratio)

    C = _host_array(boundary.state.C[1])
    singular_values = Float64.(abs.(svdvals(C)))
    probabilities = abs2.(singular_values)
    probabilities ./= sum(probabilities)
    entropy_vn = -sum(p > eps(Float64) ? p * log(p) : 0.0 for p in probabilities)
    renyi(alpha) = log(sum(probabilities .^ alpha)) / (1 - alpha)
    entropy_renyi2 = renyi(2)
    entropy_renyi3 = renyi(3)
    entropy_renyi4 = renyi(4)
    entropy_infinity = -log(maximum(probabilities))

    magnetization = vumps_ising_magnetization(boundary)
    exact_magnetization = ising_exact_magnetization(beta)
    exact_spin_xi = ising_exact_correlation_length(beta)
    free_energy = vumps_free_energy(boundary, beta)
    exact_free_energy = onsager_f(beta)
    free_energy_abs_error = abs(free_energy - exact_free_energy)
    free_energy_rel_error = free_energy_abs_error / abs(exact_free_energy)

    (; xi, exact_spin_xi, transfer_spectrum=spectrum,
       lambda1_abs=magnitudes[1], lambda2_abs=magnitudes[2],
       lambda2_over_lambda1=ratio,
       schmidt_singular_values=singular_values,
       schmidt_probabilities=probabilities,
       entropy_vn, entropy_renyi2, entropy_renyi3, entropy_renyi4,
       entropy_infinity, schmidt_gap=probabilities[1] - probabilities[2],
       effective_rank=exp(entropy_vn),
       magnetization, magnetization_abs=abs(magnetization),
       exact_magnetization,
       magnetization_abs_error=abs(abs(magnetization) - exact_magnetization),
       row_logz=vumps_logz(boundary), free_energy, exact_free_energy,
       free_energy_abs_error, free_energy_rel_error)
end
