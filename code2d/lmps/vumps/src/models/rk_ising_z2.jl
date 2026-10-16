"""Opt-in bosonic Z2-symmetric RK-Ising VUMPS adapter.

Load this file after `LMPSVUMPS.jl`.  It is deliberately not included by the
legacy module: loading or running the existing dense campaigns changes nothing.
The optimization is graded; its result is exported losslessly to the original
spin basis for the existing bare/direct contractions and observables.
"""
module RKIsingZ2

import ..LMPSVUMPS
using LinearAlgebra
using TensorKit
using MPSKit
using PEPSKit

export rk_ising_z2_peps, dense_original_spin_mps, solve_rk_ising_z2

const _HADAMARD = Float64[1 1; 1 -1] / sqrt(2.0)

"Apply a matrix on one explicit dense leg, preserving all other leg orderings."
function _rotate_leg(A::AbstractArray, H::AbstractMatrix, axis::Integer)
    1 <= axis <= ndims(A) || throw(ArgumentError("invalid tensor axis $axis"))
    size(A, axis) == size(H, 2) || throw(DimensionMismatch("basis rotation"))
    order = (axis, (i for i in 1:ndims(A) if i != axis)...)
    moved = permutedims(A, order)
    transformed = H * reshape(moved, size(H, 2), :)
    shape = (size(H, 1), size(moved)[2:end]...)
    permutedims(reshape(transformed, shape), invperm(collect(order)))
end

function _rotate_all_ising_legs(A)
    result = A
    for axis in 1:ndims(A)
        size(A, axis) == 2 || throw(DimensionMismatch("Ising legs must have dimension 2"))
        result = _rotate_leg(result, _HADAMARD, axis)
    end
    result
end

_relative_difference(A, B) = Float64(norm(A - B) / max(norm(A), norm(B), eps(Float64)))

"""
    rk_ising_z2_peps(beta)

Construct the same RK state as `LMPSVUMPS.rk_ising`, in the spin-flip parity
basis on physical AND all four virtual legs.  The physical charge equals the
sum of the virtual charges modulo two.  This is bosonic `Z2Irrep`, not fermion
parity.  The original spin-basis source is retained for dense/direct export.
"""
function rk_ising_z2_peps(beta::Real)
    source = LMPSVUMPS.rk_ising(beta)
    original = permutedims(source.A, (1, 2, 5, 4, 3)) # physical,N,E,S,W
    parity_tensor = _rotate_all_ising_legs(original)
    physical = Z2Space(0 => 1, 1 => 1)
    ne = Z2Space(0 => 1, 1 => 1)
    sw = dual(ne)
    tensor = TensorKit.TensorMap(
        parity_tensor, physical ← ne ⊗ ne ⊗ sw ⊗ sw; tol=1e-12,
    )
    reconstructed_parity = convert(Array, tensor)
    reconstructed_original = _rotate_all_ising_legs(reconstructed_parity)
    forbidden_weight = 0.0
    for I in CartesianIndices(parity_tensor)
        if isodd(sum(Tuple(I)) - 5)
            forbidden_weight += abs2(parity_tensor[I])
        end
    end
    audits = (
        peps_roundtrip_residual=_relative_difference(reconstructed_original, original),
        peps_graded_projection_residual=_relative_difference(reconstructed_parity, parity_tensor),
        peps_parity_residual=Float64(sqrt(forbidden_weight) / max(norm(parity_tensor), eps(Float64))),
    )
    maximum(values(audits)) <= 1e-12 || error("RK-Ising parity conversion failed: $audits")
    (; source, peps=PEPSKit.InfinitePEPS(tensor), parity_tensor,
       hadamard=copy(_HADAMARD), audits)
end

_dense_space(V::TensorKit.ElementarySpace) = ComplexSpace(dim(V); dual=isdual(V))
_dense_product(V) = reduce(⊗, (_dense_space(factor) for factor in V))

"Expand a graded tensor without projecting, truncating, or changing its bond basis."
function _expanded_tensor(tensor, data=convert(Array, tensor))
    TensorKit.TensorMap(data, _dense_product(codomain(tensor)) ← _dense_product(domain(tensor)))
end

"Convert the two explicit ket/bra PEPS virtual legs from parity to spin basis."
function _original_spin_rank4(tensor)
    raw = convert(Array, tensor)
    ndims(raw) == 4 || throw(DimensionMismatch("expected chi,Dket,Dbra,chi"))
    size(raw, 2) == size(raw, 3) == 2 || throw(DimensionMismatch("RK ket/bra legs"))
    data = _rotate_leg(_rotate_leg(raw, _HADAMARD, 2), _HADAMARD, 3)
    _expanded_tensor(tensor, data)
end

"""
    dense_original_spin_mps(symmetric_state) -> InfiniteMPS

Losslessly expand AL/AR/AC/C and undo the Hadamard on the two local PEPS
ket/bra legs.  The retained bond basis remains the concatenated parity-sector
basis.  The four-array constructor does not reoptimize, regauge, or discard C.
"""
function dense_original_spin_mps(state::MPSKit.InfiniteMPS)
    AL = [_original_spin_rank4(t) for t in state.AL]
    AR = [_original_spin_rank4(t) for t in state.AR]
    AC = [_original_spin_rank4(t) for t in state.AC]
    C = [_expanded_tensor(t) for t in state.C]
    MPSKit.InfiniteMPS(AL, AR, C, AC)
end

function _dense_original_environments(envs)
    # GL and GR have the same explicit (chi,Dket,Dbra,chi) leg order.
    # Transforming, rather than resolving their fixed points, also preserves
    # the selected invariant environment when the leading space is degenerate.
    GLs = MPSKit.PeriodicVector([_original_spin_rank4(t) for t in envs.GLs])
    GRs = MPSKit.PeriodicVector([_original_spin_rank4(t) for t in envs.GRs])
    MPSKit.InfiniteEnvironments(GLs, GRs)
end

function _sector_signs(V)
    signs = ones(Float64, dim(V))
    for i in axes(V, Z2Irrep(1))
        signs[i] = -1.0
    end
    signs
end

function _symmetry_residual(state)
    # In this explicitly unfused ordering, invariance is
    # Q_left * Z_ket * Z_bra * A * Q_right = A.
    worst = 0.0
    for tensors in (state.AL, state.AR, state.AC), tensor in tensors
        raw = convert(Array, tensor)
        qs = [_sector_signs(space(tensor, axis)) for axis in 1:4]
        transformed = raw .* reshape(qs[1], :, 1, 1, 1) .*
                      reshape(qs[2], 1, :, 1, 1) .* reshape(qs[3], 1, 1, :, 1) .*
                      reshape(qs[4], 1, 1, 1, :)
        worst = max(worst, _relative_difference(raw, transformed))
    end
    Float64(worst)
end

function _canonical_residual(state)
    worst = 0.0
    for i in 1:length(state)
        AL = convert(Array, state.AL[i])
        AR = convert(Array, state.AR[i])
        L = reshape(AL, :, size(AL, 4))
        R = reshape(AR, size(AR, 1), :)
        Il = Matrix{ComplexF64}(I, size(L, 2), size(L, 2))
        Ir = Matrix{ComplexF64}(I, size(R, 1), size(R, 1))
        worst = max(worst, _relative_difference(L' * L, Il),
                    _relative_difference(R * R', Ir))
    end
    max(Float64(worst), LMPSVUMPS._center_residual(state))
end

function _export_roundtrip_residual(graded, dense)
    worst = 0.0
    for name in (:AL, :AR, :AC)
        for (before, after) in zip(getproperty(graded, name), getproperty(dense, name))
            original = convert(Array, before)
            returned = _rotate_leg(_rotate_leg(convert(Array, after), _HADAMARD, 2), _HADAMARD, 3)
            worst = max(worst, _relative_difference(original, returned))
        end
    end
    for (before, after) in zip(graded.C, dense.C)
        worst = max(worst, _relative_difference(convert(Array, before), convert(Array, after)))
    end
    Float64(worst)
end

function _xi_ratio(numerator, denominator)
    ratio = abs(numerator / denominator)
    isfinite(ratio) && ratio > 0 || return NaN
    abs(ratio - 1) <= 1e-12 && return Inf
    ratio < 1 ? -1 / log(ratio) : NaN
end

function _spectra(graded, dense; num_vals, tol)
    even = ComplexF64.(MPSKit.transfer_spectrum(
        graded; sector=Z2Irrep(0), num_vals, tol,
    ))
    odd = ComplexF64.(MPSKit.transfer_spectrum(
        graded; sector=Z2Irrep(1), num_vals, tol,
    ))
    sort!(even; by=abs, rev=true)
    sort!(odd; by=abs, rev=true)
    combined = sort(vcat(even, odd); by=abs, rev=true)
    dense_spectrum = ComplexF64.(MPSKit.transfer_spectrum(dense; num_vals, tol))
    sort!(dense_spectrum; by=abs, rev=true)
    length(even) >= 2 || error("fewer than two even transfer eigenvalues")
    isempty(odd) && error("empty odd transfer sector")
    discrepancy = maximum(minimum(abs.(combined .- value)) for value in dense_spectrum)
    (
        even, odd, combined, dense=dense_spectrum,
        xi_even=_xi_ratio(even[2], even[1]),
        xi_odd=_xi_ratio(odd[1], even[1]),
        xi_all=_xi_ratio(combined[2], combined[1]),
        dense_spectrum_abs_error=Float64(discrepancy),
    )
end

"""
    solve_rk_ising_z2(beta, chi; chi_even=cld(chi,2), ...)

Run VUMPS with exact bosonic Z2 selection rules.  `chi` is the TOTAL retained
dimension; `chi_even` and `chi-chi_even` are fixed throughout this run.  Seed
the caller's random generator to make initialization reproducible.

Returns a legacy-compatible dense `boundary` in the ORIGINAL spin basis,
the graded state/transfer/environments, both charge-resolved spectra, and
conversion/convergence audits.  No dense VUMPS optimization is performed.
The sector spectra are always saved because requesting only the default even
sector would silently change the correlation-length definition.
"""
function solve_rk_ising_z2(
        beta::Real, chi::Integer;
        chi_even::Integer=cld(chi, 2),
        tolerance::Real=1e-9,
        diagnostic_tolerance::Real=max(100tolerance, 1e-8),
        max_iterations::Integer=500,
        verbosity::Integer=0,
        accept_unconverged::Bool=false,
        spectrum_num::Integer=8,
        spectrum_tol::Real=1e-11,
    )
    chi >= 2 || throw(ArgumentError("Z2 benchmark needs chi >= 2"))
    1 <= chi_even < chi || throw(ArgumentError("both charge sectors must have nonzero dimension"))
    tolerance > 0 && diagnostic_tolerance > 0 && spectrum_tol > 0 ||
        throw(ArgumentError("all tolerances must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    spectrum_num >= 2 || throw(ArgumentError("spectrum_num must be at least two"))
    model = rk_ising_z2_peps(beta)
    transfer = PEPSKit.InfiniteTransferPEPS(model.peps, 1, 1)
    bond = Z2Space(0 => Int(chi_even), 1 => Int(chi - chi_even))
    initial = PEPSKit.initialize_mps(transfer, [bond])
    iterations = Ref(0)
    finalize = function (iteration, state, operator, environments)
        iterations[] = max(iterations[], Int(iteration) + 1)
        state, environments
    end
    algorithm = MPSKit.VUMPS(;
        tol=Float64(tolerance), maxiter=Int(max_iterations), verbosity=Int(verbosity),
        alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false), finalize,
    )
    state, environments, galerkin = MPSKit.leading_boundary(initial, transfer, algorithm)

    dense_state = dense_original_spin_mps(state)
    dense_transfer = PEPSKit.InfiniteTransferPEPS(LMPSVUMPS.pepskit_from_lmps(model.source), 1, 1)
    dense_environments = _dense_original_environments(environments)
    MPSKit.issamespace(dense_environments, dense_state, dense_transfer, dense_state) ||
        error("dense exported environment spaces do not match original-spin transfer")
    eigengraded = ComplexF64(LMPSVUMPS._scalarize(LMPSVUMPS._expectation_value(state, transfer, environments)))
    eigenvalue = ComplexF64(LMPSVUMPS._scalarize(LMPSVUMPS._expectation_value(dense_state, dense_transfer, dense_environments)))
    left_residual, right_residual = LMPSVUMPS._environment_residuals(dense_state, dense_transfer, dense_environments)
    center_residual = LMPSVUMPS._center_residual(dense_state)
    symmetry_residual = _symmetry_residual(state)
    canonical_residual = _canonical_residual(dense_state)
    export_roundtrip_residual = _export_roundtrip_residual(state, dense_state)
    eigenvalue_basis_residual = Float64(abs(eigenvalue - eigengraded) / max(abs(eigenvalue), abs(eigengraded), eps(Float64)))
    magnitude = abs(eigenvalue)
    finite_scale = isfinite(real(eigenvalue)) && isfinite(imag(eigenvalue)) && isfinite(magnitude) && magnitude > 0
    algebra_ok = max(symmetry_residual, export_roundtrip_residual, eigenvalue_basis_residual) <= 1e-10
    algebra_ok || error("lossless Z2-to-dense export failed: symmetry=$symmetry_residual, roundtrip=$export_roundtrip_residual, eigenvalue=$eigenvalue_basis_residual")
    converged = finite_scale && isfinite(galerkin) && galerkin <= tolerance &&
                left_residual <= diagnostic_tolerance && right_residual <= diagnostic_tolerance &&
                center_residual <= diagnostic_tolerance && canonical_residual <= diagnostic_tolerance
    diagnostics = LMPSVUMPS.VUMPSBoundaryDiagnostics(
        converged, Float64(galerkin), Float64(tolerance), eigenvalue,
        finite_scale ? log(magnitude) : NaN, Float64(left_residual), Float64(right_residual),
        Float64(center_residual), iterations[],
    )
    if !converged && !accept_unconverged
        error("Z2 VUMPS diagnostics failed: galerkin=$galerkin, GL=$left_residual, GR=$right_residual, center=$center_residual, canonical=$canonical_residual")
    end
    boundary = LMPSVUMPS.VUMPSBoundary(
        dense_state, dense_transfer, dense_environments, model.source, :cpu, diagnostics,
    )
    spectra = _spectra(state, dense_state; num_vals=Int(spectrum_num), tol=Float64(spectrum_tol))
    audits = merge(model.audits, (
        symmetry="bosonic_Z2", chi_total=Int(chi), chi_even=Int(chi_even), chi_odd=Int(chi-chi_even),
        dense_basis="original_spin", dense_reoptimized=false,
        symmetry_residual, canonical_residual, export_roundtrip_residual, eigenvalue_basis_residual,
        graded_galerkin_residual=Float64(galerkin),
        dense_left_environment_residual=Float64(left_residual),
        dense_right_environment_residual=Float64(right_residual),
        dense_center_residual=Float64(center_residual),
        dense_spectrum_abs_error=spectra.dense_spectrum_abs_error,
        converged, iterations=iterations[],
    ))
    virtual_parity = Matrix(Diagonal(_sector_signs(space(state.AL[1], 1))))
    (; boundary, symmetric_state=state, symmetric_transfer=transfer,
       symmetric_environments=environments, virtual_parity, audits, sector_spectra=spectra)
end

end # module RKIsingZ2
