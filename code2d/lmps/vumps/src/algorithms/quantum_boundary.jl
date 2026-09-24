"A VUMPS boundary solved for one geometrical side of an arbitrary PEPS." 
struct QuantumVUMPSBoundary{P,O,S,E}
    side::Symbol
    source::P
    rotated_source::P
    transfer::O
    state::S
    environments::E
    row_eigenvalue::ComplexF64
    galerkin_residual::Float64
    left_environment_residual::Float64
    right_environment_residual::Float64
    center_residual::Float64
    iterations::Int
    requested_tolerance::Float64
    converged::Bool
    backend::Symbol
end

"Four independent boundary solves; no rotation is inferred from symmetry." 
struct QuantumDirectionalBoundaries{N,S,E,W}
    north::N
    south::S
    east::E
    west::W
end

const _QUANTUM_SIDES = (:north, :south, :east, :west)

function _quantum_side_source(peps::PEPSKit.InfinitePEPS, side::Symbol)
    side === :north && return peps
    side === :south && return Base.rot180(peps)
    side === :east && return Base.rotl90(peps)
    side === :west && return Base.rotr90(peps)
    throw(ArgumentError("side must be one of $(_QUANTUM_SIDES), got $side"))
end

"Internal VUMPS solve for a PEPS already oriented with its side facing north." 
function _solve_quantum_transfer(
        source::PEPSKit.InfinitePEPS, rotated::PEPSKit.InfinitePEPS, chi::Integer;
        backend::Symbol=:cpu,
        seed::Union{Nothing,Integer}=nothing,
        tolerance::Real=1e-9,
        diagnostic_tolerance::Real=max(100tolerance, 1e-8),
        max_iterations::Integer=500,
        verbosity::Integer=0,
        accept_unconverged::Bool=false,
    )
    chi > 0 || throw(ArgumentError("chi must be positive"))
    transfer_cpu = PEPSKit.InfiniteTransferPEPS(rotated, 1, 1)
    state_cpu = if isnothing(seed)
        PEPSKit.initialize_mps(transfer_cpu, [TensorKit.ComplexSpace(chi)])
    else
        rng = Random.MersenneTwister(seed)
        draw = (T, spaces...) -> randn(rng, T, spaces...)
        PEPSKit.initialize_mps(
            draw, TensorKit.scalartype(transfer_cpu), transfer_cpu,
            [TensorKit.ComplexSpace(chi)],
        )
    end
    transfer, state0, environments0 = _prepare_backend(transfer_cpu, state_cpu, backend)
    iterations = Ref(0)
    finalize = function (iteration, state, operator, environments)
        iterations[] = max(iterations[], Int(iteration) + 1)
        state, environments
    end
    algorithm = MPSKit.VUMPS(; tol=Float64(tolerance), maxiter=Int(max_iterations),
        verbosity=Int(verbosity),
        alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false), finalize)
    state, environments, galerkin = if environments0 === nothing
        MPSKit.leading_boundary(state0, transfer, algorithm)
    else
        MPSKit.leading_boundary(state0, transfer, algorithm, environments0)
    end
    eigenvalue = ComplexF64(_scalarize(_expectation_value(state, transfer, environments)))
    left_residual, right_residual = _environment_residuals(state, transfer, environments)
    center_residual = _center_residual(state)
    converged = isfinite(real(eigenvalue)) && isfinite(imag(eigenvalue)) &&
        abs(eigenvalue) > 0 && isfinite(galerkin) &&
        galerkin <= tolerance && left_residual <= diagnostic_tolerance &&
        right_residual <= diagnostic_tolerance && center_residual <= diagnostic_tolerance
    if !converged && !accept_unconverged
        error("quantum VUMPS diagnostics failed: galerkin=$galerkin, " *
            "GL=$left_residual, GR=$right_residual, center=$center_residual")
    end
    QuantumVUMPSBoundary(
        :oriented_north, source, rotated, transfer, state, environments,
        eigenvalue, Float64(galerkin), left_residual, right_residual,
        center_residual, iterations[], Float64(tolerance), converged, backend,
    )
end

"""
    solve_quantum_vumps_boundary(peps, chi; side=:north, kwargs...)

Solve one boundary independently.  `side` determines a physical rotation of
the PEPS before the common north-facing transfer construction:
`north -> identity`, `south -> rot180`, `east -> rotl90`, and
`west -> rotr90`.  Thus an asymmetric quantum PEPS gets four genuinely
independent fixed-point problems.
"""
function solve_quantum_vumps_boundary(
        peps::PEPSKit.InfinitePEPS, chi::Integer;
        side::Symbol=:north, kwargs...
    )
    rotated = _quantum_side_source(peps, side)
    result = _solve_quantum_transfer(peps, rotated, chi; kwargs...)
    QuantumVUMPSBoundary(
        side, result.source, result.rotated_source, result.transfer, result.state,
        result.environments, result.row_eigenvalue, result.galerkin_residual,
        result.left_environment_residual, result.right_environment_residual,
        result.center_residual, result.iterations, result.requested_tolerance,
        result.converged, result.backend,
    )
end

"Solve north, south, east, and west without reusing a boundary or gauge." 
function solve_quantum_vumps_boundaries(peps::PEPSKit.InfinitePEPS, chi::Integer; kwargs...)
    results = map(side -> solve_quantum_vumps_boundary(peps, chi; side, kwargs...), _QUANTUM_SIDES)
    QuantumDirectionalBoundaries(results...)
end

"Measure a one-site physical operator through a saved quantum VUMPS boundary." 
function quantum_one_site_expectation(boundary::QuantumVUMPSBoundary, O)
    boundary.backend === :cpu || throw(ArgumentError(
        "quantum_one_site_expectation currently requires backend=:cpu"))
    top, bottom = boundary.transfer[1]
    codomain(O) == codomain(top) || throw(DimensionMismatch(
        "operator codomain does not match the PEPS physical space"))
    impurity = (O * top, bottom)
    center, left, right = boundary.state.AC[1],
        boundary.environments.GLs[1], boundary.environments.GRs[1]
    numerator = MPSKit.contract_mpo_expval(center, left, impurity, right)
    denominator = MPSKit.contract_mpo_expval(center, left, boundary.transfer[1], right)
    value = numerator / denominator
    Float64(real(value))
end

