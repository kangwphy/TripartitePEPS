# Quantum-model extension: PEPS ground-state preparation followed by VUMPS
# boundary contractions.  Nothing in this file is used by the RK-Ising or
# six-R entry points; the public functions below are additive.

"""
    TransverseIsing2D(; J=1.0, h=1.0, unitcell=(2, 2))

Parameters for the square-lattice transverse-field Ising Hamiltonian

```text
H = -J sum_<i,j> Z_i Z_j - h sum_i X_i.
```

A one-site cell is allowed for the translationally invariant TFIM.  A larger
cell can represent an explicitly broken ansatz, but the generic six-R bridge
must first block the complete transfer period.
"""
struct TransverseIsing2D{T<:Real}
    J::T
    h::T
    unitcell::Tuple{Int,Int}
    function TransverseIsing2D{T}(J::T, h::T, unitcell::Tuple{Int,Int}) where {T<:Real}
        isfinite(J) && isfinite(h) || throw(ArgumentError("J and h must be finite"))
        !iszero(J) || throw(ArgumentError("J must be nonzero so h/J is defined"))
        all(>(0), unitcell) || throw(ArgumentError(
            "unitcell entries must be positive"
        ))
        new{T}(J, h, unitcell)
    end
end

function TransverseIsing2D(; J::Real=1.0, h::Real=1.0, unitcell::Tuple{Int,Int}=(2, 2))
    T = promote_type(typeof(float(J)), typeof(float(h)))
    return TransverseIsing2D{T}(T(J), T(h), unitcell)
end

"Construct the local-operator Hamiltonian for `model` without CTMRG." 
function tfim_hamiltonian(model::TransverseIsing2D; scalar_type::Type{<:Number}=ComplexF64)
    lattice = PEPSKit.InfiniteSquare(model.unitcell...)
    return PEPSKit.transverse_field_ising(
        scalar_type, TensorKit.Trivial, lattice; J=model.J, g=model.h / model.J,
    )
end

"Construct a seeded, unconstrained complex PEPS used as the SimpleUpdate start." 
function tfim_initial_peps(
        model::TransverseIsing2D; D::Integer=2, seed::Integer=0,
        scalar_type::Type{<:Number}=ComplexF64,
    )
    D > 0 || throw(ArgumentError("D must be positive"))
    scalar_type <: Number || throw(ArgumentError("scalar_type must be numeric"))
    rng = Random.MersenneTwister(seed)
    physical = TensorKit.ComplexSpace(2)
    bond = TensorKit.ComplexSpace(D)
    # TensorKit's constructor calls the supplied function as f(T, space).
    draw = (T, space) -> randn(rng, T, space)
    PEPSKit.InfinitePEPS(
        draw, scalar_type, physical, bond; unitcell=model.unitcell,
    )
end

"""
    SimpleUpdateGroundState

Container for the PEPS produced by imaginary-time SimpleUpdate.  The method
is deliberately separated from the boundary solver: a generic quantum model
can supply its own `InfinitePEPS` and `LocalOperator`, then call
`solve_simple_update_groundstate` followed by the VUMPS boundary routines.
`truncation_errors` records the final gate truncation error of each stage; it
is not a variational energy estimate.
"""
struct SimpleUpdateGroundState{P,W,H}
    peps::P
    weights::W
    hamiltonian::H
    dt_schedule::Tuple
    nstep::Int
    truncation_errors::Vector{Float64}
end

"A one-site variational PEPS ground state and its optimized CTMRG environment."
struct VariationalGroundState{P,E,H,I,W}
    peps::P
    environment::E
    hamiltonian::H
    optimization_info::I
    energy::Float64
    gradient_norm::Float64
    converged::Bool
    D::Int
    environment_chi::Int
    tolerance::Float64
    max_iterations::Int
    initialization::Symbol
    initialization_info::W
end

"""
    materialize_simple_update_state(result)

Return the physical PEPS produced by PEPSKit's SimpleUpdate.

In PEPSKit 0.8, `result.peps` already contains the square-root singular values
inserted by each bond update.  `result.weights` is the SimpleUpdate environment
used to dress the *next* local gate and to monitor convergence; it is not a set
of missing network factors.  Absorbing it once more would double-count the
bond weights.  This named helper is retained so older callers remain explicit
about crossing from the SU result container to the physical PEPS.
"""
function materialize_simple_update_state(result::SimpleUpdateGroundState)
    return result.peps
end

"""
    solve_simple_update_groundstate(peps, H; kwargs...)

Prepare a ground-state candidate with imaginary-time SimpleUpdate.  This
route avoids CTMRG and can be used either as a separately labeled baseline or
as the initial state of the variational/autodiff solver below.

If `stage_iterations_out` is an integer vector, it is replaced with the number
of sweeps actually executed for each entry of `dt_schedule`.  A positive
`tolerance` may stop a stage before the configured `nstep` maximum.
"""
function solve_simple_update_groundstate(
        peps::PEPSKit.InfinitePEPS, H::PEPSKit.LocalOperator;
        dt_schedule=(1e-2, 1e-3, 1e-4),
        nstep::Integer=1000,
        tolerance::Real=1e-8,
        truncation=PEPSKit.FixedSpaceTruncation(),
        bipartite::Bool=false,
        symmetrize_gates::Bool=false,
        verbosity::Integer=0,
        check_interval::Integer=0,
        stage_iterations_out::Union{Nothing,AbstractVector{Int}}=nothing,
    )
    isempty(dt_schedule) && throw(ArgumentError("dt_schedule cannot be empty"))
    all(isfinite, dt_schedule) && all(>(0), dt_schedule) ||
        throw(ArgumentError("dt_schedule must contain positive finite steps"))
    nstep > 0 || throw(ArgumentError("nstep must be positive"))
    tolerance >= 0 || throw(ArgumentError("tolerance must be nonnegative"))
    check_interval >= 0 || throw(ArgumentError("check_interval must be nonnegative"))
    all(>=(2), size(peps)) || throw(ArgumentError(
        "PEPSKit SimpleUpdate requires a unit cell of at least (2, 2)"
    ))

    state = peps
    weights = PEPSKit.SUWeight(state)
    alg = PEPSKit.SimpleUpdate(; trunc=truncation, imaginary_time=true,
        bipartite=bipartite)
    actual_stage_iterations = Int[]
    errors = Float64[]
    for dt in dt_schedule
        # With check_interval=0 PEPSKit performs no CTMRG energy logging.  A
        # positive tolerance still checks the local SU-weight fixed point.
        state, weights, info = MPSKit.time_evolve(
            state, H, dt, Int(nstep), alg, weights;
            symmetrize_gates, tol=Float64(tolerance), verbosity=Int(verbosity),
            check_interval=Int(check_interval),
        )
        iterations = round(Int, real(info.t / dt))
        1 <= iterations <= nstep || error(
            "PEPSKit returned invalid SU evolution time t=$(info.t) for dt=$dt"
        )
        push!(actual_stage_iterations, iterations)
        push!(errors, Float64(info.ϵ))
    end
    if !isnothing(stage_iterations_out)
        empty!(stage_iterations_out)
        append!(stage_iterations_out, actual_stage_iterations)
    end
    SimpleUpdateGroundState(state, weights, H, Tuple(dt_schedule), Int(nstep), errors)
end

"""
    quantum_one_site_representative(peps; site=(1, 1))

Extract one tensor from a multi-site PEPS and repeat it as a one-site state.
This is an explicit translationally invariant approximation, not an exact
blocking of a multi-site unit cell.  It is useful for the present one-site
quantum six-R prototype after `materialize_simple_update_state`; full
multi-site support requires a
multiline VUMPS boundary and a complete two-dimensional block.
"""
function quantum_one_site_representative(
        peps::PEPSKit.InfinitePEPS; site::Tuple{Int,Int}=(1, 1)
    )
    all(>(0), site) || throw(ArgumentError("site entries must be positive"))
    1 <= site[1] <= size(peps, 1) && 1 <= site[2] <= size(peps, 2) ||
        throw(BoundsError(peps, site))
    PEPSKit.InfinitePEPS(peps[site...])
end

function solve_tfim_groundstate(
        model::TransverseIsing2D;
        D::Integer=2, seed::Integer=0, scalar_type::Type{<:Number}=ComplexF64,
        kwargs...
    )
    peps = tfim_initial_peps(model; D, seed, scalar_type)
    H = tfim_hamiltonian(model; scalar_type)
    solve_simple_update_groundstate(peps, H; kwargs...)
end

function _tfim_variational_initial_state(
        model::TransverseIsing2D;
        D::Integer, seed::Integer, scalar_type::Type{<:Number},
        initialization::Symbol,
        simple_update_unitcell::Tuple{Int,Int},
        simple_update_site::Tuple{Int,Int},
        simple_update_kwargs::NamedTuple,
    )
    initialization in (:random, :simple_update) || throw(ArgumentError(
        "initialization must be :random or :simple_update, got $initialization"
    ))
    if initialization === :random
        local initial
        seconds = @elapsed initial = tfim_initial_peps(model; D, seed, scalar_type)
        info = (;
            method=:random,
            seconds,
            auxiliary_unitcell=nothing,
            representative_site=nothing,
            simple_update_dt_schedule=(),
            simple_update_nstep=0,
            simple_update_stage_iterations=Int[],
            simple_update_truncation_errors=Float64[],
        )
        return initial, info
    end

    all(>=(2), simple_update_unitcell) || throw(ArgumentError(
        "PEPSKit SimpleUpdate requires simple_update_unitcell >= (2,2)"
    ))
    all(>(0), simple_update_site) &&
        simple_update_site[1] <= simple_update_unitcell[1] &&
        simple_update_site[2] <= simple_update_unitcell[2] || throw(ArgumentError(
            "simple_update_site=$simple_update_site is outside " *
            "simple_update_unitcell=$simple_update_unitcell"
        ))
    for forbidden in (:D, :seed, :scalar_type, :stage_iterations_out)
        haskey(simple_update_kwargs, forbidden) && throw(ArgumentError(
            "simple_update_kwargs must not override $forbidden"
        ))
    end

    # PEPSKit's SimpleUpdate does not support a one-site time-evolution cell.
    # Evolve an auxiliary cell and use one deterministic local tensor only as
    # the starting point of the one-site
    # variational problem.  This is a warm start, not an exact reduction of the
    # auxiliary multi-site wavefunction.
    auxiliary_model = TransverseIsing2D(;
        J=model.J, h=model.h, unitcell=simple_update_unitcell,
    )
    local simple_update
    stage_iterations = Int[]
    seconds = @elapsed simple_update = solve_tfim_groundstate(
        auxiliary_model; D, seed, scalar_type, stage_iterations_out=stage_iterations,
        simple_update_kwargs...,
    )
    materialized = materialize_simple_update_state(simple_update)
    initial = quantum_one_site_representative(materialized; site=simple_update_site)
    info = (;
        method=:simple_update,
        seconds,
        auxiliary_unitcell=simple_update_unitcell,
        representative_site=simple_update_site,
        simple_update_dt_schedule=simple_update.dt_schedule,
        simple_update_nstep=simple_update.nstep,
        simple_update_stage_iterations=stage_iterations,
        simple_update_truncation_errors=copy(simple_update.truncation_errors),
    )
    initial, info
end

"""
    solve_tfim_variational_groundstate(model; D=2, environment_chi=8,
                                        initialization=:random, ...)

Optimize a genuine one-site TFIM iPEPS using PEPSKit's differentiable
fixed-point solver.  CTMRG is used only inside this ground-state optimization;
the returned PEPS can subsequently be measured by the independent LMPS/VUMPS
six-R route.  No spin-flip symmetry is imposed, so the optimized state may be
symmetry broken.

`initialization=:random` preserves the seeded random one-site start.
`initialization=:simple_update` first evolves an auxiliary cell of size
`simple_update_unitcell` (at least `(2,2)`) and uses `simple_update_site` as a
one-site warm start.  PEPSKit cannot perform
one-site SU, so this local representative is not an exact reduction of the
auxiliary state; it is subsequently refined by the full differentiable
one-site optimization.  Evolution controls are passed in
`simple_update_kwargs`, for example
`(; dt_schedule=(1e-2,1e-3), nstep=500, tolerance=1e-8)`.
"""
function solve_tfim_variational_groundstate(
        model::TransverseIsing2D;
        D::Integer=2, environment_chi::Integer=8, seed::Integer=0,
        tolerance::Real=1e-3, scalar_type::Type{<:Number}=ComplexF64,
        max_iterations::Integer=100, spatial_symmetrization=nothing,
        initialization::Symbol=:random,
        simple_update_unitcell::Tuple{Int,Int}=(2, 2),
        simple_update_site::Tuple{Int,Int}=(1, 1),
        simple_update_kwargs::NamedTuple=(;),
    )
    model.unitcell == (1, 1) || throw(ArgumentError(
        "the current variational TFIM solver requires unitcell=(1,1)"
    ))
    D > 0 || throw(ArgumentError("D must be positive"))
    environment_chi > 0 || throw(ArgumentError("environment_chi must be positive"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    initial, initialization_info = _tfim_variational_initial_state(
        model; D, seed, scalar_type, initialization, simple_update_unitcell,
        simple_update_site, simple_update_kwargs,
    )
    if !isnothing(spatial_symmetrization)
        # OptimKit assumes that the initial point already lies on the selected
        # symmetry manifold. Symmetrizing only gradients/retractions leaves a
        # generic random PEPS off-manifold and can stall the first line search.
        initial = PEPSKit.peps_normalize(
            PEPSKit.symmetrize!(initial, spatial_symmetrization),
        )
    end
    hamiltonian = tfim_hamiltonian(model; scalar_type)
    environment0 = PEPSKit.CTMRGEnv(
        initial, TensorKit.ComplexSpace(environment_chi),
    )
    environment0, = PEPSKit.leading_boundary(environment0, initial)
    result = PEPSKit.fixedpoint(
        hamiltonian, initial, environment0;
        tol=Float64(tolerance),
        optimizer_alg=(; tol=Float64(tolerance), maxiter=Int(max_iterations)),
        symmetrization=spatial_symmetrization,
    )
    peps, environment, energy, info = result[1], result[2], result[3], result[4]
    gradient_norm = isempty(info.gradnorms) ? Inf : Float64(last(info.gradnorms))
    VariationalGroundState(
        peps, environment, hamiltonian, info, Float64(real(energy)), gradient_norm,
        gradient_norm <= tolerance, Int(D), Int(environment_chi), Float64(tolerance),
        Int(max_iterations), initialization, initialization_info,
    )
end

"Measure TFIM one-site spin components with the optimized CTMRG environment."
function tfim_groundstate_observables(result::VariationalGroundState)
    return tfim_peps_observables(result.peps, result.hamiltonian, result.environment)
end

"""
    tfim_peps_observables(peps, hamiltonian, environment)

Measure the TFIM energy and the unit-cell-averaged Pauli components of an
arbitrary PEPS unit cell using an already converged CTMRG environment.  The
Hamiltonian expectation is divided by the number of sites in the cell; this is
essential for comparing a `(2,2)` SimpleUpdate state with a `(2,2)` AD state.
The site-resolved values are returned as matrices so a checkerboard or an
accidentally mismatched branch cannot be hidden by the average.
"""
function tfim_peps_observables(
        peps::PEPSKit.InfinitePEPS, hamiltonian::PEPSKit.LocalOperator,
        environment::PEPSKit.CTMRGEnv,
    )
    size(peps) == size(hamiltonian) || throw(DimensionMismatch(
        "PEPS size $(size(peps)) does not match Hamiltonian size $(size(hamiltonian))"
    ))
    physical_spaces = PEPSKit.physicalspace(hamiltonian)
    sites = collect(CartesianIndices(size(peps)))
    x_sites = Matrix{Float64}(undef, size(peps))
    z_sites = Matrix{Float64}(undef, size(peps))
    for site in sites
        physical = physical_spaces[site]
        X = TensorKit.TensorMap(ComplexF64[0 1; 1 0], physical ← physical)
        Z = TensorKit.TensorMap(ComplexF64[1 0; 0 -1], physical ← physical)
        Xop = PEPSKit.LocalOperator(physical_spaces, (site,) => X)
        Zop = PEPSKit.LocalOperator(physical_spaces, (site,) => Z)
        x_sites[site] = Float64(real(PEPSKit.expectation_value(peps, Xop, environment)))
        z_sites[site] = Float64(real(PEPSKit.expectation_value(peps, Zop, environment)))
    end
    nsites = length(sites)
    energy_complex = PEPSKit.expectation_value(peps, hamiltonian, environment) /
        nsites
    energy = Float64(real(energy_complex))
    energy_imag = Float64(imag(energy_complex))
    return (;
        energy,
        energy_imag,
        x=sum(x_sites) / nsites,
        z=sum(z_sites) / nsites,
        abs_z=sum(abs, z_sites) / nsites,
        x_sites,
        z_sites,
    )
end

