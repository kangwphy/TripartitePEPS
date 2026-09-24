# Straight norm-transfer VUMPS boundary.  PEPS ket and bra remain distinct
# TensorKit factors; they are fused only in the explicit dense bridge after the
# VUMPS solve has finished.

"Convert A[s,u,l,d,r] to PEPSKit's A[physical;N,E,S,W]."
function pepskit_from_lmps(state::PEPS)
    A_snesw = permutedims(state.A, (1, 2, 5, 4, 3))
    physical = TensorKit.ComplexSpace(state.d)
    # PEPSKit's local sandwich contracts incoming north/east spaces against
    # outgoing south/west spaces.  They have equal dimensions but opposite
    # TensorKit orientations; using four identical spaces is numerically
    # tempting and wrong (the one-site periodic PEPS constructor rejects it).
    # TensorKit's `ComplexSpace(D)` carries the dual orientation used by
    # PEPSKit for north/east; its adjoint carries the south/west orientation.
    # (The names refer to PEPSKit's reported `virtualspace`, not to whether a
    # factor appears in TensorMap's domain.)
    north_east_virtual = TensorKit.ComplexSpace(state.D)
    south_west_virtual = dual(north_east_virtual)
    tensor = TensorKit.TensorMap(
        A_snesw,
        physical ← north_east_virtual ⊗ north_east_virtual ⊗
                   south_west_virtual ⊗ south_west_virtual,
    )
    PEPSKit.InfinitePEPS(tensor)
end

struct VUMPSBoundaryDiagnostics
    converged::Bool
    galerkin_residual::Float64
    requested_tolerance::Float64
    row_eigenvalue::ComplexF64
    row_logz::Float64
    left_environment_residual::Float64
    right_environment_residual::Float64
    center_residual::Float64
    iterations::Int
end

struct VUMPSBoundary{S,O,E,P}
    state::S
    transfer::O
    environments::E
    source::P
    backend::Symbol
    diagnostics::VUMPSBoundaryDiagnostics
end

_scalarize(value::Number) = value
_scalarize(value) = prod(value)

function _expectation_value(state, transfer, environments)
    states = convert(MPSKit.MultilineMPS, state)
    operators = convert(MPSKit.MultilineMPO, transfer)
    envs = MPSKit.Multiline([environments])
    MPSKit.expectation_value(states, operators, envs)
end

function _scaled_residual(image, target)
    target_norm2 = real(dot(target, target))
    (!isfinite(target_norm2) || target_norm2 <= eps(Float64)) &&
        return Inf
    lambda = dot(target, image) / target_norm2
    denominator = max(norm(image), abs(lambda) * norm(target), eps(Float64))
    Float64(norm(image - lambda * target) / denominator)
end

function _environment_residuals(state, transfer, environments)
    left_transfer = MPSKit.TransferMatrix(state.AL, transfer, state.AL)
    right_transfer = MPSKit.TransferMatrix(state.AR, transfer, state.AR)
    left_target = environments.GLs[1]
    right_target = environments.GRs[end]
    (_scaled_residual(left_target * left_transfer, left_target),
     _scaled_residual(right_transfer * right_target, right_target))
end

function _center_residual(state)
    center = state.AC[1]
    from_left = state.AL[1] * state.C[1]
    from_right = MPSKit._mul_front(state.C[0], state.AR[1])
    left = norm(center - from_left) / max(norm(center), norm(from_left), eps(Float64))
    right = norm(center - from_right) / max(norm(center), norm(from_right), eps(Float64))
    Float64(max(left, right))
end

"Move an uniterated MPSKit GL/GR seed to the same storage as state and MPO."
function _adapt_vumps_environments(to, environments)
    MPSKit.InfiniteEnvironments(
        MPSKit.PeriodicVector([Adapt.adapt(to, tensor) for tensor in environments.GLs]),
        MPSKit.PeriodicVector([Adapt.adapt(to, tensor) for tensor in environments.GRs]),
    )
end

function _prepare_backend(transfer, state, backend::Symbol)
    backend in (:cpu, :cuda) ||
        throw(ArgumentError("backend must be :cpu or :cuda, got $backend"))
    backend === :cpu && return transfer, state, nothing
    CUDA.functional() || error("backend=:cuda requested but CUDA.functional() is false")
    cuTENSOR.functional() || error("backend=:cuda requested but cuTENSOR.functional() is false")
    # MPSKit's default `environments(state_gpu, transfer_gpu)` currently
    # allocates CPU TensorMaps for this PEPS sandwich.  Build only the random
    # GL/GR *seed* in the CPU frame, adapt it before any fixed-point solve, and
    # pass it explicitly to `leading_boundary`.  No environment iteration is
    # performed on the CPU in the CUDA route.
    GLs, GRs = MPSKit.initialize_environments(state, transfer, state)
    environments_cpu = MPSKit.InfiniteEnvironments(GLs, GRs)
    transfer_gpu = Adapt.adapt(CUDA.CuArray, transfer)
    state_gpu = Adapt.adapt(CUDA.CuArray, state)
    environments_gpu = _adapt_vumps_environments(CUDA.CuArray, environments_cpu)
    transfer_gpu, state_gpu, environments_gpu
end

function vumps_storage_backend(boundary::VUMPSBoundary)
    # Inspect an actual TensorMap rather than relying on a container-level
    # `storagetype(::InfiniteMPS)` forwarding method.  Keep the CPU-only load
    # path independent of the CUDA module binding.
    storage = TensorKit.storagetype(typeof(boundary.state.AC[1]))
    if isdefined(@__MODULE__, :CUDA)
        cuda = getfield(@__MODULE__, :CUDA)
        storage <: cuda.CuArray && return :cuda
    end
    :cpu
end

"Solve the dominant boundary MPS of the explicit PEPS ket/bra norm transfer."
function solve_vumps_boundary(
        source::PEPS, chi::Integer;
        backend::Symbol=:cpu,
        tolerance::Real=1e-9,
        diagnostic_tolerance::Real=max(100tolerance, 1e-8),
        max_iterations::Integer=500,
        verbosity::Integer=0,
        accept_unconverged::Bool=false,
    )
    chi > 0 || throw(ArgumentError("chi must be positive"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    diagnostic_tolerance > 0 || throw(ArgumentError("diagnostic_tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))

    peps = pepskit_from_lmps(source)
    transfer_cpu = PEPSKit.InfiniteTransferPEPS(peps, 1, 1)
    state_cpu = PEPSKit.initialize_mps(transfer_cpu, [TensorKit.ComplexSpace(chi)])
    transfer, state0, environments0 = _prepare_backend(transfer_cpu, state_cpu, backend)

    iterations = Ref(0)
    finalize = function (iteration, state, operator, environments)
        iterations[] = max(iterations[], Int(iteration) + 1)
        state, environments
    end
    algorithm = MPSKit.VUMPS(;
        tol=Float64(tolerance),
        maxiter=Int(max_iterations),
        verbosity=Int(verbosity),
        alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false),
        finalize,
    )
    state, environments, galerkin = if environments0 === nothing
        MPSKit.leading_boundary(state0, transfer, algorithm)
    else
        MPSKit.leading_boundary(state0, transfer, algorithm, environments0)
    end
    eigenvalue = ComplexF64(_scalarize(_expectation_value(state, transfer, environments)))
    left_residual, right_residual = _environment_residuals(state, transfer, environments)
    center_residual = _center_residual(state)
    magnitude = abs(eigenvalue)
    finite_scale = isfinite(real(eigenvalue)) && isfinite(imag(eigenvalue)) &&
                   isfinite(magnitude) && magnitude > 0
    converged = finite_scale && isfinite(galerkin) && galerkin <= tolerance &&
                left_residual <= diagnostic_tolerance &&
                right_residual <= diagnostic_tolerance &&
                center_residual <= diagnostic_tolerance
    diagnostics = VUMPSBoundaryDiagnostics(
        converged, Float64(galerkin), Float64(tolerance), eigenvalue,
        finite_scale ? log(magnitude) : NaN,
        left_residual, right_residual, center_residual, iterations[],
    )
    if !converged && !accept_unconverged
        error("VUMPS diagnostics failed: galerkin=$(diagnostics.galerkin_residual), " *
              "GL=$(diagnostics.left_environment_residual), " *
              "GR=$(diagnostics.right_environment_residual), " *
              "center=$(diagnostics.center_residual)")
    end
    result = VUMPSBoundary(state, transfer, environments, source, backend, diagnostics)
    vumps_storage_backend(result) === backend ||
        error("requested backend=$backend but result storage is $(vumps_storage_backend(result))")
    result
end

vumps_logz(boundary::VUMPSBoundary) = boundary.diagnostics.row_logz

