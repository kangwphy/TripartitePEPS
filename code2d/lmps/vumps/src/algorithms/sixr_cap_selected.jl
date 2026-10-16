# Opt-in far-cap-selected seam limits.  The legacy Arnoldi endpoint solver and
# its default callers are deliberately not modified by this file.

"Choose the representative of an angle closest to a previously unwrapped angle."
_cap_unwrap_angle(theta::Real, reference::Real) =
    Float64(theta + 2pi * round((reference - theta) / (2pi)))

"Projective distance, used only between independently propagated iterates."
function _cap_projective_distance(x, y)
    nx, ny = norm(x), norm(y)
    nx > 0 && ny > 0 || return Inf
    overlap = dot(vec(x), vec(y))
    phase = abs(overlap) > 0 ? overlap / abs(overlap) : one(overlap)
    norm(y / ny - phase * (x / nx))
end

"A centered least-squares slope, without subtracting two large intercepts."
function _cap_tail_slope(ns, values)
    length(ns) == length(values) >= 2 || throw(ArgumentError("invalid tail window"))
    ts = Float64.(ns .- first(ns))
    ys = values .- first(values)
    tbar = sum(ts) / length(ts)
    ybar = sum(ys) / length(ys)
    denominator = sum((t - tbar)^2 for t in ts)
    sum((ts[j] - tbar) * (ys[j] - ybar) for j in eachindex(ts)) / denominator
end

"""
    cap_selected_power(transfer, boundary; maxiter=1200, tol=1e-10, ...)

Propagate the **specified** finite far cap, rather than selecting an arbitrary
member of a dominant eigenspace.  The phase-aligned normalized recurrence saves
both the accumulated log norm and the accumulated complex phase, so that

    transfer^n(boundary) = exp(lognorm_n + im*phase_n) * x_n.

If the cap-accessible dominant eigenvalue is semisimple and isolated in modulus
from the remaining cap-accessible spectrum, the returned `R * far_prefactor`
approximates `lambda^(-n) * transfer^n(boundary)`.  In particular, exact
same-eigenvalue degeneracy retains the cap's entire mixture.  No eigenvalues
are clustered and no nearly degenerate modes are declared exactly degenerate.
The method does not claim to find sectors to which this cap has zero overlap.

Convergence requires independently evaluated eigen-residual, every single-step
state change in a full window, the change across that window, and stability of
both the magnitude and phase of the asymptotic prefactor.  A second complete
window is needed to compare prefactors.  Different peripheral phases (for
example +lambda and -lambda) generally prevent this one-vector limit; such a
run is rejected rather than averaged or silently replaced by a pure sector.
At finite precision these are numerical convergence tests, not a certificate
that an arbitrarily small spectral splitting is exactly zero.

`far_prefactor` is complex and already includes `prefactor_phase`; do not
multiply that phase a second time.  `log_prefactor` is its log magnitude.
`prefactor_alignment` is a compatibility field containing the independently
measured window state residual, NOT the distance of the final R to itself.
`history` contains scalar diagnostics at window checks, not copies of all
potentially large seam tensors.  With `retain_unconverged=true`, a final finite
diagnostic iterate is returned with `converged=false`; it is not a valid limit.
"""
function cap_selected_power(transfer, boundary::AbstractArray;
        maxiter::Int=1200,
        tol::Real=1e-10,
        state_tolerance::Real=max(1000Float64(tol), 1e-7),
        eigen_tolerance::Real=max(10Float64(tol), sqrt(eps(Float64))),
        amplitude_tolerance::Real=max(100Float64(tol), 1e-9),
        window::Int=24,
        miniter::Int=window,
        retain_unconverged::Bool=false,
        on_progress=nothing,
        progress_every::Int=1000)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    window >= 4 || throw(ArgumentError("window must be at least four"))
    miniter >= 0 || throw(ArgumentError("miniter must be nonnegative"))
    progress_every > 0 || throw(ArgumentError("progress_every must be positive"))
    for (name, value) in ((:tol, tol), (:state_tolerance, state_tolerance),
                         (:eigen_tolerance, eigen_tolerance),
                         (:amplitude_tolerance, amplitude_tolerance))
        isfinite(value) && value > 0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    all(isfinite, boundary) || throw(ArgumentError("far boundary must be finite"))
    initial_norm = norm(boundary)
    isfinite(initial_norm) && initial_norm > 0 ||
        throw(ArgumentError("far boundary must have finite nonzero norm"))
    x = boundary / initial_norm
    anchor = copy(x)
    lognorm = Float64(log(initial_norm))
    phase_accumulated = 0.0
    previous_step_phase = NaN
    phase_defined_in_window = true
    max_step_in_window = 0.0
    window_ns = Int[0]
    window_logs = Float64[lognorm]
    window_phases = Float64[phase_accumulated]
    history = NamedTuple[]
    previous_logpref = NaN
    previous_phasepref = NaN
    converged = false
    failure_reason = "maximum iterations reached without a certified cap-selected limit"
    lambda = ComplexF64(NaN)
    residual = Inf
    state_residual = Inf
    window_state_residual = Inf
    amplitude_drift = Inf
    phase_drift = Inf
    amplitude_slope_match = Inf
    phase_slope_match = Inf
    log_prefactor = NaN
    prefactor_phase = NaN
    far_prefactor = ComplexF64(NaN)
    iterations = 0

    for n in 1:maxiter
        y = transfer(x)
        size(y) == size(x) || throw(DimensionMismatch("transfer changes cap shape"))
        scale = norm(y)
        all(isfinite, y) && isfinite(scale) && scale > 0 ||
            error("cap propagation became nonfinite or zero at iteration $n")
        overlap = dot(vec(x), vec(y))
        phase_defined = abs(overlap) > 100eps(Float64) * scale * norm(x)
        theta = phase_defined ? Float64(angle(overlap)) : 0.0
        if phase_defined && isfinite(previous_step_phase)
            theta = _cap_unwrap_angle(theta, previous_step_phase)
        end
        phase_defined && (previous_step_phase = theta)
        next_x = (y / scale) * cis(-theta)
        step_residual = norm(next_x - x) / max(norm(next_x), norm(x), eps(Float64))
        max_step_in_window = max(max_step_in_window, Float64(step_residual))
        phase_defined_in_window &= phase_defined
        x = next_x
        lognorm += log(scale)
        phase_accumulated += theta
        push!(window_ns, n)
        push!(window_logs, lognorm)
        push!(window_phases, phase_accumulated)
        iterations = n

        # Only one extra transfer action per check.  In particular the returned
        # R itself, not the previous iterate, has its residual evaluated.
        check = n % window == 0 || n == maxiter
        check || continue
        ycheck = transfer(x)
        size(ycheck) == size(x) || throw(DimensionMismatch("transfer changes cap shape"))
        all(isfinite, ycheck) || error("nonfinite cap residual evaluation")
        lambda = ComplexF64(dot(vec(x), vec(ycheck)) / dot(vec(x), vec(x)))
        residual = Float64(norm(ycheck - lambda * x) /
            max(norm(ycheck), abs(lambda) * norm(x), eps(Float64)))
        state_residual = max_step_in_window
        window_state_residual = Float64(_cap_projective_distance(anchor, x))
        valid_lambda = isfinite(lambda) && abs(lambda) > 0
        if valid_lambda
            loglambda = log(abs(lambda))
            theta_lambda = isfinite(previous_step_phase) ?
                _cap_unwrap_angle(angle(lambda), previous_step_phase) : Float64(angle(lambda))
            logs = window_logs .- window_ns .* loglambda
            phases = window_phases .- window_ns .* theta_lambda
            log_prefactor = sum(logs) / length(logs)
            # Unwrapped accumulation makes this an ordinary local mean;
            # differences of intercept phases are tested modulo 2pi below.
            phase_intercept = sum(phases) / length(phases)
            prefactor_phase = Float64(angle(cis(phase_intercept)))
            within_log = maximum(abs(v - log_prefactor) for v in logs)
            within_phase = maximum(abs(angle(cis(v - phase_intercept))) for v in phases)
            amplitude_drift = isfinite(previous_logpref) ?
                max(within_log, abs(log_prefactor - previous_logpref)) : Inf
            phase_drift = isfinite(previous_phasepref) ?
                max(within_phase, abs(angle(cis(prefactor_phase - previous_phasepref)))) : Inf
            amplitude_slope_match = abs(_cap_tail_slope(window_ns, window_logs) - loglambda)
            phase_slope_match = abs(angle(cis(
                _cap_tail_slope(window_ns, window_phases) - theta_lambda)))
            far_prefactor = ComplexF64(exp(log_prefactor) * cis(prefactor_phase))
        else
            amplitude_drift = phase_drift = Inf
            amplitude_slope_match = phase_slope_match = Inf
            log_prefactor = prefactor_phase = NaN
            far_prefactor = ComplexF64(NaN)
        end
        complete_window = length(window_ns) == window + 1
        finite_prefactor = isfinite(far_prefactor) && abs(far_prefactor) > 0
        converged = n >= miniter && complete_window && phase_defined_in_window &&
            valid_lambda && finite_prefactor &&
            residual <= eigen_tolerance && state_residual <= state_tolerance &&
            window_state_residual <= state_tolerance &&
            amplitude_drift <= amplitude_tolerance && phase_drift <= amplitude_tolerance &&
            amplitude_slope_match <= amplitude_tolerance && phase_slope_match <= amplitude_tolerance
        rec = (; iteration=n, lambda, residual, state_residual, window_state_residual,
            amplitude_drift, phase_drift, amplitude_slope_match, phase_slope_match,
            log_prefactor, prefactor_phase, lognorm_accumulated=lognorm,
            phase_accumulated, phase_defined=phase_defined_in_window, converged)
        push!(history, rec)
        if on_progress !== nothing && (length(history) == 1 || converged || n == maxiter ||
                                      n ÷ progress_every != (n-window) ÷ progress_every)
            on_progress(rec)
        end
        if converged
            failure_reason = ""
            break
        end
        previous_logpref = log_prefactor
        previous_phasepref = prefactor_phase
        anchor = copy(x)
        window_ns = Int[n]
        window_logs = Float64[lognorm]
        window_phases = Float64[phase_accumulated]
        max_step_in_window = 0.0
        phase_defined_in_window = true
    end
    if !converged
        failure_reason = "cap-selected limit did not converge in $maxiter iterations; " *
            "state=$state_residual, window=$window_state_residual, eigen=$residual, " *
            "amplitude=$amplitude_drift, phase=$phase_drift; " *
            "possible slow tail or multiple peripheral phases (not automatically classified)"
        retain_unconverged || error(failure_reason)
    end
    (; R=x, lambda, residual, converged, failure_reason,
       far_prefactor, log_prefactor, prefactor_phase,
       state_residual, window_state_residual,
       prefactor_alignment=window_state_residual,
       prefactor_drift=max(amplitude_drift, phase_drift),
       prefactor_slope_match=max(amplitude_slope_match, phase_slope_match),
       prefactor_iters=iterations, amplitude_drift, phase_drift,
       amplitude_slope_match, phase_slope_match,
       lognorm_accumulated=lognorm, phase_accumulated,
       selection_method="normalized_far_cap_power",
       state_tolerance_used=Float64(state_tolerance),
       eigen_tolerance_used=Float64(eigen_tolerance),
       amplitude_tolerance_used=Float64(amplitude_tolerance),
       convergence_window=window, minimum_iterations=miniter,
       maximum_iterations=maxiter, history)
end

"""
    m2_endpoint_cap_selected(Ts; ..., tail_maxiter=nmax)

Opt-in replacement for the seam endpoint solver.  The named-index transfer,
physical far cap and adjoint test are identical to `m2_endpoint`; only the
selection of the asymptotic vector is changed.  It returns legacy fields plus
explicit cap-convergence metadata.  `gap=NaN` and `iters=0` mean no Arnoldi
spectrum/restart count was computed, not evidence for a nondegenerate spectrum.
`krylovdim` is accepted solely for caller compatibility and is not used.
The two supplied far-boundary tolerances are used exactly, without the legacy
`1000tol`/`100tol` floors.  The independent eigen-residual threshold is
`max(10tol, 100eps(Float64))`.  All effective thresholds are returned explicitly.

Call this for BOTH the one-copy metrics and the replica endpoints.  The latter
must receive the existing synchronized `m2_metric_product_boundary` cap made
from the newly selected one-copy metrics, including their complex prefactors.
Do not change only the six replica R while retaining arbitrarily selected
one-copy caps.  Existing scalar scale/intercept identities remain necessary but
do not replace the state-convergence tests here.
"""
function m2_endpoint_cap_selected(Ts::Vector{ITensor};
        nmax::Int=1200, tol::Real=1e-10, krylovdim::Int=40,
        far_boundary=nothing,
        far_boundary_tolerance::Real=1e-7,
        far_boundary_tail_tolerance::Real=1e-9,
        tail_maxiter::Int=nmax,
        retain_unconverged::Bool=false,
        window::Int=24,
        miniter::Int=window,
        verbose::Bool=false)
    nmax > 0 && tail_maxiter > 0 || throw(ArgumentError("iteration limits must be positive"))
    krylovdim > 0 || throw(ArgumentError("krylovdim must be positive"))
    for (name, value) in ((:tol, tol), (:far_boundary_tolerance, far_boundary_tolerance),
                         (:far_boundary_tail_tolerance, far_boundary_tail_tolerance))
        isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    lefts, lmap, rmap = m2_left_right(Ts)
    shape = Tuple(ITensors.dim.(lefts))
    boundary0 = far_boundary === nothing ? m2_corner_boundary(lefts) : far_boundary
    size(boundary0) == shape || throw(DimensionMismatch(
        "far boundary has size $(size(boundary0)), expected $shape"))
    T = foldl(promote_type, (eltype(tensor) for tensor in Ts);
              init=promote_type(Float64, eltype(boundary0)))
    boundary = T.(boundary0)
    peak = Ref(0)
    transfer = x -> begin
        y, p = m2_apply_array(Ts, lefts, lmap, rmap, x)
        peak[] = max(peak[], p)
        y
    end
    adjoint_gate = m2_adjoint_residual(Ts)
    adjoint_gate <= 1e-11 || error("named seam adjoint residual $adjoint_gate exceeds gate")
    progress = verbose ? (rec -> begin
        println("CAP_TRACE n=", rec.iteration, " state=", rec.state_residual,
            " window=", rec.window_state_residual, " eigen=", rec.residual,
            " amplitude=", rec.amplitude_drift, " phase=", rec.phase_drift,
            " converged=", rec.converged)
        flush(stdout)
    end) : nothing
    solved = cap_selected_power(transfer, boundary;
        maxiter=tail_maxiter, tol,
        state_tolerance=Float64(far_boundary_tolerance),
        eigen_tolerance=max(10Float64(tol), 100eps(Float64)),
        amplitude_tolerance=Float64(far_boundary_tail_tolerance),
        window, miniter, retain_unconverged, on_progress=progress)
    # NamedTuple merge keeps the full kernel history available for opt-in
    # artifact writers while preserving every legacy endpoint field.
    merge(solved, (; lefts, gap=NaN, iters=0, peak=peak[],
        left_residual=NaN, left_gap=NaN, left_iters=0, left_match=0,
        adjoint_gate, eigenvalue_match=NaN,
        arnoldi_used=false, alignment_definition="independent_window_state_distance",
        far_cap_norm=Float64(norm(boundary)), far_cap_shape=shape))
end
