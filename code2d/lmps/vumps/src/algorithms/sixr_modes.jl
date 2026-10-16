"""Opt-in campaign dispatch; loading this module never replaces legacy methods."""
module SixRModes

import ..LMPSVUMPS
using LinearAlgebra, ITensors, KrylovKit

export m2_endpoint_legacy_retained, sixr_options

"""
The historical Arnoldi-direction/tail-intercept algorithm, with an explicit
tail budget and retain-last diagnostic option. This is the static equivalent
of the old `direct_longtail_retry.jl` wrapper, not the cap-selected fix. The
legacy public `LMPSVUMPS.m2_endpoint` method is untouched. Failed alignment is
recorded in `converged`; retaining a finite result does not certify convergence.
"""
function m2_endpoint_legacy_retained(Ts::Vector{ITensor}; nmax::Int=1200,
        tol::Real=1e-10, krylovdim::Int=40, far_boundary=nothing,
        far_boundary_tolerance::Real=1e-7, far_boundary_tail_tolerance::Real=1e-9,
        tail_maxiter::Int=nmax, retain_unconverged::Bool=true, verbose::Bool=false)
    min(nmax, tail_maxiter, krylovdim) > 0 || throw(ArgumentError("positive iteration budgets required"))
    all(x -> isfinite(x) && x > 0,
        (tol, far_boundary_tolerance, far_boundary_tail_tolerance)) ||
        throw(ArgumentError("positive finite tolerances required"))
    lefts, lmap, rmap = LMPSVUMPS.m2_left_right(Ts)
    shape = Tuple(ITensors.dim.(lefts))
    boundary0 = far_boundary === nothing ? LMPSVUMPS.m2_corner_boundary(lefts) : far_boundary
    size(boundary0) == shape || throw(DimensionMismatch("far cap shape mismatch"))
    T = foldl(promote_type, (eltype(tensor) for tensor in Ts);
              init=promote_type(Float64, eltype(boundary0)))
    boundary = T.(boundary0)
    norm(boundary) > 0 || throw(ArgumentError("far boundary must be nonzero"))
    x0 = boundary / norm(boundary)
    peak = Ref(0)
    transfer = x -> begin
        y, p = LMPSVUMPS.m2_apply_array(Ts, lefts, lmap, rmap, x)
        peak[] = max(peak[], p)
        y
    end
    adjoint_gate = LMPSVUMPS.m2_adjoint_residual(Ts)
    adjoint_gate <= 1e-11 || error("named seam adjoint residual $adjoint_gate exceeds gate")
    alg = KrylovKit.Arnoldi(; tol=Float64(tol), maxiter=nmax,
        krylovdim=min(prod(shape), krylovdim), verbosity=0)
    vals, vecs, info = KrylovKit.eigsolve(transfer, x0, min(2, prod(shape)), :LM, alg)
    info.converged >= 1 || error("endpoint Arnoldi did not converge")
    lambda = vals[1]
    R = vecs[1] / norm(vecs[1])
    j = argmax(abs.(R)); abs(R[j]) > 0 && (R /= R[j] / abs(R[j]))
    y = transfer(R)
    residual = norm(y - lambda * R) / max(norm(y), abs(lambda) * norm(R), eps(Float64))
    residual <= max(10tol, sqrt(eps(Float64))) || error("endpoint residual $residual exceeds gate")
    gap = length(vals) > 1 ? (abs(vals[1]) - abs(vals[2])) / abs(vals[1]) : NaN
    initial_norm = norm(boundary)
    x = boundary / initial_norm
    lognorm = log(initial_norm)
    ns = Float64[0.0]
    lognorms = Float64[lognorm]
    far_overlap = zero(T)
    prefactor_alignment = prefactor_drift = prefactor_slope_match = Inf
    prefactor_iters = 0
    prefactor_gate = max(100Float64(tol), Float64(far_boundary_tail_tolerance))
    alignment_gate = max(1000Float64(tol), Float64(far_boundary_tolerance))
    fit_window = min(24, nmax)
    fit_window >= 4 || throw(ArgumentError("nmax must be at least 4"))
    history = NamedTuple[]
    for n in 1:tail_maxiter
        yprop = transfer(x)
        scale = norm(yprop)
        isfinite(scale) && scale > 0 || error("far-boundary propagation became nonfinite or zero")
        x = yprop / scale
        lognorm += log(scale)
        push!(ns, n); push!(lognorms, lognorm)
        coefficient = dot(vec(R), vec(x))
        abs(coefficient) > 100eps(Float64) ||
            error("corner far boundary has zero overlap with the dominant seam sector")
        prefactor_alignment = norm(x - coefficient * R) / max(norm(x), abs(coefficient), eps(Float64))
        prefactor_iters = n
        if length(ns) >= fit_window
            xfit = @view ns[(end-fit_window+1):end]
            yfit = @view lognorms[(end-fit_window+1):end]
            xbar, ybar = sum(xfit)/fit_window, sum(yfit)/fit_window
            denominator = sum((u-xbar)^2 for u in xfit)
            slope = sum((xfit[j]-xbar)*(yfit[j]-ybar) for j in eachindex(xfit))/denominator
            intercept = ybar - slope*xbar
            prefactor_drift = maximum(abs(yfit[j]-(intercept+slope*xfit[j])) for j in eachindex(xfit))
            prefactor_slope_match = abs(slope-log(abs(lambda)))
            far_overlap = T(exp(intercept))
            done = prefactor_alignment <= alignment_gate && prefactor_drift <= prefactor_gate
            if n == fit_window || n % 1000 == 0 || n == tail_maxiter || done
                push!(history, (; iteration=n, prefactor_alignment, prefactor_drift, prefactor_slope_match))
                if verbose
                    println("TAIL_TRACE n=", n, " alignment=", prefactor_alignment,
                        " drift=", prefactor_drift, " slope_match=", prefactor_slope_match)
                    flush(stdout)
                end
            end
            done && break
        end
    end
    converged = prefactor_alignment <= alignment_gate && prefactor_drift <= prefactor_gate
    failure_reason = converged ? "" : "legacy far cap did not align in $tail_maxiter iterations"
    converged || retain_unconverged || error(failure_reason)
    abs(far_overlap) > 100eps(Float64) || error("far-boundary tail fit produced a zero prefactor")
    (; R, lefts, lambda, residual, gap, iters=Int(info.numiter), peak=peak[],
       far_prefactor=far_overlap, log_prefactor=log(abs(far_overlap)),
       left_residual=NaN, left_gap=NaN, left_iters=0, left_match=0,
       adjoint_gate, eigenvalue_match=NaN, prefactor_alignment, prefactor_drift,
       prefactor_iters, prefactor_slope_match, converged, failure_reason, history,
       selection_method="legacy_arnoldi_direction_with_far_cap_intercept",
       alignment_definition="distance_to_Arnoldi_direction",
       state_tolerance_used=alignment_gate, amplitude_tolerance_used=prefactor_gate,
       maximum_iterations=tail_maxiter, arnoldi_used=true)
end

"Independent fix switch; symmetry does not enter six-R solver dispatch."
function sixr_options(config)
    fix = get(config, "use_cap_fix", false)
    fix isa Bool || throw(ArgumentError("use_cap_fix must be a boolean"))
    tail = Int(get(config, "sixr_far_tail_maxiter", 240000))
    common = (nmax=Int(get(config,"sixr_maxiter",8000)),
        scale_tolerance=Float64(get(config,"scale_tolerance",5e-8)))
    if fix
        merge(common, (tol=Float64(get(config,"cap_fix_tolerance",1e-12)),
            far_boundary_tolerance=Float64(get(config,"cap_fix_state_tolerance",1e-10)),
            far_boundary_tail_tolerance=Float64(get(config,"cap_fix_amplitude_tolerance",1e-10)),
            endpoint_solver=LMPSVUMPS.m2_endpoint_cap_selected,
            endpoint_options=(tail_maxiter=tail,
                retain_unconverged=Bool(get(config,"cap_fix_retain_unconverged",true)),
                window=Int(get(config,"cap_fix_window",24)),
                miniter=Int(get(config,"cap_fix_miniter",24)),verbose=true)))
    else
        merge(common, (tol=Float64(get(config,"sixr_tolerance",1e-10)),
            far_boundary_tolerance=Float64(get(config,"far_boundary_tolerance",5e-7)),
            far_boundary_tail_tolerance=Float64(get(config,"far_boundary_tail_tolerance",5e-7)),
            endpoint_solver=m2_endpoint_legacy_retained,
            endpoint_options=(tail_maxiter=tail,retain_unconverged=true,verbose=true)))
    end
end

end # module
