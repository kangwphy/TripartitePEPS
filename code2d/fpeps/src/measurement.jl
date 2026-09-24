"Relative comparison of scaled complex contractions, including their phases."
function _sector_relative_error(a,b)
    s=max(a.logscale,b.logscale)
    x=a.value*exp(a.logscale-s); y=b.value*exp(b.logscale-s)
    abs(x-y)/max(abs(x),abs(y),floatmin(Float64))
end

struct SeamSectorError <: Exception
    name::Symbol
    gap::Float64
end
Base.showerror(io::IO,e::SeamSectorError)=print(io,
    "one-copy seam $(e.name) has no resolved isolated leading state (gap=$(e.gap)); choose compatible boundary sectors or supply an explicit physical cap")

"One-copy mixed caps in the actual regional frames; identities are only seeds."
function solve_seam_caps(seams;seeds=nothing,require_isolated=true,kwargs...)
    results=map(keys(seams)) do name
        X,Y=getproperty(seams,name)
        seed=seeds===nothing ? seam_identity_cap(X,Y) : getproperty(seeds,name)
        result=leading_seam_endpoint(X,Y,(1,),(1,),seed;kwargs...)
        require_isolated && !result.isolated && throw(SeamSectorError(name,result.gap))
        result
    end
    diagnostics=NamedTuple{keys(seams)}(Tuple(results))
    caps=NamedTuple{keys(seams)}(Tuple(r.endpoint for r in results))
    (;caps,diagnostics)
end

function _line_fit(x,y)
    xc=sum(x)/length(x); yc=sum(y)/length(y)
    slope=sum((a-xc)*(b-yc) for (a,b) in zip(x,y))/sum((a-xc)^2 for a in x)
    intercept=yc-slope*xc
    residual=maximum(abs(b-intercept-slope*a) for (a,b) in zip(x,y))
    (;slope,intercept,residual)
end

"Residual weighted by each signed term's actual contribution; no term is dropped."
function _active_endpoint_residual(sector)
    denominator=log(abs(sector.value))+sector.logscale
    sum(sector.sectors) do term
        z=term.result
        iszero(z.value) && return 0.0
        weight=exp(log(abs(term.weight*z.value))+z.logscale-denominator)
        weight*maximum(d.residual for d in z.diagnostics)
    end
end

"""Measure the specified finite-chi LMPS network over increasing seam lengths.

All five raw sectors share the same caps, lengths and parity compiler. The
six-R propagation retains complex amplitudes and uses no eigenvector phase
replacement. Near-null parity terms are kept with their actual log weights.
`sector_builder` defaults to the full signed expansion; an alternative must
return the same five scaled complex sectors and their endpoint diagnostics.
When `audit_depth` is supplied it is checked against the full four-copy graph.
The limit gate checks three successive S_tilde values, linear tails of all
five log magnitudes, and contribution-weighted endpoint residuals. It tests
the length limit of this supplied network, not a chi extrapolation or the
physical accuracy of its environment. `caps` are part of the definition.

`audit_depth` compares the full four-copy seams against cycle factorization;
this independent contraction is intended for small chi because its endpoints
have eight retained legs. Set it to nothing after separate validation.
"""
function measure_lmps_stilde(seams,caps;depths=(8,16,32,64),tolerance=1e-7,
                             phase_tolerance=1e-7,max_cancellation=1e8,
                             endpoint_tolerance=1e-6,audit_depth=nothing,
                             audit_tolerance=1e-10,accept_unconverged=false,
                             on_sample=(_->nothing),sector_builder=finite_lmps_sectors)
    ds=collect(depths)
    length(ds)>=3 && all(d->d isa Int && d>=0,ds) && all(diff(ds).>0) ||
        throw(ArgumentError("at least three strictly increasing nonnegative integer depths required"))
    all(t->isfinite(t)&&t>0,(tolerance,endpoint_tolerance,audit_tolerance)) ||
        throw(ArgumentError("finite positive measurement tolerances required"))
    audit=nothing
    if audit_depth!==nothing
        audit_depth isa Int && audit_depth>=0 || throw(ArgumentError("invalid audit depth"))
        full=finite_lmps_sectors(seams,caps;depth=audit_depth,factorized=false)
        six=sector_builder(seams,caps;depth=audit_depth)
        errors=NamedTuple{keys(full)}(Tuple(_sector_relative_error(a,b) for (a,b) in zip(values(full),values(six))))
        maximum(values(errors))<=audit_tolerance || error("full/cycle replica audit failed: $errors")
        audit=(;depth=audit_depth,errors)
    end
    cache=Dict{Any,Any}(); samples=NamedTuple[]
    for depth in ds
        sectors=sector_builder(seams,caps;depth,propagation_cache=cache)
        entropies=lmps_sector_entropies(sectors;phase_tolerance,max_cancellation)
        logs=NamedTuple{keys(sectors)}(Tuple(log(abs(z.value))+z.logscale for z in values(sectors)))
        residual=maximum(_active_endpoint_residual(z) for z in values(sectors))
        sample=(;depth,entropies,logs,endpoint_residual=residual,sectors)
        push!(samples,sample)
        on_sample(sample)
    end
    tail=samples[end-2:end]
    drift=maximum(abs(tail[j].entropies.stilde-tail[j-1].entropies.stilde) for j in 2:3)
    fits=NamedTuple{keys(last(samples).logs)}(Tuple(
        _line_fit([s.depth for s in tail],[getproperty(s.logs,name) for s in tail])
        for name in keys(last(samples).logs)))
    linear_residual=maximum(f.residual for f in values(fits))
    endpoint_residual=last(samples).endpoint_residual
    converged=drift<=tolerance && linear_residual<=tolerance && endpoint_residual<=endpoint_tolerance
    !converged && !accept_unconverged && error(
        "LMPS length limit failed: S_tilde drift=$drift, log-tail residual=$linear_residual, endpoint residual=$endpoint_residual; increase depths or inspect the environment")
    (;stilde=last(samples).entropies.stilde,converged,samples,fits,audit,
      diagnostics=(;drift,linear_residual,endpoint_residual),
      convention=:occupation_ABC,network=:finite_chi_oriented_lmps,
      caps,depths=ds,tolerance,phase_tolerance,endpoint_tolerance)
end

"""Fixed KSVC fPEPS -> three regional VUMPS environments -> signed LMPS S_tilde.

A and B use the same north fixed-point branch by default; their left/right
rails remain distinct. C is a separate south solve. This identifies identical
north transfers and does not assume rotation or reflection of the PEPS.
`independent_north=true` audits separate seeds, which can select incompatible
graded boundary sectors; unresolved mixed caps then fail explicitly.
Supplied `boundaries` may reuse saved, converged checkpoints;
each must carry its physical PEPS and direction. `caps` can specify a chosen
far-boundary sector; otherwise isolated one-copy mixed fixed points are used.
Convergence here is at fixed chi and must be checked against larger chi.
"""
function _prepare_fpeps_stilde(peps;chi=(2,2),seed=1729,
                            boundaries=nothing,caps=nothing,independent_north=false,
                            boundary_kwargs=(;tolerance=1e-8),
                            cap_kwargs=(;))
    bs=if boundaries===nothing
        A=solve_boundary(peps;chi,direction=1,seed,boundary_kwargs...)
        B=independent_north ? solve_boundary(peps;chi,direction=1,seed=seed+1,boundary_kwargs...) : A
        C=solve_boundary(peps;chi,direction=3,seed,boundary_kwargs...)
        (;A,B,C)
    else
        boundaries
    end
    for (name,direction) in ((:A,1),(:B,1),(:C,3))
        b=getproperty(bs,name)
        b.converged || throw(ArgumentError("unconverged boundary $name"))
        b.direction==direction || throw(ArgumentError("wrong boundary direction for $name"))
        b.chi==chi || throw(ArgumentError("boundary $name has different chi"))
        size(b.peps)==size(peps) && all(b.peps[i]≈peps[i] for i in eachindex(peps)) ||
            throw(ArgumentError("boundary $name belongs to a different physical PEPS"))
    end
    objects=map(lmps_objects,bs)
    rails=map(fermionic_rails,objects)
    seams=regional_seam_rails(rails)
    # Do not interpret splitting below environment accuracy as a physical gap.
    cap_options=merge((;gap_tolerance=max(1e-8,1000maximum(b.galerkin for b in bs))),cap_kwargs)
    cap_solve=caps===nothing ? solve_seam_caps(seams;cap_options...) : nothing
    physical_caps=caps===nothing ? cap_solve.caps : caps
    (;boundaries=bs,objects,rails,seams,caps=physical_caps,cap_solve,chi,
      density=map(boundary_number,bs),
      environment_converged=all(b.converged for b in bs))
end

"""Prepare compatible regional environments without evaluating any entropy.

For fresh solves, unresolved seam sectors trigger a bounded new seeded solve.
The selection criterion is the one-copy spectral gap relative to the VUMPS
residual, never a desired S_tilde. Every attempt is reported. Saved boundaries
are checked as supplied and are never replaced by random states.
"""
function prepare_fpeps_stilde(peps=ksvc_ipeps();seed=1729,boundaries=nothing,
                              boundary_trials=8,on_attempt=(_->nothing),kwargs...)
    boundary_trials isa Int && boundary_trials>0 || throw(ArgumentError("positive boundary_trials required"))
    attempts=NamedTuple[]
    for trial in 1:(boundaries===nothing ? boundary_trials : 1)
        trial_seed=seed+trial-1
        prepared=try
            _prepare_fpeps_stilde(peps;seed=trial_seed,boundaries,kwargs...)
        catch err
            err isa SeamSectorError || rethrow()
            attempt=(;trial,seed=boundaries===nothing ? trial_seed : nothing,
                      status=:unresolved_seam_sector,error=sprint(showerror,err))
            push!(attempts,attempt); on_attempt(attempt)
            boundaries!==nothing && rethrow()
            continue
        end
        attempt=(;trial,seed=boundaries===nothing ? trial_seed : nothing,status=:accepted,error="")
        push!(attempts,attempt); on_attempt(attempt)
        return (;prepared...,attempts,selected_seed=attempt.seed)
    end
    error("no compatible regional boundary sectors in $boundary_trials attempts: $(last(attempts).error)")
end

"""Legacy stationary-rail LMPS diagnostic, retained for compatibility.

This network failed the independent gapped Gaussian entropy benchmark even
when its length checks passed. Use `measure_finite_fpeps_stilde` for physical
finite-window region sewing and check size and chi separately.
"""
function solve_fpeps_stilde(peps=ksvc_ipeps();measurement_kwargs=(;),kwargs...)
    prepared=prepare_fpeps_stilde(peps;kwargs...)
    result=measure_lmps_stilde(prepared.seams,prepared.caps;measurement_kwargs...)
    (;result,prepared...)
end
