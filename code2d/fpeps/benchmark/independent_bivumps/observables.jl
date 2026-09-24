# New-schema adapter of the unchanged signed physical measurement.
# Upstream measure_joint_observables.jl sha256=13ba244828da876881e52cb3254bc5d118dd2af56d1c2331069ad9440ed9cd09
# Joint boundary diagnostics at a fixed target chi, up to 32.
# Reuses the audited physical caps/sign conventions. No operator-mode xi is
# reported here; complete caps, RDMs, and direct correlators are measured.
include("../stability_observable_modes.jl")
using SHA, DelimitedFiles

function measure_joint_observables(source,output)
    BLAS.set_num_threads(parse(Int,get(ENV,"SLURM_CPUS_PER_TASK","2")))
    digest(file)=bytes2hex(sha256(read(file)))
    files=[joinpath(source,f) for f in ("boundary_1.jls","boundary_3.jls")]
    hashes=Dict(file=>digest(file) for file in files)
    n,s=deserialize.(files);chi=sum(n.chi)
    iseven(chi) && 2<=chi<=32 && sum(s.chi)==chi || error("bounded even 2<=chi<=32")
    n.direction==1 && s.direction==3 && n.peps[1]≈s.peps[1] || error("invalid opposite pair")
    isfile(joinpath(output,"correlations.toml")) && error("refusing overwrite")
    mkpath(output)
    peps,env,partial,converged=diagnostic_caps(source;full_spectrum=true)
    !partial || error("partial physical spectrum")
    rdms=Dict{String,Any}[];density=0.0im
    for sites in (1,2,3)
        rho=PEPSKit.reduced_densitymatrix(Tuple((1,j) for j in 1:sites),peps,env)
        physical=reshape(convert(Array,rho),2^sites,2^sites)*Diagonal([(-1)^count_ones(i) for i in 0:2^sites-1])
        writedlm(joinpath(output,"rho$(sites)_real.csv"),real.(physical),',')
        writedlm(joinpath(output,"rho$(sites)_imag.csv"),imag.(physical),',')
        sites==1 && (density=physical[2,2])
        push!(rdms,Dict("sites"=>sites,"trace_error"=>abs(tr(physical)-1),
            "hermiticity_error"=>norm(physical-physical'),"min_eigenvalue"=>eigmin(Hermitian((physical+physical')/2))))
    end
    distances=collect(1:128);site=CartesianIndex(1,1)
    targets=[CartesianIndex(1,r+1) for r in distances]
    measurements=Dict{String,Any}()
    for name in ("normal","anomalous","nn")
        values=MPSKit.correlator(peps,signed_operators(space(peps[1],1))[name],site,targets,env)
        all(isfinite,values) || error("nonfinite correlator")
        measurements[name]=Dict("real"=>real.(values),"imag"=>imag.(values))
    end
    joint=TOML.parsefile(joinpath(source,"report.toml"))
    joint["schema"]=="independent_bivumps_v1" || error("wrong solver schema")
    all(digest(file)==hash for (file,hash) in hashes) || error("source changed")
    row=Dict("chi"=>chi,"source"=>source,"source_hashes"=>hashes,
        "bivumps_converged"=>joint["converged"],"bivumps_residual"=>joint["final"]["outer_residual"],
        "operator_mode_spectrum_measured"=>false,"native_converged"=>converged,"north_native_residual"=>n.galerkin,
        "south_native_residual"=>s.galerkin,"complete_unique_physical_caps"=>true,
        "density_real"=>real(density),"density_imag"=>imag(density),
        "distances"=>distances,"measurements"=>measurements,"rdms"=>rdms)
    open(joinpath(output,"correlations.toml"),"w") do io
        TOML.print(io,Dict("complete"=>true,"accepted_entropy"=>false,
            "script_sha256"=>digest(@__FILE__),"cases"=>[row]))
    end
    println("chi=",chi," native residuals=",(n.galerkin,s.galerkin)," physical benchmark complete")
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    measure_joint_observables(ARGS[1],ARGS[2])
end
