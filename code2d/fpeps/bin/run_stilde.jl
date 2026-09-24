# sbatch jobs/run_cpu.sh bin/run_stilde.jl EVEN ODD NEW_OUTPUT [BOUNDARIES]
# BOUNDARIES optionally names a previous run's directory containing A/B/C.jls.
include("../src/FermionicPEPS.jl")
using .FermionicPEPS, Serialization, TOML, SHA, Dates, Printf

function atomic_toml(path,data)
    open(path*".partial","w") do io; TOML.print(io,data); end
    mv(path*".partial",path;force=true)
end

function main(args)
    length(args) in (3,4) || error("usage: run_stilde.jl EVEN ODD NEW_OUTPUT [BOUNDARIES]")
    chi=(parse(Int,args[1]),parse(Int,args[2]))
    FermionicPEPS._environment_space(chi)
    output=abspath(args[3])
    ispath(output) && error("output already exists: $output")
    depths=parse.(Int,split(get(ENV,"FPEPS_DEPTHS","8,16,32,64"),','))
    tolerance=parse(Float64,get(ENV,"FPEPS_TOLERANCE","1e-7"))
    boundary_tolerance=parse(Float64,get(ENV,"FPEPS_BOUNDARY_TOLERANCE","1e-8"))
    boundary_maxiter=parse(Int,get(ENV,"FPEPS_BOUNDARY_MAXITER","1000"))
    boundary_trials=parse(Int,get(ENV,"FPEPS_BOUNDARY_TRIALS","8"))
    seed=parse(Int,get(ENV,"FPEPS_SEED","1729"))
    weights=parse.(Float64,split(get(ENV,"FPEPS_BOND_WEIGHT","1,1"),','))
    length(weights)==1 && (weights=repeat(weights,2))
    length(weights)==2 && all(isfinite,weights) || error("FPEPS_BOND_WEIGHT needs one or two finite real weights")
    bond_weight=Tuple(weights)
    rotate_180=get(ENV,"FPEPS_ROTATE_180","false")=="true"
    independent_north=get(ENV,"FPEPS_INDEPENDENT_NORTH","false")=="true"
    accept_unconverged=get(ENV,"FPEPS_ACCEPT_UNCONVERGED","false")=="true"
    root=normpath(joinpath(@__DIR__,".."))
    mkpath(joinpath(output,"boundaries"))
    provenance=Dict{String,Any}(
        "created_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
        "model"=>"KSVC exact D=2 Q projector", "convention"=>"occupation_ABC",
        "physical_entropy_validated"=>false,"method"=>"stationary_lmps_diagnostic",
        "physical_benchmark_status"=>"failed_gapped_gaussian_comparison_use_run_finite_stilde",
        "chi_even"=>chi[1],"chi_odd"=>chi[2],"depths"=>depths,"seed"=>seed,
        "bond_weight"=>weights,"rotate_180"=>rotate_180,
        "independent_north"=>independent_north,
        "tolerance"=>tolerance,"boundary_tolerance"=>boundary_tolerance,
        "boundary_maxiter"=>boundary_maxiter,"accept_unconverged"=>accept_unconverged,
        "boundary_trials"=>boundary_trials,
        "source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(root,"src",f))))
            for f in readdir(joinpath(root,"src")) if endswith(f,".jl")),
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(root,"environments","default","Manifest.toml")))),
        "status"=>"running","environment_extrapolated"=>false)
    metadata_path=joinpath(output,"run.toml")
    atomic_toml(metadata_path,provenance)
    try
        peps=ksvc_ipeps(;bond_weight)
        rotate_180 && (peps=rot180(peps))
        bs=Any[]
        if length(args)==4
            for name in (:A,:B,:C)
                path=abspath(joinpath(args[4],"$name.jls"))
                provenance["boundary_$(name)_source"]=path
                provenance["boundary_$(name)_sha256"]=bytes2hex(sha256(read(path)))
                push!(bs,deserialize(path))
            end
        end
        attempt_log=Dict{String,Any}[]
        prepared=prepare_fpeps_stilde(peps;chi,seed,independent_north,boundary_trials,
            boundaries=isempty(bs) ? nothing : (A=bs[1],B=bs[2],C=bs[3]),
            boundary_kwargs=(;tolerance=boundary_tolerance,maxiter=boundary_maxiter),
            on_attempt=a->begin
                row=Dict{String,Any}("trial"=>a.trial,"status"=>string(a.status),"error"=>a.error)
                a.seed!==nothing && (row["seed"]=a.seed)
                push!(attempt_log,row)
                provenance["boundary_attempts"]=attempt_log
                atomic_toml(metadata_path,provenance)
                @show a
                flush(stdout)
            end)
        prepared.selected_seed!==nothing && (provenance["selected_seed"]=prepared.selected_seed)
        serialize(joinpath(output,"prepared.jls"),prepared)
        for (name,boundary) in pairs(prepared.boundaries)
            serialize(joinpath(output,"boundaries","$name.jls"),boundary)
            @printf("BOUNDARY %s chi=%s lambda=%.12g%+.3gi galerkin=%.3g density=%.12g\n",
                name,string(chi),real(boundary.lambda),imag(boundary.lambda),
                boundary.galerkin,real(boundary_number(boundary)))
            flush(stdout)
        end
        open(joinpath(output,"depth_scan.csv"),"w") do io
            println(io,"depth,S2A,S2B,S2C,S3,stilde,endpoint_residual,max_phase_error,max_cancellation")
            callback=s->begin
                e=s.entropies
                phase=max(maximum(abs.(e.phase2.-1)),abs(e.phase4-1))
                cancellation=maximum(z.cancellation_condition for z in values(s.sectors))
                println(io,join((s.depth,e.S2...,e.S3,e.stilde,s.endpoint_residual,phase,cancellation),','))
                flush(io)
                serialize(joinpath(output,"depth_$(s.depth).jls"),s)
                @printf("MEASUREMENT depth=%d stilde=%.12g residual=%.3g phase=%.3g\n",
                    s.depth,e.stilde,s.endpoint_residual,phase)
                flush(stdout)
            end
            result=measure_lmps_stilde(prepared.seams,prepared.caps;
                depths,tolerance,accept_unconverged,on_sample=callback)
            run=(;result,prepared...)
            serialize(joinpath(output,"result.jls"),run)
            provenance["status"]=run.result.converged ? "converged_lmps_length_limit" : "unconverged_lmps_length_limit"
            provenance["stilde"]=run.result.stilde
            provenance["diagnostics"]=Dict(string(k)=>v for (k,v) in pairs(run.result.diagnostics))
        end
    catch err
        provenance["status"]="failed"
        provenance["error"]=sprint(showerror,err)
        rethrow()
    finally
        provenance["finished_utc"]=string(now(UTC))
        atomic_toml(metadata_path,provenance)
    end
end

main(ARGS)
