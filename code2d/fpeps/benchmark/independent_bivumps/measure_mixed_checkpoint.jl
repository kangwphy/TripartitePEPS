# Physical diagnostics of an unaccepted copied iteration. Never export it
# as a converged environment or feed it into entropy acceptance.
include("export_snapshot_audit.jl")
module MixedCheckpointPhysical
include("observables.jl")
end

function mp_checkpoint(snapshot,template,checkpoint_audit,out)
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    check=TOML.parsefile(checkpoint_audit)
    check["complete"] && check["passed"] && check["diagnostic_only"] || error("checkpoint diagnostic incomplete")
    !check["accepted_curve_point"] && !check["accepted_entropy"] || error("expected unaccepted checkpoint")
    check["chi"]==16 && check["caps_dominant"] || error("wrong chi or non-dominant caps")
    hashes=copy(check["source_hashes"])
    for (p,h) in hashes
        bytes2hex(sha256(read(p)))==h || error("stale checkpoint source: $p")
    end
    pairpath=joinpath(snapshot,"independent_pair.jls")
    for p in vcat([pairpath,joinpath(snapshot,"progress.toml")],joinpath.(template,["boundary_1.jls","boundary_3.jls"]))
        get(hashes,p,"")==bytes2hex(sha256(read(p))) || error("audit does not cover input: $p")
    end
    for p in (checkpoint_audit,@__FILE__,joinpath(@__DIR__,"export_snapshot_audit.jl"),
              joinpath(@__DIR__,"observables.jl"),joinpath(@__DIR__,"../stability_observable_modes.jl"))
        hashes[p]=bytes2hex(sha256(read(p)))
    end
    mkpath(out)
    status=Dict{String,Any}("complete"=>false,"diagnostic_only"=>true,
        "accepted_entropy"=>false,"accepted_curve_point"=>false,
        "zero_optimization_steps"=>true,"chi"=>16,"iteration"=>check["iteration"],
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>ENV["SLURM_JOB_PARTITION"],
        "source_hashes"=>hashes)
    bv_write(joinpath(out,"status.toml"),status)
    try
        public=joinpath(out,"diagnostic_boundary")
        es_prepare_audit(pairpath,template,public)
        preparationpath=joinpath(public,"preparation.toml")
        preparation=TOML.parsefile(preparationpath)
        preparation["complete"] && preparation["seed_only"] || error("diagnostic export incomplete")
        after=preparation["initial_independent_audit"]["outer_residual"]
        before=check["original"]["outer_residual"]
        abs(after-before)/before<1e-6 || error("public roundtrip changed original residual")
        hashes[preparationpath]=bytes2hex(sha256(read(preparationpath)))
        for (p,h) in preparation["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("stale export source")
            hashes[p]=h
        end
        # Honest metadata for the existing diagnostic physical API. All
        # convergence/acceptance flags stay false, regardless of RDM quality.
        bv_write(joinpath(public,"report.toml"),Dict("schema"=>"independent_bivumps_v1",
            "complete"=>true,"converged"=>false,"diagnostic_only"=>true,
            "accepted_entropy"=>false,"accepted_curve_point"=>false,
            "source_kind"=>"copied_nonstationary_iteration",
            "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))),
            "final"=>preparation["initial_independent_audit"],"source_hashes"=>copy(hashes)))
        physical=joinpath(out,"physical")
        MixedCheckpointPhysical.measure_joint_observables(public,physical)
        measurement=TOML.parsefile(joinpath(physical,"correlations.toml"))
        measurement["complete"] && !measurement["cases"][1]["bivumps_converged"] || error("wrong physical flags")
        status["exported_residual"]=after
        status["physical_source"]=physical
        for p in (joinpath(public,"report.toml"),joinpath(physical,"correlations.toml"))
            hashes[p]=bytes2hex(sha256(read(p)))
        end
    catch err
        status["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        status["complete"]=true;bv_write(joinpath(out,"status.toml"),status)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    mp_checkpoint(ARGS...)
end
