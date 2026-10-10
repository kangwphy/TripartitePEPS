#!/usr/bin/env julia
# Environment-only recovery for the two failed low-endpoint supplementary points.
# Required: TFIM_REPAIR_POINT, TFIM_REPAIR_EXPECTED_H,
#           TFIM_REPAIR_EXPECTED_CHECKPOINT_SHA256, real SLURM_JOB_ID.
# Optional: TFIM_REPAIR_MAXITER (1..10000; default 10000).
# Run with the ORIGINAL frozen package's --project and Julia 1.12.6.
# Exit 0 means the environment-only replacement and audit both succeeded.
# No optimization, PEPS normalization, cold start, or Eigh is performed.
using Dates, LinearAlgebra, Serialization, SHA, TOML, Sockets, Printf

isempty(get(ENV, "SLURM_JOB_ID", "")) && error("Repair requires a real Slurm compute-node allocation")
Sys.islinux() || error("This repair uses Linux advisory locking and atomic POSIX rename")
using LMPSVUMPS, PEPSKit, MPSKit, TensorKit

if VERSION >= v"1.12"
    pepskit_source = dirname(pathof(PEPSKit))
    for relative in (
        "algorithms/contractions/ctmrg/enlarge_corner.jl",
        "algorithms/contractions/ctmrg/enlarge_corner_column.jl",
        "algorithms/contractions/ctmrg/projector.jl",
        "algorithms/contractions/ctmrg/halfinf_env.jl",
        "algorithms/contractions/ctmrg/fullinf_env.jl",
        "algorithms/contractions/ctmrg/renormalize_corner.jl",
        "algorithms/contractions/ctmrg/renormalize_edge.jl",
        "algorithms/contractions/ctmrg/contract_site.jl",
        "algorithms/contractions/ctmrg/gaugefix.jl",
    )
        Base.include(PEPSKit, joinpath(pepskit_source, relative))
    end
end

const EXPECTED_SOURCE_SHA = "c207e7ad2a8a21b5a9547541e063ae5b5ea92ed4dd9ed7c7f2603d3b743ac811"
const PHYSICS_SHA = "7f315606819f56cf47cb9805728f19167ca80c2dc45484619151100387101abd"
const BASE_COMMIT = "1e7cfd36531b670db8d3cc3b975cc417963e3d50"
const TOLERANCE = 1e-10
const JOB = ENV["SLURM_JOB_ID"]
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

sha_file(path) = bytes2hex(open(SHA.sha256, path))
function serialized_bytes(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end
object_sha(value) = bytes2hex(SHA.sha256(serialized_bytes(value)))

function sync_file(io)
    flush(io)
    ccall(:fsync, Cint, (Cint,), fd(io)) == 0 || error("fsync failed")
end
function sync_directory(path)
    handle = ccall(:open, Cint, (Cstring, Cint), path, 0)
    handle >= 0 || error("Cannot open directory for fsync: $path")
    try
        ccall(:fsync, Cint, (Cint,), handle) == 0 || error("Directory fsync failed: $path")
    finally
        ccall(:close, Cint, (Cint,), handle)
    end
end
function atomic_rename(source, target)
    # Julia mv(force=true) may unlink the destination first; do not use it here.
    ccall(:rename, Cint, (Cstring, Cstring), source, target) == 0 ||
        error("Atomic rename failed; errno=$(Base.Libc.errno())")
    sync_directory(dirname(target))
end
function save_object(path, value)
    open(path, "w") do io
        serialize(io, value)
        sync_file(io)
    end
end
function save_audit(path, audit)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        TOML.print(io, audit; sorted=true)
        sync_file(io)
    end
    atomic_rename(temporary, path)
end

function check_peps(peps)
    size(peps) == (1, 1) || error("Expected a one-site PEPS")
    tensor = peps.A[1, 1]
    dim(space(tensor, 1)) == 2 || error("Expected a spin-one-half PEPS")
    all(i -> dim(space(tensor, i)) == 3, 2:5) || error("Actual PEPS D is not 3")
    eltype(tensor.data) == ComplexF64 || error("Expected ComplexF64 PEPS")
    tensor.data isa Array || error("Expected CPU tensor storage")
    all(isfinite, tensor.data) && isfinite(norm(tensor)) && norm(tensor) > 0 ||
        error("Invalid PEPS tensor")
end
function check_environment(environment)
    environment isa PEPSKit.CTMRGEnv || error("Expected a CTMRG environment")
    all(t -> dim(space(t, 1)) == 48 && dim(space(t, 2)) == 48, environment.corners) ||
        error("Actual environment chi is not 48")
    for tensors in (environment.corners, environment.edges), tensor in tensors
        tensor.data isa Array && eltype(tensor.data) == ComplexF64 ||
            error("Expected CPU ComplexF64 CTM tensors")
        all(isfinite, tensor.data) || error("Nonfinite CTM tensor")
    end
end
function require_identical_peps(first, second)
    typeof(first) === typeof(second) && size(first) == size(second) || error("PEPS type/shape changed")
    for i in eachindex(first.A)
        a, b = first.A[i], second.A[i]
        typeof(a) === typeof(b) && space(a) == space(b) || error("PEPS tensor spaces changed")
        # Raw bytes also distinguish signed zero and preserve exact complex data.
        reinterpret(UInt8, vec(a.data)) == reinterpret(UInt8, vec(b.data)) ||
            error("PEPS tensor entries changed during environment repair")
    end
end
function preserved_field_hashes(checkpoint)
    Dict(string(k) => object_sha(getproperty(checkpoint, k))
         for k in keys(checkpoint) if k != :environment)
end

function expected_config(h)
    (; format_version=1, D=3, h, branch="from_low_endpoint", ctm_chi=48,
       source_sha256=EXPECTED_SOURCE_SHA, code_sha256=PHYSICS_SHA,
       base_git_commit=BASE_COMMIT, optimization_backend="cpu",
       ad_tolerance=1e-5, maximum_iterations=1000, ctm_tolerance=1e-9,
       gradient_tolerance=2e-7, base_seed=20261006, spatial_symmetry="rotatereflect",
       optimization_projector="C4vQRProjector",
       checkpoint_semantics="PEPS/environment restart; LBFGS memory restarts at segment boundary")
end

function validate_checkpoint(checkpoint, point, h)
    checkpoint isa NamedTuple || error("Expected the original NamedTuple checkpoint payload")
    config = expected_config(h)
    isequal(checkpoint.config, config) || error("Frozen optimizer configuration differs")
    checkpoint.config_hash == bytes2hex(SHA.sha256(codeunits(repr(config)))) ||
        error("Original config_hash differs from expected exact configuration")
    checkpoint.optimization_backend == "cpu" || error("Checkpoint backend differs")
    checkpoint.source_sha256 == EXPECTED_SOURCE_SHA && checkpoint.source_h == 3.038 ||
        error("Checkpoint does not originate from the frozen low endpoint")
    source = realpath(checkpoint.source_state)
    campaign = dirname(dirname(dirname(point)))
    source == realpath(joinpath(campaign, "seeds", "endpoint_h3.03800000.jls")) ||
        error("Source checkpoint path is not this campaign's fixed low endpoint")
    sha_file(source) == EXPECTED_SOURCE_SHA || error("Fixed source checkpoint changed")
    0 < checkpoint.completed_iterations < 1000 || error("Repair is only for a partial optimization")
    checkpoint.reason in ("accepted_step", "segment_endpoint", "initial_or_resume_validated",
                          "tight_audit_requires_refinement") || error("Unsupported checkpoint phase")
    checkpoint.strict_mode isa Bool || error("Invalid strict-mode checkpoint metadata")
    isfinite(checkpoint.optimizer_stop_tolerance) && 0 < checkpoint.optimizer_stop_tolerance <= 1e-5 ||
        error("Invalid saved optimizer tolerance")
    !hasproperty(checkpoint, :validated_endpoint) && !hasproperty(checkpoint, :selected_groundstate) ||
        error("Terminal GS payload cannot be environment-repaired as an iteration checkpoint")
    check_peps(checkpoint.peps)
    check_environment(checkpoint.environment)
    return source
end

function repair_locked(point, h, expected_old_sha, maxiter)
    for name in ("warmup_state.jls", "accepted.csv", "point_result.json")
        !ispath(joinpath(point, name)) || error("Refusing to repair a terminal point: $name")
    end
    checkpoint_path = joinpath(point, "iteration_checkpoint.jls")
    sha_file(checkpoint_path) == expected_old_sha || error("Iteration checkpoint changed before repair")
    checkpoint = open(deserialize, checkpoint_path)
    source = validate_checkpoint(checkpoint, point, h)
    preserved = preserved_field_hashes(checkpoint)
    recovery = joinpath(point, "recovery")
    mkpath(recovery)
    attempt = mktempdir(recovery; prefix="qr_environment_job$(JOB)_", cleanup=false)
    backup = joinpath(attempt, "original_iteration_checkpoint.jls")
    open(backup, "w") do output
        write(output, read(checkpoint_path))
        sync_file(output)
    end
    sha_file(backup) == expected_old_sha || error("Backup differs from the original checkpoint")
    sync_directory(attempt)
    peps, environment = deepcopy(checkpoint.peps), deepcopy(checkpoint.environment)
    environment_before_sha = object_sha(checkpoint.environment)
    audit_path = joinpath(attempt, "audit.toml")
    candidate_path = joinpath(attempt, "candidate_environment.jls")
    ctmrg_source = joinpath(dirname(pathof(PEPSKit)), "algorithms", "ctmrg", "ctmrg.jl")
    audit = Dict{String, Any}(
        "status" => "started", "h" => h, "D" => 3, "chi" => 48,
        "point" => point, "checkpoint" => checkpoint_path, "backup" => backup,
        "candidate_environment" => candidate_path, "old_checkpoint_sha256" => expected_old_sha,
        "source_state" => source, "source_sha256" => EXPECTED_SOURCE_SHA,
        "config_hash" => checkpoint.config_hash, "physics_code_sha256" => PHYSICS_SHA,
        "completed_iterations_preserved" => checkpoint.completed_iterations,
        "checkpoint_phase_preserved" => checkpoint.reason,
        "preserved_field_sha256" => preserved, "environment_before_sha256" => environment_before_sha,
        "peps_sha256" => object_sha(checkpoint.peps), "script_sha256" => sha_file(@__FILE__),
        "ctmrg_library_source" => ctmrg_source, "ctmrg_library_source_sha256" => sha_file(ctmrg_source),
        "slurm_job_id" => JOB, "hostname" => gethostname(), "runtime_version" => string(VERSION),
        "project" => Base.active_project(), "started_at" => string(Dates.now()),
        "method" => "continue_saved_C4v_QR_environment_same_exact_PEPS",
        "cold_start_used" => false, "eigh_used" => false, "optimizer_called" => false,
        "tolerance" => TOLERANCE, "maxiter" => maxiter, "miniter" => 1,
        "iterations" => "not exposed by frozen leading_boundary API", "converged" => false, "replaced" => false,
        "saved_energy_and_gradient" => "preserved verbatim; not recomputed by this repair")
    save_audit(audit_path, audit)
    println("REPAIR_AUDIT ", audit_path)
    flush(stdout)
    started = time()
    try
        # Keep the original frozen solver API and its convergence decision.
        # That API reports the residual and convergence flag, but no count.
        environment, info = PEPSKit.leading_boundary(environment, peps;
            alg=:C4vCTMRG, projector_alg=:C4vQRProjector,
            tol=TOLERANCE, maxiter=maxiter, miniter=1, verbosity=-1)
        error_value = Float64(info.convergence_error)
        isfinite(error_value) || error("Nonfinite QR environment residual")
        audit["convergence_error"] = error_value
        audit["converged"] = info.converged && error_value <= TOLERANCE
        audit["seconds"] = time() - started
        save_audit(audit_path, audit)
        println("REPAIR_QR maxiter=", maxiter, " error=", error_value,
                " converged=", audit["converged"])
        flush(stdout)
        require_identical_peps(checkpoint.peps, peps)
        check_environment(environment)
        preserved_field_hashes(checkpoint) == preserved || error("Original checkpoint fields were mutated")
        audit["environment_after_sha256"] = object_sha(environment)
        save_object(candidate_path, (; peps, environment, source_checkpoint_sha256=expected_old_sha,
            source_sha256=EXPECTED_SOURCE_SHA, h, completed_iterations=checkpoint.completed_iterations,
            converged=audit["converged"], iterations=audit["iterations"],
            convergence_error=audit["convergence_error"], tolerance=TOLERANCE))
        audit["candidate_environment_sha256"] = sha_file(candidate_path)
        audit["converged"] || error("QR continuation exhausted maxiter without meeting the unchanged tolerance")

        repaired = merge(checkpoint, (; environment))
        keys(repaired) == keys(checkpoint) || error("Checkpoint field layout changed")
        preserved_field_hashes(repaired) == preserved || error("Non-environment checkpoint fields changed")
        # Candidate is on the same filesystem as the original checkpoint.
        candidate_checkpoint = joinpath(attempt, "repaired_iteration_checkpoint.jls")
        save_object(candidate_checkpoint, repaired)
        restored = open(deserialize, candidate_checkpoint)
        require_identical_peps(checkpoint.peps, restored.peps)
        keys(restored) == keys(checkpoint) && preserved_field_hashes(restored) == preserved ||
            error("Serialization roundtrip changed non-environment checkpoint fields")
        check_environment(restored.environment)
        object_sha(restored.environment) == object_sha(environment) ||
            error("Serialization roundtrip changed the candidate environment")
        validate_checkpoint(restored, point, h)
        audit["roundtrip_verified"] = true
        audit["new_checkpoint_sha256"] = sha_file(candidate_checkpoint)
        audit["status"] = "validated_before_atomic_replace"
        audit["seconds"] = time() - started
        save_audit(audit_path, audit)

        # Detect any writer outside the advisory-lock protocol before commit.
        sha_file(checkpoint_path) == expected_old_sha || error("Original checkpoint changed while repairing")
        sha_file(source) == EXPECTED_SOURCE_SHA || error("Fixed endpoint changed while repairing")
        for name in ("warmup_state.jls", "accepted.csv", "point_result.json")
            !ispath(joinpath(point, name)) || error("Point became terminal during repair")
        end
        atomic_rename(candidate_checkpoint, checkpoint_path)
        sha_file(checkpoint_path) == audit["new_checkpoint_sha256"] || error("Committed checkpoint hash mismatch")
        audit["replaced"] = true
        audit["status"] = "complete"
        audit["finished_at"] = string(Dates.now())
        save_audit(audit_path, audit)
        println("REPAIR_COMPLETE old_sha=", expected_old_sha, " new_sha=", audit["new_checkpoint_sha256"])
        return 0
    catch err
        # Retain the returned candidate, or the input environment if the solver
        # threw before returning. Never replace the checkpoint in this path.
        try
            if !isfile(candidate_path)
                save_object(candidate_path, (; peps, environment,
                    source_checkpoint_sha256=expected_old_sha, h,
                    completed_iterations=checkpoint.completed_iterations,
                    converged=false, iterations=audit["iterations"]))
            end
            audit["candidate_environment_sha256"] = sha_file(candidate_path)
        catch candidate_error
            audit["candidate_save_error"] = sprint(showerror, candidate_error)
        end
        current_sha = isfile(checkpoint_path) ? sha_file(checkpoint_path) : "missing"
        audit["checkpoint_sha256_after_attempt"] = current_sha
        audit["original_checkpoint_unchanged"] = current_sha == expected_old_sha
        audit["replaced"] = haskey(audit, "new_checkpoint_sha256") && current_sha == audit["new_checkpoint_sha256"]
        audit["status"] = audit["replaced"] ? "post_replace_verification_failed" : "failed_original_preserved"
        audit["error"] = sprint(showerror, err)
        audit["finished_at"] = string(Dates.now())
        audit["seconds"] = time() - started
        save_audit(audit_path, audit)
        rethrow()
    end
end

function main()
    point = realpath(ENV["TFIM_REPAIR_POINT"])
    h = parse(Float64, ENV["TFIM_REPAIR_EXPECTED_H"])
    h in (3.0505, 3.051) || error("Repair is restricted to the two failed supplementary points")
    basename(point) == @sprintf("h%.8f", h) || error("Point directory and expected h differ")
    basename(dirname(point)) == "from_h3.03800000" || error("Expected low-endpoint group")
    basename(dirname(dirname(point))) == "D3" || error("Expected D3 campaign directory")
    basename(dirname(dirname(dirname(point)))) == "qr_fixed_endpoints_20261006" ||
        error("Expected the separate fixed-endpoint campaign")
    expected_old_sha = lowercase(ENV["TFIM_REPAIR_EXPECTED_CHECKPOINT_SHA256"])
    occursin(r"^[0-9a-f]{64}$", expected_old_sha) || error("Invalid expected checkpoint SHA")
    maxiter = parse(Int, get(ENV, "TFIM_REPAIR_MAXITER", "10000"))
    1 <= maxiter <= 10000 || error("Repair iteration cap must be within 1..10000")
    project = Base.active_project()
    isnothing(project) && error("Use the frozen package with --project")
    package = dirname(project)
    helper = joinpath(package, "production", "tfim", "run_bidirectional_chain.py")
    isfile(helper) || error("The active project is not the frozen TFIM physics package")
    readchomp(`python3 $helper --package $package --print-code-hash`) == PHYSICS_SHA ||
        error("Frozen physics package fingerprint changed")
    lock_path = joinpath(point, "additional_controller.lock")
    open(lock_path, "a+") do lock
        ccall(:flock, Cint, (Cint, Cint), fd(lock), 2 | 4) == 0 ||
            error("Point is locked by its supplementary optimizer/controller")
        try
            return repair_locked(point, h, expected_old_sha, maxiter)
        finally
            ccall(:flock, Cint, (Cint, Cint), fd(lock), 8)
        end
    end
end

try
    exit(main())
catch err
    println(stderr, "QR_ENVIRONMENT_REPAIR_FAILED: ", sprint(showerror, err))
    rethrow()
end
