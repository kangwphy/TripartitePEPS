# Fixed-final-PEPS CTM and north VUMPS measurements. No entropy or six-R closure.
# Run scientific work only within a Slurm allocation.
using LMPSVUMPS, MPSKit, PEPSKit, TensorKit
using Serialization, SHA, Printf, LinearAlgebra, Dates

# Same Julia 1.12 contraction workaround as the canonical CPU GS driver.
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

file_sha256(path) = bytes2hex(open(SHA.sha256, path))
field(object, name::Symbol, default=nothing) = hasproperty(object, name) ? getproperty(object, name) : default

function atomic_write(writer, path)
    mkpath(dirname(path))
    temporary = path * ".tmp." * get(ENV, "SLURM_JOB_ID", string(getpid()))
    try
        open(writer, temporary, "w")
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
end

function json_string(value)
    text = replace(string(value), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n",
        '\r' => "\\r", '\t' => "\\t", '\b' => "\\b", '\f' => "\\f")
    return "\"" * text * "\""
end
json_value(value::Union{AbstractString,Symbol}) = json_string(value)
json_value(value::Bool) = value ? "true" : "false"
json_value(::Nothing) = "null"
json_value(value::Real) = isfinite(value) ? string(value) : "null"
json_value(value::NamedTuple) = "{" * join((json_string(k) * ":" * json_value(v) for (k, v) in pairs(value)), ',') * "}"
json_value(value::AbstractVector) = "[" * join(json_value.(value), ',') * "]"
function save_json(path, row)
    atomic_write(path) do io
        println(io, json_value(row))
    end
end
function csv_cell(value)
    text = string(value)
    return occursin(r"[,\"\r\n]", text) ? "\"" * replace(text, "\"" => "\"\"") * "\"" : text
end
function save_row(path, row)
    atomic_write(path) do io
        println(io, join(string.(keys(row)), ','))
        println(io, join(csv_cell.(values(row)), ','))
    end
end

# Only the controller's terminal Boolean and exact hash are needed. Requiring a
# unique key prevents an ambiguous/nested JSON hash from being silently used.
function marker_hash(point)
    marker = joinpath(point, "point_result.json")
    isfile(marker) || error("Final point_result.json missing")
    text = read(marker, String)
    terminals = collect(eachmatch(r"\"terminal\"\s*:\s*(true|false)", text))
    length(terminals) == 1 && terminals[1].captures[1] == "true" || error("Point is not terminal")
    hashes = collect(eachmatch(r"\"checkpoint_sha256\"\s*:\s*\"([0-9a-fA-F]{64})\"", text))
    length(hashes) == 1 || error("Terminal marker must contain one checkpoint SHA256")
    return lowercase(hashes[1].captures[1])
end

function same_peps(first, second)
    size(first) == size(second) || error("PEPS unit cells differ")
    maximum(norm(first[I] - second[I]) / max(norm(first[I]), eps(Float64))
        for I in CartesianIndices(size(first)))
end
function compare_tensors(first, second, label)
    length(first) == length(second) || error("Saved $label tensor count differs")
    error_max = 0.0
    for i in eachindex(first)
        space(first[i]) == space(second[i]) || error("Saved $label[$i] spaces differ")
        difference = norm(first[i] - second[i]) / max(norm(first[i]), eps(Float64))
        isfinite(difference) && difference <= 1e-14 || error("Saved $label[$i] tensor differs")
        error_max = max(error_max, difference)
    end
    return error_max
end
function require_cpu_tensors(tensors, label)
    all(TensorKit.storagetype(typeof(t)) <: Array for t in tensors) || error("$label contains non-CPU tensor storage")
end
function boundary_shape(boundary, chi, D)
    boundary.side == :north && boundary.backend == :cpu || error("Expected north CPU boundary")
    length(boundary.transfer) == 1 || error("Only one-site boundaries are supported")
    for name in (:AL, :AR, :AC, :C)
        require_cpu_tensors(getproperty(boundary.state, name), "MPS.$name")
    end
    for name in (:GLs, :GRs)
        require_cpu_tensors(getproperty(boundary.environments, name), "VUMPS.$name")
    end
    require_cpu_tensors(boundary.transfer[1], "VUMPS transfer")
    require_cpu_tensors((boundary.source[I] for I in CartesianIndices(size(boundary.source))), "VUMPS source PEPS")
    require_cpu_tensors((boundary.rotated_source[I] for I in CartesianIndices(size(boundary.rotated_source))), "VUMPS rotated PEPS")
    M = LMPSVUMPS._quantum_boundary_tensor(boundary)
    size(M) == (chi, D^2, chi) || error("Actual boundary shape $(size(M)) differs from ($chi,$(D^2),$chi)")
    return size(M)
end
function verify_boundary_roundtrip(path, boundary, expected, chi, D)
    hash_before = file_sha256(path)
    restored = open(deserialize, path)
    restored.source_sha256 == expected || error("Saved boundary final-GS SHA differs")
    saved = restored.selected_boundary
    boundary_shape(saved, chi, D)
    peps_error = same_peps(boundary.source, saved.source)
    peps_error <= 1e-14 || error("Saved boundary source PEPS differs")
    differences = Float64[]
    for name in (:AL, :AR, :AC, :C)
        push!(differences, compare_tensors(getproperty(boundary.state, name), getproperty(saved.state, name), "MPS.$name"))
    end
    for name in (:GLs, :GRs)
        push!(differences, compare_tensors(getproperty(boundary.environments, name), getproperty(saved.environments, name), "VUMPS.$name"))
    end
    push!(differences, compare_tensors(boundary.transfer[1], saved.transfer[1], "VUMPS transfer"))
    file_sha256(path) == hash_before || error("Boundary file changed during reload audit")
    return (; hash=hash_before, maximum_tensor_relative_error=maximum(differences), source_peps_relative_error=peps_error)
end

function measure_point(options)
    isempty(get(ENV, "SLURM_JOB_ID", "")) && error("Use a Slurm compute-node allocation; login-node measurements are forbidden")
    BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))
    point = realpath(options["point-dir"])
    source = joinpath(point, "warmup_state.jls")
    expected = marker_hash(point)
    haskey(options, "source-sha") && lowercase(options["source-sha"]) != expected && error("CLI and terminal marker hashes differ")
    file_sha256(source) == expected || error("Final PEPS checkpoint hash differs from terminal marker")
    payload = open(deserialize, source)
    gs = field(payload, :selected_groundstate)
    gs === nothing && error("Final checkpoint has no selected_groundstate")
    config = payload.config
    D, h, branch = Int(config.D), Float64(config.h), string(config.branch)
    D == 3 && gs.D == D && gs.environment_chi == 48 && config.ctm_chi == 48 || error("Expected D3, optimization CTM chi48")
    size(gs.peps) == (1, 1) || error("Expected a one-site final PEPS")
    dim(PEPSKit.north_virtualspace(gs.peps, 1, 1)) == D || error("Actual PEPS D differs from metadata")
    isfinite(h) && occursin(r"^[A-Za-z0-9_-]+$", branch) || error("Invalid checkpoint h/branch")
    haskey(options, "expected-h") && abs(parse(Float64, options["expected-h"]) - h) > 1e-10 && error("Expected h differs from final checkpoint")
    haskey(options, "expected-branch") && options["expected-branch"] != branch && error("Expected branch differs from final checkpoint")
    chi = parse(Int, get(options, "chi", "48"))
    chi == 48 || error("This campaign measures chi48 only")
    tolerance, diagnostic_tolerance = 1e-9, 1e-8
    seed = parse(Int, get(options, "seed", string(2026100600 + round(Int, 1e8h) + (branch == "ordered" ? 100_000 : 200_000))))
    max_vumps = parse(Int, get(options, "vumps-maxiter", "1000"))
    max_ctm = parse(Int, get(options, "ctm-maxiter", "2400"))
    max_vumps > 0 && max_ctm > 0 || error("Iteration limits must be positive")
    output = joinpath(point, "measurement")
    mkpath(output)
    result_path = joinpath(output, "result.json")
    script_hash = file_sha256(@__FILE__)
    code_revision = get(ENV, "TFIM_CODE_REVISION", "1e7cfd3")
    context = (; D, h, branch, point_dir=point, source_state=source, source_sha256=expected,
        source_final_checkpoint=source, source_final_checkpoint_sha256=expected,
        optimization_input_sha256=string(field(config, :source_sha256, "")),
        gs_converged=gs.converged, gs_stopping_reason=string(field(payload, :stopping_reason, "unknown")),
        chi, ctm_chi=chi, boundary_chi=chi, requested_measurement_tolerance=tolerance,
        script_sha256=script_hash, code_revision, slurm_job_id=ENV["SLURM_JOB_ID"])
    started = time()
    save_json(result_path, (; context..., status="running", measurement_converged=false))
    try
        # The optimizer saves an independently refreshed CPU QR environment at
        # tol1e-10. Reuse it when its explicit diagnostics meet this request.
        final_ctm_error = Float64(field(payload, :ctm_convergence_error, Inf))
        final_ctm_tol = Float64(field(payload, :ctm_tolerance, Inf))
        retain = field(payload, :ctm_converged, false) && final_ctm_tol <= tolerance &&
            isfinite(final_ctm_error) && final_ctm_error <= tolerance &&
            string(field(payload, :ctm_projector, "")) == "C4vQRProjector" &&
            string(field(payload, :ctm_measurement_backend, "")) == "cpu"
        environment, ctm_converged, ctm_error, ctm_tol, ctm_method = if retain
            (gs.environment, true, final_ctm_error, final_ctm_tol, "retained_final_CPU_QR_environment")
        else
            env, info = PEPSKit.leading_boundary(deepcopy(gs.environment), gs.peps;
                alg=:C4vCTMRG, projector_alg=:C4vQRProjector, tol=tolerance,
                maxiter=max_ctm, miniter=1, verbosity=-1)
            (env, info.converged, Float64(info.convergence_error), tolerance, "CPU_QR_refinement_same_final_PEPS")
        end
        require_cpu_tensors(environment.corners, "CTM corners")
        require_cpu_tensors(environment.edges, "CTM edges")
        ctm_checkpoint = joinpath(output, "ctm_state.jls")
        ctm_diagnostics = (; converged=ctm_converged, convergence_error=ctm_error,
            tolerance=ctm_tol, requested_tolerance=tolerance, projector="C4vQRProjector", backend="cpu", method=ctm_method)
        atomic_write(ctm_checkpoint) do io
            serialize(io, (; format_version=1, context..., peps=gs.peps, environment, diagnostics=ctm_diagnostics))
        end
        reloaded_ctm = open(deserialize, ctm_checkpoint)
        reloaded_ctm.source_sha256 == expected && same_peps(gs.peps, reloaded_ctm.peps) <= 1e-14 || error("Saved CTM provenance or PEPS differs")
        compare_tensors(environment.corners, reloaded_ctm.environment.corners, "CTM corners")
        compare_tensors(environment.edges, reloaded_ctm.environment.edges, "CTM edges")
        ctm_converged && isfinite(ctm_error) && ctm_error <= tolerance || error("Fixed-PEPS CTM did not converge")
        xih, xiv, _, _ = MPSKit.correlation_length(gs.peps, environment; num_vals=3, tol=1e-11)
        xi_ctm_h, xi_ctm_v = Float64(first(xih)), Float64(first(xiv))
        cobs = tfim_peps_observables(gs.peps, gs.hamiltonian, environment)
        final_energy_difference = cobs.energy - gs.energy
        energy_check_tolerance = retain ? 1e-10 : 1e-8
        isfinite(final_energy_difference) && abs(final_energy_difference) <= energy_check_tolerance || error("CTM energy differs from final GS beyond declared tolerance")
        ctm_row = (; context..., xi_CTM_horizontal=xi_ctm_h, xi_CTM_vertical=xi_ctm_v,
            x_CTM=cobs.x, z_CTM=cobs.z, abs_z_CTM=abs(cobs.z), energy_CTM=cobs.energy,
            ctm_converged, ctm_convergence_error=ctm_error, ctm_tolerance=ctm_tol,
            ctm_environment_method=ctm_method, final_checkpoint_energy=gs.energy,
            energy_difference_from_final_checkpoint=final_energy_difference, energy_check_tolerance,
            ctm_checkpoint, ctm_checkpoint_sha256=file_sha256(ctm_checkpoint))
        save_row(joinpath(output, "ctm.csv"), ctm_row)

        boundary_checkpoint = joinpath(output, "boundary_state.jls")
        recovered = false
        boundary = if isfile(boundary_checkpoint)
            previous = open(deserialize, boundary_checkpoint)
            previous.source_sha256 == expected || error("Existing boundary belongs to another final PEPS")
            previous.D == D && previous.h == h && previous.branch == branch && previous.boundary_chi == chi || error("Existing boundary labels differ")
            previous.selected_boundary.requested_tolerance <= tolerance || error("Existing boundary has insufficient requested tolerance")
            recovered = true
            previous.selected_boundary
        else
            solve_quantum_vumps_boundary(gs.peps, chi; side=:north, backend=:cpu, seed,
                tolerance, diagnostic_tolerance, max_iterations=max_vumps, verbosity=1,
                accept_unconverged=true)
        end
        shape = boundary_shape(boundary, chi, D)
        peps_error = same_peps(gs.peps, boundary.source)
        isfinite(peps_error) && peps_error <= 1e-12 || error("VUMPS boundary source PEPS differs from final GS")
        accepted = boundary.converged && boundary.galerkin_residual <= tolerance &&
            boundary.left_environment_residual <= diagnostic_tolerance &&
            boundary.right_environment_residual <= diagnostic_tolerance &&
            boundary.center_residual <= diagnostic_tolerance
        save_path = accepted ? boundary_checkpoint : joinpath(output, "boundary_unconverged.jls")
        atomic_write(save_path) do io
            serialize(io, (; format_version=1, context..., selected_boundary=boundary,
                boundary_side="north", physical_dimension=shape[2], seed,
                diagnostic_tolerance, recovered_boundary=recovered, saved_at=string(Dates.now())))
        end
        roundtrip = verify_boundary_roundtrip(save_path, boundary, expected, chi, D)
        diagnostics = (; context..., vumps_converged=accepted, vumps_reported_converged=boundary.converged,
            vumps_galerkin=boundary.galerkin_residual, vumps_left_residual=boundary.left_environment_residual,
            vumps_right_residual=boundary.right_environment_residual, vumps_center_residual=boundary.center_residual,
            vumps_iterations=boundary.iterations, vumps_tolerance=boundary.requested_tolerance,
            vumps_diagnostic_tolerance=diagnostic_tolerance, boundary_checkpoint=save_path,
            boundary_sha256=roundtrip.hash, boundary_roundtrip_verified=true,
            boundary_roundtrip_scope="all_AL_AR_AC_C_GLs_GRs_transfer_and_source_PEPS",
            boundary_tensor_roundtrip_relative_error=roundtrip.maximum_tensor_relative_error,
            actual_boundary_left_chi=shape[1], actual_boundary_right_chi=shape[3], actual_boundary_physical_dimension=shape[2],
            boundary_backend="cpu", boundary_side="north", peps_match_relative=peps_error, seed,
            recovered_boundary=recovered)
        save_row(joinpath(output, "vumps_diagnostics.csv"), diagnostics)
        accepted || error("VUMPS diagnostics failed; full unaccepted boundary retained at $save_path")
        xi_M = Float64(first(MPSKit.correlation_length(boundary.state; num_vals=3, tol=1e-11)))
        vobs = tfim_one_site_observables(boundary)
        all(isfinite, (xi_M, xi_ctm_h, xi_ctm_v, vobs.x, vobs.z, cobs.x, cobs.z, cobs.energy)) || error("Nonfinite measurement")
        all(>(0), (xi_M, xi_ctm_h, xi_ctm_v)) || error("Nonpositive correlation length")
        file_sha256(source) == expected || error("Final GS checkpoint changed during measurement")
        vumps_row = (; diagnostics..., xi_M, x_VUMPS=vobs.x, z_VUMPS=vobs.z, abs_z_VUMPS=abs(vobs.z))
        save_row(joinpath(output, "vumps.csv"), vumps_row)
        summary = (; vumps_row..., xi_CTM_horizontal=xi_ctm_h, xi_CTM_vertical=xi_ctm_v,
            x_CTM=cobs.x, z_CTM=cobs.z, abs_z_CTM=abs(cobs.z), energy_CTM=cobs.energy,
            ctm_converged, ctm_convergence_error=ctm_error, ctm_tolerance=ctm_tol,
            ctm_environment_method=ctm_method, ctm_checkpoint,
            ctm_checkpoint_sha256=file_sha256(ctm_checkpoint), final_checkpoint_energy=gs.energy,
            energy_difference_from_final_checkpoint=final_energy_difference, energy_check_tolerance,
            seconds=time() - started, runtime_version=string(VERSION), project_path=Base.active_project(),
            saved_at=string(Dates.now()))
        summary_path = joinpath(output, "summary.csv")
        save_row(summary_path, summary)
        save_json(result_path, (; context..., status="complete", measurement_converged=true,
            ctm_converged=true, vumps_converged=true, summary_csv=summary_path,
            summary_sha256=file_sha256(summary_path), boundary_checkpoint,
            boundary_sha256=file_sha256(boundary_checkpoint), ctm_checkpoint,
            ctm_checkpoint_sha256=file_sha256(ctm_checkpoint), boundary_roundtrip_verified=true,
            boundary_roundtrip_scope=diagnostics.boundary_roundtrip_scope, seconds=time() - started))
        println("MEASUREMENT_COMPLETE h=$h branch=$branch source_sha256=$expected xi_M=$xi_M xi_CTM=$xi_ctm_h gs_converged=$(gs.converged)")
    catch exception
        save_json(result_path, (; context..., status="failed", measurement_converged=false,
            error=sprint(showerror, exception), seconds=time() - started))
        rethrow()
    end
end

function parse_options(arguments)
    options = Dict{String,String}()
    allowed = Set(["point-dir", "source-sha", "expected-h", "expected-branch", "chi", "seed", "vumps-maxiter", "ctm-maxiter"])
    iseven(length(arguments)) || error("Options require --name value pairs")
    for i in 1:2:length(arguments)
        startswith(arguments[i], "--") || error("Expected --option")
        name = arguments[i][3:end]
        name in allowed && !haskey(options, name) || error("Unknown or repeated --$name")
        options[name] = arguments[i + 1]
    end
    haskey(options, "point-dir") || error("Usage: measure_bidirectional_point.jl --point-dir POINT [--source-sha FINAL_SHA]")
    return options
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    measure_point(parse_options(ARGS))
end
