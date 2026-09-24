# One resumable production task for a single beta and a list of bond dimensions.
#
# Every (beta,chi) owns its directory, so Slurm array tasks never append to a
# shared CSV.  The one-copy boundary checkpoint is written before the six-R
# stage and can be reused if a timeout or preemption happens later.

using LMPSVUMPS
using Dates
using Printf
using Random
using Serialization

const FORMAT_VERSION = 1

beta = parse(Float64, ENV["VUMPS_BATCH_BETA"])
chis = parse.(Int, split(ENV["VUMPS_BATCH_CHIS"], ':'))
group = get(ENV, "VUMPS_BATCH_GROUP", "unspecified")
root = get(ENV, "VUMPS_BATCH_ROOT", joinpath(
    @__DIR__, "..", "..", "data", "production_v1",
))
force = get(ENV, "VUMPS_BATCH_FORCE", "0") == "1"
accept_bridge = get(ENV, "VUMPS_BATCH_ACCEPT_BRIDGE", "0") == "1"

vumps_tol = parse(Float64, get(ENV, "VUMPS_BATCH_VUMPS_TOL", "1e-9"))
diagnostic_tol = parse(Float64, get(ENV, "VUMPS_BATCH_DIAGNOSTIC_TOL", "1e-7"))
max_iterations = parse(Int, get(ENV, "VUMPS_BATCH_MAXITER", "2400"))
spectrum_tol = parse(Float64, get(ENV, "VUMPS_BATCH_SPECTRUM_TOL", "1e-11"))
bridge_tol = parse(Float64, get(ENV, "VUMPS_BATCH_BRIDGE_TOL", "1e-6"))
mixed_tol = parse(Float64, get(ENV, "VUMPS_BATCH_MIXED_TOL", "1e-12"))
mixed_nmax = parse(Int, get(ENV, "VUMPS_BATCH_MIXED_NMAX", "30000"))
endpoint_tol = parse(Float64, get(ENV, "VUMPS_BATCH_ENDPOINT_TOL", "1e-10"))
endpoint_nmax = parse(Int, get(ENV, "VUMPS_BATCH_ENDPOINT_NMAX", "3000"))

beta_tag(x) = replace(@sprintf("%.14f", x), "." => "p", "-" => "m")
timestamp() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")

function csv_cell(x::AbstractString)
    occursin(r"[,\"\n\r]", x) || return x
    "\"" * replace(x, "\"" => "\"\"") * "\""
end
csv_cell(x::Bool) = x ? "1" : "0"
csv_cell(x::Integer) = string(x)
csv_cell(x::Real) = @sprintf("%.17g", Float64(x))
csv_cell(x) = csv_cell(string(x))

function atomic_csv(path, header, values)
    length(header) == length(values) || error("CSV header/value length mismatch")
    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    open(tmp, "w") do io
        println(io, join(csv_cell.(header), ','))
        println(io, join(csv_cell.(values), ','))
        flush(io)
    end
    mv(tmp, path; force=true)
end

function atomic_checkpoint(path, payload)
    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    open(tmp, "w") do io
        serialize(io, payload)
        flush(io)
    end
    mv(tmp, path; force=true)
end

function write_failure(path, err, bt)
    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    open(tmp, "w") do io
        println(io, "time=", timestamp())
        showerror(io, err, bt)
        println(io)
    end
    mv(tmp, path; force=true)
end

function save_spectrum(path, spectrum)
    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    open(tmp, "w") do io
        println(io, "rank,real,imag,abs")
        for (rank, value) in enumerate(spectrum)
            @printf(io, "%d,%.17g,%.17g,%.17g\n",
                    rank, real(value), imag(value), abs(value))
        end
    end
    mv(tmp, path; force=true)
end

function save_schmidt(path, singular_values, probabilities)
    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    open(tmp, "w") do io
        println(io, "rank,schmidt_value,probability")
        for rank in eachindex(probabilities)
            @printf(io, "%d,%.17g,%.17g\n",
                    rank, singular_values[rank], probabilities[rank])
        end
    end
    mv(tmp, path; force=true)
end

function load_checkpoint(path, expected_beta, expected_chi)
    payload = open(deserialize, path)
    payload.format_version == FORMAT_VERSION || error(
        "checkpoint format $(payload.format_version) is not $FORMAT_VERSION"
    )
    isapprox(payload.beta, expected_beta; atol=1e-14, rtol=0) ||
        error("checkpoint beta mismatch")
    payload.chi == expected_chi || error("checkpoint chi mismatch")
    payload.boundary
end

for chi in chis
    point_started = time()
    tag = beta_tag(beta)
    point = joinpath(root, "points", "beta_$(tag)", @sprintf("chi_%02d", chi))
    checkpoint_path = joinpath(root, "boundaries", "beta_$(tag)",
                               @sprintf("chi_%02d.jls", chi))
    observables_path = joinpath(point, "observables.csv")
    spectrum_path = joinpath(point, "transfer_spectrum.csv")
    schmidt_path = joinpath(point, "schmidt_spectrum.csv")
    sixr_path = joinpath(point, "sixr.csv")

    if !force && isfile(observables_path) && isfile(sixr_path) &&
       isfile(checkpoint_path)
        @printf("skip complete beta=%.14f chi=%d\n", beta, chi)
        flush(stdout)
        continue
    end

    Random.seed!(0x620000 + 1000chi + round(Int, 1_000_000beta))
    local boundary
    boundary_sec = 0.0
    checkpoint_loaded = false
    try
        if !force && isfile(checkpoint_path)
            boundary = load_checkpoint(checkpoint_path, beta, chi)
            checkpoint_loaded = true
        else
            boundary_sec = @elapsed boundary = solve_vumps_boundary(
                rk_ising(beta), chi;
                backend=:cpu,
                tolerance=vumps_tol,
                diagnostic_tolerance=diagnostic_tol,
                max_iterations=max_iterations,
                verbosity=0,
            )
            atomic_checkpoint(checkpoint_path, (
                format_version=FORMAT_VERSION,
                created_utc=timestamp(),
                beta=beta,
                chi=chi,
                julia_version=string(VERSION),
                boundary=boundary,
            ))
        end
    catch err
        write_failure(joinpath(point, "boundary_error.txt"), err, catch_backtrace())
        @error "boundary stage failed" beta chi exception=(err, catch_backtrace())
        continue
    end

    local obs
    observable_sec = 0.0
    try
        observable_sec = @elapsed obs = vumps_boundary_observables(
            boundary, beta; spectrum_num=min(16, chi^2), spectrum_tol=spectrum_tol,
        )
        save_spectrum(spectrum_path, obs.transfer_spectrum)
        save_schmidt(schmidt_path, obs.schmidt_singular_values,
                     obs.schmidt_probabilities)
        bd = boundary.diagnostics
        header = [
            "format_version", "created_utc", "group", "beta", "chi",
            "xi_boundary", "xi_spin_exact", "lambda1_abs", "lambda2_abs",
            "lambda2_over_lambda1", "entropy_vn", "entropy_renyi2",
            "entropy_renyi3", "entropy_renyi4", "entropy_infinity",
            "schmidt_gap", "effective_rank", "magnetization",
            "magnetization_abs", "magnetization_exact", "magnetization_abs_error",
            "row_logz", "free_energy", "free_energy_exact",
            "free_energy_abs_error", "free_energy_rel_error",
            "vumps_converged", "galerkin", "vumps_iterations",
            "left_environment_residual", "right_environment_residual",
            "center_residual", "checkpoint_loaded", "boundary_sec",
            "observable_sec", "point_elapsed_sec", "checkpoint",
        ]
        values = Any[
            FORMAT_VERSION, timestamp(), group, beta, chi,
            obs.xi, obs.exact_spin_xi, obs.lambda1_abs, obs.lambda2_abs,
            obs.lambda2_over_lambda1, obs.entropy_vn, obs.entropy_renyi2,
            obs.entropy_renyi3, obs.entropy_renyi4, obs.entropy_infinity,
            obs.schmidt_gap, obs.effective_rank, obs.magnetization,
            obs.magnetization_abs, obs.exact_magnetization,
            obs.magnetization_abs_error, obs.row_logz, obs.free_energy,
            obs.exact_free_energy, obs.free_energy_abs_error,
            obs.free_energy_rel_error, bd.converged, bd.galerkin_residual,
            bd.iterations, bd.left_environment_residual,
            bd.right_environment_residual, bd.center_residual,
            checkpoint_loaded, boundary_sec, observable_sec,
            time() - point_started, relpath(checkpoint_path, root),
        ]
        atomic_csv(observables_path, header, values)
        rm(joinpath(point, "boundary_error.txt"); force=true)
        @printf("boundary beta=%.14f chi=%d xi=%.9g m=%+.8g f_err=%.3e time=%.2fs loaded=%d\n",
                beta, chi, obs.xi, obs.magnetization,
                obs.free_energy_abs_error, boundary_sec, checkpoint_loaded)
        flush(stdout)
    catch err
        write_failure(joinpath(point, "observable_error.txt"), err, catch_backtrace())
        @error "observable stage failed" beta chi exception=(err, catch_backtrace())
        continue
    end

    if !force && isfile(sixr_path)
        @printf("skip existing sixR beta=%.14f chi=%d\n", beta, chi)
        flush(stdout)
        continue
    end

    try
        local objects, rails, result
        bridge_sec = @elapsed begin
            objects = vumps_projector_objects(
                boundary; require_converged=!accept_bridge,
                tolerance=bridge_tol, mixed_tolerance=mixed_tol,
                mixed_nmax=mixed_nmax,
            )
            rails = LMPSVUMPS.m2_oriented_rails(objects.H, objects.L, objects.R)
        end
        sixr_sec = @elapsed result = m2_contract_shared_six_R(
            rails, boundary.source.D;
            nmax=endpoint_nmax, tol=endpoint_tol, verbose=false,
        )
        d = objects.diagnostics
        header = [
            "format_version", "created_utc", "group", "beta", "chi",
            "tildeS", "S2A", "S2B", "S2C", "S3",
            "S2A_raw", "S2B_raw", "S2C_raw", "S3_raw",
            "kappaA_A", "kappaA_B", "kappaY", "mirror",
            "endpoint_residual", "map_residual", "scale_gate",
            "mixed_left_residual", "mixed_right_residual",
            "local_fixedpoint_residual", "projector_p2_residual",
            "projector_left_residual", "projector_right_residual",
            "metric_condition", "metric_dropped", "mixed_left_iterations",
            "mixed_right_iterations", "bridge_audit_ok",
            "accepted_incompatible", "quality_class", "effective_metric_rank",
            "bridge_sec", "sixr_sec",
            "point_elapsed_sec",
        ]
        values = Any[
            FORMAT_VERSION, timestamp(), group, beta, chi,
            result.tildeS, result.S2A, result.S2B, result.S2C, result.S3,
            result.S2A_raw, result.S2B_raw, result.S2C_raw, result.S3_raw,
            result.kappaA_A, result.kappaA_B, result.kappaY, result.mirror,
            result.endpoint_residual, result.map_residual, result.scale_gate,
            d.mixed_left_residual, d.mixed_right_residual,
            d.local_fixedpoint_residual, d.projector_p2_residual,
            d.projector_left_residual, d.projector_right_residual,
            d.metric_condition, d.metric_dropped, d.mixed_left_iterations,
            d.mixed_right_iterations, d.compatible,
            accept_bridge && !d.compatible,
            d.compatible ? "strict" :
                d.metric_dropped > 0 ? "accepted_low_rank" : "accepted_tolerance",
            chi - d.metric_dropped, bridge_sec, sixr_sec,
            time() - point_started,
        ]
        atomic_csv(sixr_path, header, values)
        rm(joinpath(point, "sixr_error.txt"); force=true)
        @printf("sixR beta=%.14f chi=%d Stilde=%+.12e bridge=%.2fs closure=%.2fs\n",
                beta, chi, result.tildeS, bridge_sec, sixr_sec)
        flush(stdout)
    catch err
        write_failure(joinpath(point, "sixr_error.txt"), err, catch_backtrace())
        @error "six-R stage failed; boundary checkpoint is reusable" beta chi exception=(err, catch_backtrace())
    end
end
