using LMPSVUMPS
using Printf
using Random

csv_values(::Type{T}, name, fallback) where {T} =
    haskey(ENV, name) ? parse.(T, split(ENV[name], ',')) : collect(fallback)

backend = Symbol(lowercase(get(ENV, "VUMPS_BACKEND", "cpu")))
backend in (:cpu, :cuda) || error("VUMPS_BACKEND must be cpu or cuda")
betac = 0.5 * log(1 + sqrt(2))
betas = csv_values(Float64, "VUMPS_FE_BETAS", (0.30, 0.40, betac, 0.46))
chis = csv_values(Int, "VUMPS_FE_CHIS", (2, 4, 8, 16))
max_iterations = parse(Int, get(ENV, "VUMPS_FE_MAXITER", "500"))
tolerance = parse(Float64, get(ENV, "VUMPS_FE_TOL", "1e-10"))
diagnostic_tolerance = parse(
    Float64, get(ENV, "VUMPS_FE_DIAGNOSTIC_TOL", "1e-8")
)
output = get(
    ENV, "VUMPS_FREE_ENERGY_OUT",
    joinpath(@__DIR__, "../../data/free_energy/vumps_$(backend).csv"),
)
mkpath(dirname(output))

open(output, "w") do io
    println(io, join((
        "method", "backend", "beta", "chi", "converged", "iterations",
        "galerkin_residual", "left_environment_residual",
        "right_environment_residual", "center_residual", "logz", "free_energy",
        "exact_free_energy", "abs_error", "rel_error", "elapsed_seconds",
    ), ','))
    for beta in betas, chi in chis
        # Use a deterministic seed for every point.  CPU/GPU comparisons then
        # probe the algorithm/backend rather than unrelated random starts.
        Random.seed!(0x510000 + 1000chi + round(Int, 100beta))
        local boundary
        elapsed = @elapsed boundary = solve_vumps_boundary(
            rk_ising(beta), chi;
            backend,
            tolerance,
            diagnostic_tolerance,
            max_iterations,
            verbosity=0,
            accept_unconverged=true,
        )
        diagnostics = boundary.diagnostics
        logz = vumps_logz(boundary)
        free_energy = vumps_free_energy(boundary, beta)
        exact_free_energy = onsager_f(beta)
        absolute_error = abs(free_energy - exact_free_energy)
        relative_error = absolute_error / abs(exact_free_energy)
        println(io, join((
            "vumps", backend, @sprintf("%.17g", beta), chi,
            diagnostics.converged, diagnostics.iterations,
            @sprintf("%.17g", diagnostics.galerkin_residual),
            @sprintf("%.17g", diagnostics.left_environment_residual),
            @sprintf("%.17g", diagnostics.right_environment_residual),
            @sprintf("%.17g", diagnostics.center_residual),
            @sprintf("%.17g", logz), @sprintf("%.17g", free_energy),
            @sprintf("%.17g", exact_free_energy), @sprintf("%.17g", absolute_error),
            @sprintf("%.17g", relative_error), @sprintf("%.17g", elapsed),
        ), ','))
        flush(io)
        @printf(
            "backend=%s beta=%.9f chi=%d converged=%s galerkin=%.3e relerr=%.3e time=%.3fs\n",
            backend, beta, chi, diagnostics.converged,
            diagnostics.galerkin_residual, relative_error, elapsed,
        )
        flush(stdout)
    end
end
println("wrote ", output)
