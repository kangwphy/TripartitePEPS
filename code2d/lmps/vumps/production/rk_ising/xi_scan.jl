# Correlation length of the converged double-layer VUMPS boundary MPS.
#
# This uses the same one-copy boundary transfer channel as the CTMRG `xi1`:
# if lambda_1,lambda_2 are the two largest-modulus eigenvalues of
# sum_p conj(M[p]) (x) M[p], then xi=-1/log|lambda_2/lambda_1|.

using LMPSVUMPS
using MPSKit
using Printf
using Random

csv_values(::Type{T}, name, fallback) where {T} =
    haskey(ENV, name) ? parse.(T, split(ENV[name], ',')) : collect(fallback)

betac = 0.5 * log(1 + sqrt(2))
betas = csv_values(Float64, "VUMPS_XI_BETAS", (0.30, betac))
chis = csv_values(Int, "VUMPS_XI_CHIS", (2, 4, 6, 8))
out = get(ENV, "VUMPS_XI_OUT", joinpath(
    @__DIR__, "..", "..", "data", "sixr", "vumps_xi_cpu.csv",
))
mkpath(dirname(out))

open(out, "w") do io
    println(io, "beta,chi,xi,lambda1_abs,lambda2_abs,lambda2_over_lambda1," *
                "galerkin,boundary_sec,spectrum_sec")
    for beta in betas, chi in chis
        # Match the production six-R scan seed exactly.  The physical spectrum
        # is gauge invariant; fixing the seed makes numerical reproductions
        # byte-level comparable as well.
        Random.seed!(0x620000 + 1000chi + round(Int, 1_000_000beta))
        local boundary
        boundary_sec = @elapsed boundary = solve_vumps_boundary(
            rk_ising(beta), chi;
            backend=:cpu,
            tolerance=parse(Float64, get(ENV, "VUMPS_XI_VUMPS_TOL", "1e-9")),
            diagnostic_tolerance=parse(
                Float64, get(ENV, "VUMPS_XI_DIAGNOSTIC_TOL", "1e-7"),
            ),
            max_iterations=parse(Int, get(ENV, "VUMPS_XI_MAXITER", "800")),
            verbosity=0,
        )
        local spectrum
        spectrum_sec = @elapsed spectrum = MPSKit.transfer_spectrum(
            boundary.state;
            num_vals=min(8, chi^2),
            tol=parse(Float64, get(ENV, "VUMPS_XI_SPECTRUM_TOL", "1e-11")),
        )
        magnitudes = sort(abs.(spectrum); rev=true)
        length(magnitudes) >= 2 || error("chi=$chi returned fewer than two eigenvalues")
        ratio = magnitudes[2] / magnitudes[1]
        0 < ratio < 1 || error("invalid transfer ratio beta=$beta chi=$chi: $ratio")
        xi = -1 / log(ratio)
        @printf(io, "%.16g,%d,%.16e,%.16e,%.16e,%.16e,%.6e,%.6f,%.6f\n",
                beta, chi, xi, magnitudes[1], magnitudes[2], ratio,
                boundary.diagnostics.galerkin_residual, boundary_sec, spectrum_sec)
        flush(io)
        @printf("beta=%.12f chi=%d xi=%.10g ratio=%.12g boundary=%.3fs spectrum=%.3fs\n",
                beta, chi, xi, ratio, boundary_sec, spectrum_sec)
        flush(stdout)
    end
end
println("wrote $out")
