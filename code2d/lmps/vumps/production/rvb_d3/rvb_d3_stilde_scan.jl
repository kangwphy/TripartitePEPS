#!/usr/bin/env julia

"""Exact square-lattice nearest-neighbour RVB (D=3) VUMPS scan.

This driver is deliberately model-specific and does not call any TFIM
constructor.  The boundary algorithm and the audited six-R closure are the
same generic VUMPS/LMPS routines used elsewhere in this project.
"""

using LMPSVUMPS
using MPSKit
using Printf
using Dates

out = get(ENV, "RVB_OUT", joinpath(@__DIR__, "../../data/square_rvb_d3"))
mkpath(out)
chis = parse.(Int, split(get(ENV, "RVB_CHIS", "2,3,4,5,6,8"), ','))
boundary_tol = parse(Float64, get(ENV, "RVB_BOUNDARY_TOL", "1e-8"))
sixr_nmax = parse(Int, get(ENV, "RVB_SIXR_NMAX", "500"))
sixr_tol = parse(Float64, get(ENV, "RVB_SIXR_TOL", "1e-8"))
max_iterations = parse(Int, get(ENV, "RVB_MAXITER", "500"))
far_boundary_tolerance = parse(Float64, get(ENV, "RVB_FAR_TOL", "1e-5"))
far_boundary_tail_tolerance = parse(Float64, get(ENV, "RVB_FAR_TAIL_TOL", "1e-6"))

peps = rvb_d3_peps()
csv = joinpath(out, "scan.csv")
open(csv, "w") do io
    println(io, "model,chi,xi_north,xi_south,xi_mean,stilde,S2A,S2B,S3," *
            "boundary_seconds,rectangle_seconds,closure_seconds,total_seconds," *
            "north_residual,south_residual,error")
    for chi in chis
        @info "RVB D=3 VUMPS/six-R" chi
        local result
        try
            result = solve_quantum_vumps_sixr(
                peps, chi;
                A_side=:north, B_side=:north, C_side=:south,
                boundary_seeds=(10_300 + chi, 20_300 + chi, 30_300 + chi),
                boundary_trials=1,
                assume_spatial_symmetry=false,
                boundary_kwargs=(;
                    tolerance=boundary_tol,
                    diagnostic_tolerance=max(100boundary_tol, 1e-7),
                    max_iterations=max_iterations,
                    verbosity=0,
                ),
                bridge_kwargs=(;
                    route=:common_chart,
                    require_converged=false,
                    tolerance=1e-6,
                    inverse_rtol=1e-10,
                    polish_endpoints=false,
                ),
                sixr_kwargs=(;
                    nmax=sixr_nmax,
                    tol=sixr_tol,
                    far_boundary_tolerance=far_boundary_tolerance,
                    far_boundary_tail_tolerance=far_boundary_tail_tolerance,
                    verbose=false,
                ),
            )
            A = result.boundaries.A
            C = result.boundaries.C
            # This is the boundary-MPS transfer correlation length, not a
            # CTMRG correlation length.  The leading eigenvalue is normalized
            # out by the ratio lambda_2/lambda_1.
            function boundary_xi(boundary)
                vals = MPSKit.transfer_spectrum(boundary.state;
                    num_vals=min(8, chi^2), tol=1e-10)
                vals = sort(abs.(vals), rev=true)
                length(vals) >= 2 || error("boundary transfer spectrum has <2 values")
                vals[2] > 0 && vals[1] > vals[2] || error("invalid boundary spectrum")
                -1 / log(vals[2] / vals[1])
            end
            xi_n = boundary_xi(A)
            xi_s = boundary_xi(C)
            r = result.result
            println(io, join(("square_rvb_d3", chi, xi_n, xi_s,
                              0.5 * (xi_n + xi_s), r.tildeS, r.S2A, r.S2B,
                              r.S3, result.boundary_seconds,
                              result.rectangle_seconds, result.closure_seconds,
                              result.total_seconds, A.galerkin_residual,
                              C.galerkin_residual, ""), ','))
            flush(io)
            @info "completed" chi xi_n xi_s stilde=r.tildeS
        catch err
            bt = sprint(showerror, err)
            @error "RVB point failed" chi exception=(err, catch_backtrace())
            println(io, join(("square_rvb_d3", chi, "NaN", "NaN", "NaN",
                              "NaN", "NaN", "NaN", "NaN", "NaN", "NaN",
                              "NaN", "NaN", "NaN", "NaN", replace(bt, ',' => ';')), ','))
            flush(io)
        end
    end
end
@info "RVB scan written" csv timestamp=now()
