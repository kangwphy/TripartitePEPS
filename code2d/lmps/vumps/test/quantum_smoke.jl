using Test
using LinearAlgebra
using LMPSVUMPS

@testset "quantum SimpleUpdate to four directional VUMPS smoke" begin
    model = TransverseIsing2D(; J=1.0, h=0.6, unitcell=(2, 2))
    gs = solve_tfim_groundstate(
        model; D=2, seed=123, dt_schedule=(1e-1,), nstep=2,
        tolerance=0, check_interval=0, verbosity=0,
    )
    @test size(gs.peps) == (2, 2)
    @test length(gs.truncation_errors) == 1
    @test all(isfinite, gs.truncation_errors)

    # Use an intentionally asymmetric random PEPS and solve all sides.  The
    # test checks independent construction and finite fixed-point data; the
    # strict production convergence gate remains the default API behavior.
    physical_peps = materialize_simple_update_state(gs)
    @test physical_peps === gs.peps
    strict_north = solve_quantum_vumps_boundary(
        physical_peps, 2; side=:north, tolerance=1e-7, diagnostic_tolerance=1e-5,
        max_iterations=120, verbosity=0,
    )
    @test strict_north.converged

    boundaries = solve_quantum_vumps_boundaries(
        physical_peps, 2; tolerance=1e-7, diagnostic_tolerance=1e-5,
        max_iterations=120, verbosity=0, accept_unconverged=true,
    )
    for boundary in (boundaries.north, boundaries.south,
                     boundaries.east, boundaries.west)
        @test boundary.side in (:north, :south, :east, :west)
        @test isfinite(real(boundary.row_eigenvalue))
        @test isfinite(boundary.galerkin_residual)
        obs = tfim_one_site_observables(boundary)
        @test all(isfinite, (obs.x, obs.z))
    end
end
