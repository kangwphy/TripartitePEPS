using Test
using LinearAlgebra
using LMPSVUMPS

@testset "quantum TFIM ground state to independent-region six-R" begin
    model = TransverseIsing2D(; J=1.0, h=0.8, unitcell=(2, 2))
    gs = solve_tfim_groundstate(
        model; D=2, seed=20260823, dt_schedule=(1e-1, 3e-2), nstep=3,
        tolerance=0, check_interval=0, verbosity=0,
    )
    @test size(gs.peps) == (2, 2)
    @test all(isfinite, gs.truncation_errors)
    println("TFIM GS truncation_errors=", gs.truncation_errors)
    physical_peps = materialize_simple_update_state(gs)
    one_site = quantum_one_site_representative(physical_peps)
    @test size(one_site) == (1, 1)

    run = solve_quantum_vumps_sixr(
        one_site, 2;
        # This deliberately nonsymmetric representative exercises three
        # independent regional frames and their mixed K_AB/K_AC/K_BC solves.
        assume_spatial_symmetry=false,
        boundary_kwargs=(;
            tolerance=1e-4, diagnostic_tolerance=1e-3,
            max_iterations=180, verbosity=0, accept_unconverged=true,
        ),
        bridge_kwargs=(;
            require_converged=false, tolerance=1e-3,
            mixed_tolerance=1e-8, mixed_nmax=3000,
        ),
        sixr_kwargs=(; nmax=300, tol=1e-7, verbose=false),
    )
    out = run.result
    obs = tfim_one_site_observables(run.boundaries.A)
    @test out.endpoint_count == 6
    @test out.metric_count == 3
    @test Set(x.seam for x in out.mixed_corner_audit) == Set(("AB", "AC", "BC"))
    @test isfinite(out.tildeS)
    @test isfinite(out.S3)
    @test all(isfinite, (out.S2A, out.S2B, out.S2C))
    @test all(isfinite, (obs.x, obs.z))
    @test out.map_residual < 1e-5
    @test run.closure_rails.A === run.rails.A
    @test run.closure_rails.B === run.rails.B
    @test run.closure_rails.C === run.rails.C
    @test run.closure_rails.A !== run.closure_rails.B
    @test run.closure_rails.A !== run.closure_rails.C
    println("quantum six-R: tildeS=$(out.tildeS), S3=$(out.S3), " *
        "S2A=$(out.S2A), S2B=$(out.S2B), S2C=$(out.S2C)")
    println("one-site GS observables from A boundary: x=$(obs.x), z=$(obs.z)")
    println("timing: boundary=$(run.boundary_seconds)s, rectangles=$(run.rectangle_seconds)s, " *
        "closure=$(run.closure_seconds)s, total=$(run.total_seconds)s")
end
