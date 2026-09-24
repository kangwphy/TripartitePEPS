using Test
using Serialization
using LMPSVUMPS

const STATE_PATH = get(ENV, "QUANTUM_REGRESSION_STATE", joinpath(
    @__DIR__, "..", "data", "quantum_tfim", "gs_one_site_v1", "states",
    "h3.100000_D2_ctm16_ad_random_1x1.jls",
))
const CHI = parse(Int, get(ENV, "QUANTUM_REGRESSION_CHI", "2"))

@testset "independent regional gauges reproduce a symmetric shared closure" begin
    payload = open(deserialize, STATE_PATH)
    peps = payload.groundstate.peps
    run = solve_quantum_vumps_sixr(
        peps, CHI;
        assume_spatial_symmetry=false,
        boundary_trials=8,
        boundary_kwargs=(;
            tolerance=1e-7, diagnostic_tolerance=1e-5,
            max_iterations=800, verbosity=0, accept_unconverged=true,
        ),
        bridge_kwargs=(;
            require_converged=false, tolerance=1e-5,
            mixed_tolerance=1e-10, mixed_nmax=20_000,
        ),
        sixr_kwargs=(; nmax=800, tol=1e-9, verbose=false),
    )
    shared = m2_contract_shared_six_R(
        run.rails.A, run.D; nmax=800, tol=1e-9, verbose=false,
    )
    fast = solve_quantum_vumps_sixr(
        peps, CHI;
        assume_spatial_symmetry=true,
        boundary_trials=8,
        boundary_kwargs=(;
            tolerance=1e-7, diagnostic_tolerance=1e-5,
            max_iterations=800, verbosity=0, accept_unconverged=true,
        ),
        bridge_kwargs=(;
            require_converged=false, tolerance=1e-5,
            mixed_tolerance=1e-10, mixed_nmax=20_000,
        ),
        sixr_kwargs=(; nmax=800, tol=1e-9, verbose=false),
    )

    obs = map(tfim_one_site_observables,
              (run.boundaries.A, run.boundaries.B, run.boundaries.C))
    observable_spread = maximum((
        abs(obs[1].x - obs[2].x), abs(obs[1].x - obs[3].x),
        abs(obs[1].z - obs[2].z), abs(obs[1].z - obs[3].z),
    ))
    stilde_difference = abs(run.result.tildeS - shared.tildeS)

    @test all(b.converged for b in
              (run.boundaries.A, run.boundaries.B, run.boundaries.C))
    @test observable_spread < 1e-5
    @test run.result.endpoint_residual < 1e-7
    @test run.result.metric_residual < 1e-7
    @test stilde_difference < 1e-6
    @test fast.boundaries.A === fast.boundaries.B
    @test fast.boundaries.A === fast.boundaries.C
    @test fast.objects.A === fast.objects.B
    @test fast.objects.A === fast.objects.C
    @test fast.rails.A === fast.rails.B
    @test fast.rails.A === fast.rails.C
    @test fast.closure_rails === fast.rails.A
    @test abs(fast.result.tildeS - shared.tildeS) < 1e-10
    println("regional/shared symmetric regression: regional=$(run.result.tildeS), " *
            "shared=$(shared.tildeS), difference=$stilde_difference, " *
            "observable_spread=$observable_spread")
end
