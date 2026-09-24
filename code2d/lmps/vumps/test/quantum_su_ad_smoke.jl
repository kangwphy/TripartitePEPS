using LMPSVUMPS
using PEPSKit
using Serialization
using Test

const COMPAT_CHECKPOINT = get(ENV, "TFIM_COMPAT_CHECKPOINT", "")

if !isempty(COMPAT_CHECKPOINT)
    @testset "existing SimpleUpdate checkpoint remains readable" begin
        payload = deserialize(COMPAT_CHECKPOINT)
        @test payload.simple_update isa SimpleUpdateGroundState
        @test size(payload.simple_update.peps) == (2, 2)
        @test isfinite(payload.observables.energy)
    end
end

@testset "TFIM SimpleUpdate warm start enters one-site differentiable solve" begin
    model = TransverseIsing2D(; J=1.0, h=3.1, unitcell=(1, 1))
    auxiliary_model = TransverseIsing2D(; J=1.0, h=3.1, unitcell=(2, 2))
    early_stage_iterations = Int[]
    early_stop = solve_tfim_groundstate(
        auxiliary_model;
        D=2,
        seed=20260824,
        dt_schedule=(1e-2,),
        nstep=3,
        tolerance=1e6,
        symmetrize_gates=true,
        verbosity=0,
        check_interval=0,
        stage_iterations_out=early_stage_iterations,
    )
    @test early_stop.nstep == 3
    @test early_stage_iterations == [1]

    gs_random = solve_tfim_variational_groundstate(
        model;
        D=2,
        environment_chi=4,
        seed=20260824,
        tolerance=1e-2,
        max_iterations=1,
        spatial_symmetrization=PEPSKit.RotateReflect(),
        initialization=:random,
    )
    @test size(gs_random.peps) == (1, 1)
    @test gs_random.initialization == :random
    @test gs_random.initialization_info.method == :random
    @test isnothing(gs_random.initialization_info.auxiliary_unitcell)
    @test isfinite(gs_random.energy)
    @test isfinite(gs_random.gradient_norm)

    gs = solve_tfim_variational_groundstate(
        model;
        D=2,
        environment_chi=4,
        seed=20260824,
        tolerance=1e-2,
        max_iterations=1,
        spatial_symmetrization=PEPSKit.RotateReflect(),
        initialization=:simple_update,
        simple_update_kwargs=(;
            dt_schedule=(1e-2,), nstep=1, tolerance=0.0,
            symmetrize_gates=true, verbosity=0, check_interval=0,
        ),
    )
    @test size(gs.peps) == (1, 1)
    @test gs.initialization == :simple_update
    @test gs.initialization_info.auxiliary_unitcell == (2, 2)
    @test gs.initialization_info.representative_site == (1, 1)
    @test gs.initialization_info.simple_update_nstep == 1
    @test gs.initialization_info.simple_update_stage_iterations == [1]
    @test length(gs.initialization_info.simple_update_truncation_errors) == 1
    @test isfinite(gs.energy)
    @test isfinite(gs.gradient_norm)

    @test_throws ArgumentError solve_tfim_variational_groundstate(
        model; initialization=:unknown, max_iterations=1,
    )
    @test_throws ArgumentError LMPSVUMPS._tfim_variational_initial_state(
        model; D=1, seed=1, scalar_type=ComplexF64,
        initialization=:simple_update, simple_update_unitcell=(1, 1),
        simple_update_site=(1, 1), simple_update_kwargs=(; nstep=1),
    )
end
