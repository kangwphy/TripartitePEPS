using Test
using Random
using Serialization
using LMPSVUMPS
using ITensors

@testset "CPU VUMPS boundary and audited local LTR construction" begin
    beta = 0.30
    Random.seed!(0x51a7)
    boundary = solve_vumps_boundary(
        rk_ising(beta), 2;
        backend=:cpu,
        tolerance=1e-9,
        diagnostic_tolerance=1e-7,
        max_iterations=400,
        verbosity=0,
    )
    @test boundary.diagnostics.converged
    @test vumps_storage_backend(boundary) === :cpu
    f = vumps_free_energy(boundary, beta)
    @test abs(f - onsager_f(beta)) / abs(onsager_f(beta)) < 2e-5

    observables = vumps_boundary_observables(boundary, beta; spectrum_num=4)
    @test observables.xi > 0
    @test observables.free_energy == f
    @test observables.free_energy_rel_error < 2e-5
    @test abs(observables.magnetization) < 1e-6
    @test observables.exact_magnetization == 0
    @test all(isfinite, (observables.entropy_vn,
                         observables.entropy_renyi2,
                         observables.entropy_renyi3,
                         observables.entropy_renyi4,
                         observables.entropy_infinity))
    @test sum(observables.schmidt_probabilities) ≈ 1
    @test ising_exact_magnetization(0.5) ≈ 0.911319377877496 atol=1e-12
    checkpoint = tempname()
    open(checkpoint, "w") do io
        serialize(io, (format_version=1, beta=beta, chi=2, boundary=boundary))
    end
    restored = open(deserialize, checkpoint).boundary
    rm(checkpoint)
    @test restored.diagnostics.row_logz == boundary.diagnostics.row_logz
    @test vumps_boundary_observables(restored, beta; spectrum_num=4).xi ≈
          observables.xi

    ltr = vumps_ltr_objects(boundary)
    @test ltr.diagnostics.compatible
    @test ltr.diagnostics.ket_dimension == 2
    @test ltr.diagnostics.bra_dimension == 2
    @test ltr.diagnostics.block_axis_residual < 1e-7
    @test ltr.diagnostics.metric_axis_residual < 1e-7
    @test ltr.diagnostics.left_fixedpoint_residual < 1e-7
    @test ltr.diagnostics.right_fixedpoint_residual < 1e-7
    @test size(ltr.B) == (2, 4, 2)
    @test size(ltr.G) == (2, 2)

    projector = vumps_projector_objects(boundary)
    @test projector.diagnostics.compatible
    @test projector.diagnostics.mixed_left_residual < 1e-10
    @test projector.diagnostics.mixed_right_residual < 1e-10
    @test projector.diagnostics.local_fixedpoint_residual < 1e-7
    @test projector.diagnostics.projector_p2_residual < 1e-10
    @test projector.diagnostics.projector_left_residual < 1e-10
    @test projector.diagnostics.projector_right_residual < 1e-10
    @test projector.diagnostics.metric_dropped == 0
    @test size(projector.L) == (2, 8)
    @test size(projector.R) == (8, 2)
    rails = vumps_sixr_rails(boundary)
    @test all(hasproperty(rails, name) for name in (:hL, :hR, :vL, :vR))
    replica = m2_contract_shared_six_R(
        rails, boundary.source.D; nmax=1000, tol=1e-10, verbose=false,
    )
    @test replica.tildeS ≈ 1.065617485e-2 atol=2e-8
    @test replica.endpoint_residual < 1e-8
    @test replica.scale_gate < 1e-8
    @info "CPU VUMPS local LTR audit" values=(
        beta=beta, chi=2, free_energy=f, ltr=ltr.diagnostics,
        projector=projector.diagnostics, tildeS=replica.tildeS,
    )
end
