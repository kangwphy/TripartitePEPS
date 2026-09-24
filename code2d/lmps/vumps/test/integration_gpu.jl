using Test
using Random
using LinearAlgebra
using CUDA
using cuTENSOR
using LMPSVUMPS

CUDA.functional() || error("GPU VUMPS regression requires functional CUDA")
cuTENSOR.functional() || error("GPU VUMPS regression requires functional cuTENSOR")
CUDA.allowscalar(false)

@testset "VUMPS CPU/CUDA equivalence and local LTR audit" begin
    source = rk_ising(0.30)
    settings = (
        tolerance=1e-9,
        diagnostic_tolerance=1e-7,
        max_iterations=400,
        verbosity=0,
    )
    Random.seed!(0x51a7)
    cpu = solve_vumps_boundary(source, 2; backend=:cpu, settings...)
    Random.seed!(0x51a7)
    gpu = solve_vumps_boundary(source, 2; backend=:cuda, settings...)
    CUDA.synchronize()
    @test cpu.diagnostics.converged
    @test gpu.diagnostics.converged
    @test vumps_storage_backend(gpu) === :cuda
    @test vumps_logz(gpu) ≈ vumps_logz(cpu) rtol=1e-10 atol=1e-11

    cpu_ltr = vumps_ltr_objects(cpu)
    gpu_ltr = vumps_ltr_objects(gpu)
    @test cpu_ltr.diagnostics.compatible
    @test gpu_ltr.diagnostics.compatible
    @test cpu_ltr.diagnostics.block_axis_residual < 1e-7
    @test gpu_ltr.diagnostics.block_axis_residual < 1e-7
    @test cpu_ltr.diagnostics.metric_axis_residual < 1e-7
    @test gpu_ltr.diagnostics.metric_axis_residual < 1e-7
    # Retained-bond tensors may differ by a harmless gauge rotation.  Their
    # metric singular values and scalar transfer eigenvalue are invariant.
    @test svdvals(gpu_ltr.G) ≈ svdvals(cpu_ltr.G) rtol=1e-8 atol=1e-10
    @test_throws ErrorException vumps_sixr_rails(cpu)
    @test_throws ErrorException vumps_sixr_rails(gpu)
end
