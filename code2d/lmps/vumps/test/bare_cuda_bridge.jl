using LMPSVUMPS, CUDA, Adapt, Serialization, Test
isdefined(@__MODULE__,:BareCUDA) || include(joinpath(@__DIR__, "../src/algorithms/bare_cuda/BareCUDA.jl"))
CUDA.functional() || error("CUDA unavailable")
CUDA.allowscalar(false)
const saved=ENV["TFIM_CUDA_BRIDGE_INPUT"]
@testset "saved physical D4 TensorKit CUDA view bridge" begin
    for side in (:north,:south,:east,:west)
        p=deserialize(joinpath(saved,"boundary_$(side).jls"))
        b=p.trials[1]
        @test b.converged
        @test b.state.AL[1].data isa CuArray
        M=BareCUDA.boundary_array(b)
        a=BareCUDA.transfer_array(b)
        @test M isa CuArray
        @test a isa CuArray
        @test Array(M) ≈ LMPSVUMPS._quantum_boundary_tensor(b) rtol=1e-13 atol=1e-14
        @test Array(a) ≈ LMPSVUMPS._quantum_transfer_site(b) rtol=1e-13 atol=1e-14
    end
end
println("PASS physical D4 CUDA bridge; starting separate full benchmark")
