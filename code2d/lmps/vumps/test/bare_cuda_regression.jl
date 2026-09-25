using LMPSVUMPS, CUDA, Adapt, ITensors, LinearAlgebra, Random, Test
isdefined(@__MODULE__,:BareCUDA) || include(joinpath(@__DIR__, "../src/algorithms/bare_cuda/BareCUDA.jl"))
CUDA.functional() || error("CUDA unavailable")
CUDA.allowscalar(false)
println("GPU=", CUDA.name(CUDA.device())); flush(stdout)
const C=BareCUDA
const rng=Xoshiro(41)
@testset "complex oriented bare/direct CPU CUDA parity" begin
    chi=2; dp=4
    ML=randn(rng,ComplexF64,chi,dp,chi); MR=randn(rng,ComplexF64,chi,dp,chi)
    a=randn(rng,ComplexF64,dp,dp,dp,dp)
    cpu=LMPSVUMPS.direct_original_lmps_fixedpoints(ML,MR,a;nmax=4)
    gpu=C.direct_original_lmps_fixedpoints(CuArray(ML),CuArray(MR),CuArray(a);nmax=4)
    for f in (:G,:B,:H)
        @test Array(getproperty(gpu,f)) ≈ getproperty(cpu,f) rtol=1e-10 atol=1e-11
    end
    for f in (:lambda_G,:lambda_B,:residual_G,:residual_B)
        @test getproperty(gpu,f) ≈ getproperty(cpu,f) rtol=1e-10 atol=1e-11
    end
    rails=C.direct_cardinal_regional_rails(gpu,CuArray(ML),CuArray(MR),CuArray(ML))
    specs=C.m2_specs(rails.rails,2,2,[2,1],[1,2],[1,2])
    for spec in specs
        Ts=spec.Ts
        lefts,lmap,rmap=C.m2_left_right(Ts)
        x=randn(rng,ComplexF64,Tuple(dim.(lefts)))
        yg,_=C.m2_apply_array(Ts,lefts,lmap,rmap,CuArray(x))
        yc,_=LMPSVUMPS.m2_apply_array([adapt(Array,t) for t in Ts],lefts,lmap,rmap,x)
        @test Array(yg) ≈ yc rtol=1e-10 atol=1e-11
        @test C.m2_adjoint_residual(Ts) <= 1e-11
    end
    specs4=C.m2_specs(rails.rails,2,4,[1,2,3,4],[2,1,4,3],[3,4,1,2])
    Rs=[ITensor(CuArray(randn(rng,ComplexF64,Tuple(dim.(s.lefts)))),s.lefts...) for s in specs4]
    vertices=[(s.Xn,s.Yn,s.cyc) for s in specs4]
    zg,_=C.gpu_sliced_octahedron(Rs,vertices)
    zc,_=LMPSVUMPS.m2_close_octahedron([adapt(Array,r) for r in Rs],vertices)
    @test zg ≈ zc rtol=1e-10 atol=1e-11

end
println("stage=full six-R product-boundary test"); flush(stdout)
@testset "full six-R CPU CUDA product state" begin
    M=reshape([1.,0.,0.,0.],1,4,1)
    orig=(G=ones(1,1),B=M)
    cr=LMPSVUMPS.direct_cardinal_regional_rails(orig,M,M,M)
    gr=C.direct_cardinal_regional_rails((G=CuArray(orig.G),B=CuArray(M)),CuArray(M),CuArray(M),CuArray(M))
    cpu=LMPSVUMPS.m2_contract_regional_six_R(cr.rails,2;nmax=100)
    gpu=C.m2_contract_regional_six_R(gr.rails,2;nmax=100)
    for f in (:tildeS,:S2A,:S2B,:S2C,:S3,:scale_gate)
        @test getproperty(gpu,f) ≈ getproperty(cpu,f) atol=1e-10
    end
end
CUDA.synchronize()
println("PASS CUDA bare/direct and six-R parity; not a physical benchmark")
