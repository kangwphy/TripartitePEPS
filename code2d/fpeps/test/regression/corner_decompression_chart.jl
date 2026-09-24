# Exact local bridge from a decompressed CTM corner to two oriented MPS
# rays and its center. No fixed point or physical entropy is assumed here.
include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, Random, LinearAlgebra, Test
@testset "graded corner decompression into oriented rays" begin
    rng=MersenneTwister(82531)
    V=Vect[FermionParity](0=>1,1=>1)
    for dims in ((1,1),(2,1),(2,2)), dualphysical in (false,true)
        W=Vect[FermionParity](0=>dims[1],1=>dims[2])
        P=dualphysical ? V' : V
        PL=randn(rng,ComplexF64,W⊗P⊗P'←W)
        PR=randn(rng,ComplexF64,W←W⊗P'⊗P)
        C=randn(rng,ComplexF64,W←W)
        left=permute(PL,((4,2,3),(1,)))
        right=permute(PR,((1,3,4),(2,)))
        center=permute(C,((1,2),()))
        @tensor reconstructed[a p1 p2;b q1 q2] :=
            left[l p1 p2;a]*right[r q1 q2;b]*center[l r]
        exact=PL*C*PR
        @test norm(reconstructed-exact)/norm(exact)<1e-12
        reversed=permute(C,((2,1),()))
        @tensor reconstructed_reversed[a p1 p2;b q1 q2] :=
            left[l p1 p2;a]*right[r q1 q2;b]*reversed[r l]
        @test norm(reconstructed_reversed-exact)/norm(exact)<1e-12
    end
end
