include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, Test, LinearAlgebra, Random, TensorKit
@testset "Infinite-local RDM occupation observable versus pure CAR cube" begin
    rng=MersenneTwister(481)
    for na in (1,2),nb in (1,2)
        nc=2; n=na+nb+nc
        psi=randn(rng,ComplexF64,2^n)
        for mask in 0:2^n-1
            isodd(count_ones(mask)) && (psi[mask+1]=0)
        end
        normalize!(psi)
        matrix=reshape(psi,2^(na+nb),2^nc)
        rho=matrix*matrix'
        actual=FermionicPEPS._occupation_rdm_sectors(rho,na)
        expected=occupation_sectors(psi,vcat(fill(:A,na),fill(:B,nb),fill(:C,nc)))
        for name in (:Z2A,:Z2B,:Z2C,:Z4)
            @test getproperty(actual.sectors,name) ≈ getproperty(expected,name) atol=2e-12
        end
        @test actual.stilde ≈ expected.stilde atol=2e-12
        V=Vect[FermionParity](0=>1,1=>1)
        P=Diagonal([(-1)^count_ones(i) for i in 0:2^(na+nb)-1])
        categorical=TensorMap(reshape(rho*P,ntuple(_->2,2*(na+nb))),V^(na+nb)←V^(na+nb))
        @test FermionicPEPS._occupation_rdm(categorical) ≈ rho atol=1e-12
    end
    @test_throws ErrorException FermionicPEPS._occupation_rdm_sectors(Diagonal([1.1,-.1,0.,0.]),1)
end
