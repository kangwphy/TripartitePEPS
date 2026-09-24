using Test,LinearAlgebra,Random
include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit
V=Vect[FermionParity](0=>1,1=>1)
Random.seed!(123)
for N in 3:4
    t=randn(ComplexF64,V^N←one(V))
    @test convert(Array,occupation_permute(t,Tuple(N:-1:1))) ≈
        permutedims(convert(Array,t),Tuple(N:-1:1)) atol=1e-12
    regions=N==3 ? [:A,:B,:C] : [:A,:B,:A,:C]
    raw=vec(convert(Array,t))
    typed=tensor_replica_sectors(t,regions)
    exact=occupation_sectors(raw,regions)
    @show N typed exact;flush(stdout)
    for key in (:Z1,:Z2A,:Z2B,:Z2C,:Z4,:stilde)
        @test getproperty(typed,key) ≈ getproperty(exact,key) rtol=1e-11 atol=1e-12
    end
end
