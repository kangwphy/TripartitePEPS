include("../../src/FermionicPEPS.jl")
using .FermionicPEPS,TensorKit,Random,Test
rng=MersenneTwister(923)
V=Vect[FermionParity](0=>1,1=>1)
cube=((1,2,3,4),(2,1,4,3),(3,4,1,2))
expansion=replica_parity_expansion(cube)
@test length(expansion.terms)==16
@test all(t->abs(t.weight)==0.25,expansion.terms)
for regions in ([:A,:B,:C],[:C,:A,:B,:A])
    N=length(regions)
    psi=randn(rng,ComplexF64,foldl(⊗,fill(V,N))←one(V))
    direct=tensor_replica_sectors(psi,regions)
    expanded=tensor_replica_parity_overlap(psi,regions,cube)
    @show regions direct.Z4 expanded
    @test expanded ≈ direct.Z4 rtol=1e-12
    for x in 1:3
        permutations=ntuple(j->j==x ? (2,1) : (1,2),3)
        expanded2=tensor_replica_parity_overlap(psi,regions,permutations)
        @test expanded2 ≈ getproperty(direct,(:Z2A,:Z2B,:Z2C)[x]) rtol=1e-12
    end
    flush(stdout)
end
