include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, Random, Test
include("../../benchmark/character_lmps_measurement.jl")
@testset "explicit even centers: character versus full signed replica graph" begin
    rng=MersenneTwister(82951)
    V=Vect[FermionParity](0=>1,1=>1)
    definitions=(Z1=((1,),(1,),(1,)),Z2A=((2,1),(1,2),(1,2)),
        Z2B=((1,2),(2,1),(1,2)),Z2C=((1,2),(1,2),(2,1)),
        Z4=((1,2,3,4),(2,1,4,3),(3,4,1,2)))
    for dims in ((1,1),(2,1))
        W=Vect[FermionParity](0=>dims[1],1=>dims[2])
        rail(chi,ket,bra)=randn(rng,ComplexF64,chi⊗ket⊗bra←chi)
        seams=(AB=(rail(W',V,V'),rail(W,V',V)),
               AC=(rail(W,V,V'),rail(W,V',V)),
               BC=(rail(W',V,V'),rail(W',V',V)))
        caps=NamedTuple{(:AB,:AC,:BC)}(Tuple(randn(rng,ComplexF64,
            space(X,1)⊗space(Y,1)←one(V)) for (X,Y) in values(seams)))
        centers=(A=randn(rng,ComplexF64,W⊗W'←one(V)),
                 B=randn(rng,ComplexF64,W'⊗W←one(V)),
                 C=randn(rng,ComplexF64,W'⊗W←one(V)))
        character=character_lmps_sectors(seams,caps;depth=2,centers)
        for (name,p) in pairs(definitions)
            full=finite_occupation_sixr(seams,p,caps;depth=2,factorized=false,centers)
            err=FermionicPEPS._sector_relative_error(full,getproperty(character,name))
            println(dims," ",name," error=",err); flush(stdout)
            @test err<1e-10
        end
        fingerprint=character_geometry_fingerprint(seams,caps;centers)
        @test fingerprint==character_geometry_fingerprint(deepcopy(seams),deepcopy(caps);centers=deepcopy(centers))
        changed=deepcopy(centers); changed.A.data .*= 2
        @test fingerprint!=character_geometry_fingerprint(seams,caps;centers=changed)
    end
end
