using Test, Random, TensorKit, LinearAlgebra
include("../../src/FermionicPEPS.jl")
using .FermionicPEPS
include("../../benchmark/even_replica_support.jl")

function character_overlap(psi,regions,permutations)
    c=even_replica_character(permutations)
    ordering=sortperm(regions;by=x->findfirst(==(x),(:A,:B,:C)))
    v=permute(psi,(Tuple(ordering),())); regs=regions[ordering]
    product=foldl((a,_)->a⊗v,2:c.copies;init=v)
    axes=[length(regs)*(r-1)+i for r in 1:c.copies for i in eachindex(regs)
          if c.insertions[3(r-1)+findfirst(==(regs[i]),(:A,:B,:C))]]
    inserted=twist!(copy(product),axes)
    dot(product,permute(inserted,(FermionicPEPS._replica_order(regs,permutations),())))
end

@testset "even ket and bra support vs physical occupation oracle" begin
    rng=MersenneTwister(48191); V=Vect[FermionParity](0=>1,1=>1)
    definitions=(Z1=((1,),(1,),(1,)),Z2A=((2,1),(1,2),(1,2)),
        Z2B=((1,2),(2,1),(1,2)),Z2C=((1,2),(1,2),(2,1)),
        Z4=((1,2,3,4),(2,1,4,3),(3,4,1,2)))
    for regs in ([:A,:B,:C],[:C,:A,:B,:A])
        psi=randn(rng,ComplexF64,V^length(regs)←one(V)); normalize!(psi)
        expected=occupation_sectors(vec(convert(Array,psi)),regs)
        for (name,permutations) in pairs(definitions)
            actual=character_overlap(psi,regs,permutations)
            @test actual ≈ getproperty(expected,name) atol=1e-12 rtol=1e-12
        end
    end
    character=even_replica_character(definitions.Z4)
    @test character.allowed_count==32
    for w in (.25,1.0)
        finite=measure_finite_fpeps_stilde(ksvc_ipeps(;bond_weight=(w,w));L=2,width=3,cut=1)
        total=finite.sectors.Z4
        for term in total.sectors
            insertion_mask(term.insertions) in character.matching_masks || continue
            @test FermionicPEPS._sector_relative_error(term.result,total)<1e-11
        end
    end
end
