# Generic complex even rails and non-product caps, without assuming a
# Gaussian state, reflection symmetry, positivity, or a fitted entropy.
using Test, TensorKit, Random
include("../../src/FermionicPEPS.jl")
using .FermionicPEPS
include("../../benchmark/character_lmps_measurement.jl")
@testset "character vs unfactorized signed generic replica graph" begin
    rng=MersenneTwister(27019)
    V=Vect[FermionParity](0=>1,1=>1)
    for dims in ((1,1),(2,1))
        C=Vect[FermionParity](0=>dims[1],1=>dims[2])
        rail(chi,ket,bra)=randn(rng,ComplexF64,chi⊗ket⊗bra←chi)
        seams=(AB=(rail(C',V,V'),rail(C,V',V)),
               AC=(rail(C,V,V'),rail(C,V',V)),
               BC=(rail(C',V,V'),rail(C',V',V)))
        caps=NamedTuple{(:AB,:AC,:BC)}(Tuple(randn(rng,ComplexF64,
            space(X,1)⊗space(Y,1)←one(V)) for (X,Y) in values(seams)))
        for depth in (1,2)
            full=finite_lmps_sectors(seams,caps;depth,factorized=false)
            character=character_lmps_sectors(seams,caps;depth)
            for name in keys(full)
                error=FermionicPEPS._sector_relative_error(getproperty(full,name),getproperty(character,name))
                @test error<1e-10
                println("chi=",dims," depth=",depth," ",name," error=",error)
            end
            flush(stdout)
        end
    end
end
