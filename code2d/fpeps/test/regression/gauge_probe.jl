include("../../src/FermionicPEPS.jl")
using .FermionicPEPS,TensorKit,Random,LinearAlgebra,Test
rng=MersenneTwister(7)
V=Vect[FermionParity](0=>1,1=>1)
for chi in (V,V')
    T=randn(rng,ComplexF64,chi⊗V⊗V'←chi)
    U=id(ComplexF64,chi)
    for (sector,block) in blocks(U)
        block .*= cis(0.3*Int(sector.isodd))
    end
    @tensor old[l k b;r] := U[l;x]*T[x k b;y]*conj(U[r;y])
    corrected=(U⊗id(ComplexF64,V)⊗id(ComplexF64,V'))*T*U'
    @show chi norm(old-corrected)/norm(T)
    seed=randn(rng,ComplexF64,chi⊗chi'←one(V))
    Y=randn(rng,ComplexF64,chi'⊗V'⊗V←chi')
    before=graded_seam_apply(T,Y,(1,),(1,),seed)
    newseed=(U⊗id(ComplexF64,chi'))*seed
    after=graded_seam_apply(corrected,Y,(1,),(1,),newseed)
    expected=(U⊗id(ComplexF64,chi'))*before
    @show norm(after-expected)/norm(expected)
    @test after ≈ expected rtol=1e-12
end
