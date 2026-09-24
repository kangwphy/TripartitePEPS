include("../../src/FermionicPEPS.jl")
using .FermionicPEPS,TensorKit,Random,LinearAlgebra,Test
rng=MersenneTwister(822)
V=Vect[FermionParity](0=>1,1=>1)
X=randn(rng,ComplexF64,V⊗V⊗V'←V)
Y=randn(rng,ComplexF64,V⊗V'⊗V←V)
R=randn(rng,ComplexF64,V⊗V⊗V⊗V←one(V))
S=randn(rng,ComplexF64,V⊗V⊗V⊗V←one(V))
permutations=(((1,2,3,4),(2,1,4,3),([1,2],[3,4])),
              ((1,2,3,4),(3,4,1,2),([1,3],[2,4])),
              ((2,1,4,3),(3,4,1,2),([1,4],[2,3])))
for (pX,pY,(c1,c2)) in permutations
    source=R⊗S
    # Source is cycle-major; change it to X1,Y1,X2,Y2,... with graded swaps.
    grouped=vcat(c1,c2)
    inverse=invperm(grouped)
    order=Tuple(j for c in 1:4 for j in (2inverse[c]-1,2inverse[c]))
    source=permute(source,(order,()))
    direct=graded_seam_apply(X,Y,pX,pY,source)
    result1=graded_seam_apply(X,Y,pX,pY,R;copies=c1)
    result2=graded_seam_apply(X,Y,pX,pY,S;copies=c2)
    factorized=permute(result1⊗result2,(order,()))
    @show pX pY norm(direct-factorized)/norm(direct)
    @test direct ≈ factorized rtol=1e-12
    flush(stdout)
end
