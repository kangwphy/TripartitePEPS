# Verify complex graded scalar invariance under contraction-tree changes.
# No rail symmetry, positivity or Gaussian structure is assumed.
using Test, TensorKit, Random, LinearAlgebra
include("../../src/FermionicPEPS.jl")
using .FermionicPEPS
@testset "optimized graded junction vs replica-number contraction" begin
    rng=MersenneTwister(81942)
    definitions=(((1,),(1,),(1,)),((2,1),(1,2),(1,2)),
        ((1,2),(2,1),(1,2)),((1,2),(1,2),(2,1)),
        ((1,2,3,4),(2,1,4,3),(3,4,1,2)))
    for dims in ((1,1),(2,1)), permutations in definitions, factorized in (true,false)
        V=Vect[FermionParity](0=>dims[1],1=>dims[2])
        k=length(first(permutations))
        specs=NamedTuple[]
        for (name,X,Y) in FermionicPEPS._REGIONAL_SEAMS
            x=findfirst(==(X),(:A,:B,:C)); y=findfirst(==(Y),(:A,:B,:C))
            cycles=factorized ? FermionicPEPS._relative_cycles(permutations[x],permutations[y]) : [collect(1:k)]
            append!(specs,[(;name,X,Y,copies) for copies in cycles])
        end
        owner=Dict((X,c)=>3(c-1)+x for (x,X) in enumerate((:A,:B,:C)) for c in 1:k)
        labels=[[owner[(X,c)] for c in s.copies for X in (s.X,s.Y)] for s in specs]
        first_spaces=Dict{Int,typeof(V)}()
        endpoints=map(labels) do ls
            spaces=map(ls) do label
                if haskey(first_spaces,label)
                    dual(first_spaces[label])
                else
                    first_spaces[label]=rand(rng,Bool) ? V : V'
                end
            end
            E=randn(rng,ComplexF64,foldl(⊗,spaces)←one(V))
            E/norm(E)
        end
        reference=ComplexF64(ncon(endpoints,labels))
        optimized=FermionicPEPS._close_replica_junction(endpoints,specs,k)
        @test abs(optimized-reference)<=1e-11*max(abs(reference),1e-6)
        if k==4 && factorized
            dimensions=Dict(label=>Float64(dim(space(E,i)))
                for (E,ls) in zip(endpoints,labels) for (i,label) in enumerate(ls))
            plan=FermionicPEPS._junction_slice_plan(labels,dimensions)
            @test plan!==nothing
            sliced=FermionicPEPS._sliced_junction_close(endpoints,labels,plan)
            @test sliced.slices==sum(dims)
            @test abs(sliced.value-reference)<=1e-11*max(abs(reference),1e-6)
        end
    end
end
