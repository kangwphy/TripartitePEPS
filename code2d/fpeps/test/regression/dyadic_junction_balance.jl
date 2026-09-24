using Test, Random
include("../../benchmark/audit_dyadic_junction_balance.jl")

@testset "dyadic gauges preserve complex graded junction" begin
    rng=MersenneTwister(13915)
    for dims in ((1,1),(2,1))
        V=Vect[FermionParity](0=>dims[1],1=>dims[2])
        permutations=((1,2,3,4),(2,1,4,3),(3,4,1,2))
        owner=Dict((X,c)=>3(c-1)+x for (x,X) in enumerate((:A,:B,:C)) for c in 1:4)
        specs=NamedTuple[]
        for (name,X,Y) in FermionicPEPS._REGIONAL_SEAMS
            x=findfirst(==(X),(:A,:B,:C));y=findfirst(==(Y),(:A,:B,:C))
            append!(specs,[(;name,X,Y,copies) for copies in
                FermionicPEPS._relative_cycles(permutations[x],permutations[y])])
        end
        network=[[owner[(X,c)] for c in s.copies for X in (s.X,s.Y)] for s in specs]
        firstspaces=Dict{Int,typeof(V)}()
        endpoints=map(network) do labels
            ports=map(labels) do label
                if haskey(firstspaces,label)
                    dual(firstspaces[label])
                else
                    firstspaces[label]=rand(rng,Bool) ? V : V'
                end
            end
            E=randn(rng,ComplexF64,foldl(⊗,ports)←one(V));E/norm(E)
        end
        reference=ComplexF64(ncon(endpoints,network))
        gauged=deepcopy(endpoints)
        for label in sort(unique(vcat(network...)))
            (a,i),(b,j)=[(a,i) for (a,ls) in enumerate(network) for (i,l) in enumerate(ls) if l==label]
            exponents=Dict(q=>rand(rng,-6:6,dim(space(gauged[a],i),q)) for q in sectors(space(gauged[a],i)))
            exact_port_scale!(gauged[a],i,exponents)
            exact_port_scale!(gauged[b],j,Dict(q=>-v for (q,v) in exponents))
        end
        balanced=balance_frozen_endpoints(gauged,network)
        @test last(balanced.history)["log_frobenius_envelope_ratio"]<0
        dimensions=Dict(label=>Float64(dim(space(E,i))) for (E,ls) in zip(endpoints,network) for (i,label) in enumerate(ls))
        plan=FermionicPEPS._junction_slice_plan(network,dimensions)
        for p in (plan,second_plan(network,plan))
            closed=FermionicPEPS._sliced_junction_close(balanced.tensors,network,p)
            @test abs(closed.value-reference)<1e-11*max(abs(reference),1e-6)
        end
        @test abs(ComplexF64(ncon(balanced.tensors,network))-reference)<1e-11*max(abs(reference),1e-6)
    end
end
