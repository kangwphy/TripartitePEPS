include("../../src/FermionicPEPS.jl")
using .FermionicPEPS,TensorKit,Random,LinearAlgebra,Test
rng=MersenneTwister(72)
V=Vect[FermionParity](0=>1,1=>1)
rail(chi,ket,bra)=randn(rng,ComplexF64,chi⊗ket⊗bra←chi)
seams=(AB=(rail(V',V,V'),rail(V,V',V)),
       AC=(rail(V,V,V'),rail(V,V',V)),
       BC=(rail(V',V,V'),rail(V',V',V)))
caps=NamedTuple{(:AB,:AC,:BC)}(Tuple(randn(rng,ComplexF64,space(X,1)⊗space(Y,1)←one(V))
                                  for (X,Y) in values(seams)))
function scaled_error(x,y)
    scale=max(x.logscale,y.logscale)
    a=x.value*exp(x.logscale-scale); b=y.value*exp(y.logscale-scale)
    abs(a-b)/max(abs(a),abs(b),eps())
end
permutations=(((1,),(1,),(1,)),((2,1),(1,2),(1,2)),
    ((1,2),(2,1),(1,2)),((1,2),(1,2),(2,1)),
    ((1,2,3,4),(2,1,4,3),(3,4,1,2)))
for p in permutations,depth in 0:2
    full=finite_graded_sixr(seams,p,caps;depth,factorized=false)
    six=finite_graded_sixr(seams,p,caps;depth,factorized=true)
    error=scaled_error(full,six)
    @show p depth length(six.endpoints) error
    @test error<1e-10
    flush(stdout)
end
p=last(permutations)
full=finite_occupation_sixr(seams,p,caps;depth=1,factorized=false)
six=finite_occupation_sixr(seams,p,caps;depth=1,factorized=true)
@show scaled_error(full,six) length(six.sectors)
@test scaled_error(full,six)<1e-10
@test length(six.sectors)==16
for (a,b) in zip(full.sectors,six.sectors)
    @test scaled_error(a.result,b.result)<1e-10
end

# Independent phase gauges in all three retained regional frames.
function phase_gauge(space,angle)
    U=id(ComplexF64,space)
    for (sector,data) in blocks(U)
        data .*= cis((isdual(space) ? -1 : 1)*angle*Int(sector.isodd))
    end
    U
end
function gauge_rail(T,U)
    (U⊗id(ComplexF64,space(T,2))⊗id(ComplexF64,space(T,3)))*T*U'
end
gauged_pairs=Tuple[]; gauged_caps=Any[]
angles=Dict(:A=>0.27,:B=>-0.4,:C=>0.83)
for (name,x,y) in FermionicPEPS._REGIONAL_SEAMS
    X,Y=getproperty(seams,name)
    UX=phase_gauge(space(X,1),angles[x]); UY=phase_gauge(space(Y,1),angles[y])
    push!(gauged_pairs,(gauge_rail(X,UX),gauge_rail(Y,UY)))
    cap=getproperty(caps,name)
    newcap=(UX⊗UY)*cap
    push!(gauged_caps,newcap)
end
gseams=NamedTuple{(:AB,:AC,:BC)}(Tuple(gauged_pairs))
gcaps=NamedTuple{(:AB,:AC,:BC)}(Tuple(gauged_caps))
gauged=finite_occupation_sixr(gseams,p,gcaps;depth=1)
@show scaled_error(six,gauged)
@test scaled_error(six,gauged)<1e-10
