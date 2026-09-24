include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, LinearAlgebra, Random, Test

# A closed triangle of three even regional ket maps, with independently
# constructed double-layer rails. This checks the physical parity pull-through,
# including the central caps, against occupation sewing on the full ket state.
function first_even(V)
    e=zeros(ComplexF64,V←one(V))
    for (_,block) in blocks(e)
        block[1]=1
    end
    e
end

function region_rails(q)
    E=ncon([q,q],[[1,-1,-3],[1,-2,-4]],[false,true])
    U,S,V=svd_compact(permute(E,((1,2),(3,4))))
    f=permute(U,((3,1,2),()))
    g=permute(S*V,((1,2,3),()))
    ef,eg=first_even(space(f,1)),first_even(space(g,1))
    F=f⊗ef'; G=g⊗eg'
    rebuilt=ncon([F,G,ef,eg],[[1,-1,-2,2],[1,-3,-4,3],[2],[3]])
    @test rebuilt ≈ E atol=1e-12
    (;F,G,ef,eg)
end

function check_maps(qA,qB,qC,expected)
    A,B,C=region_rails.((qA,qB,qC))
    seams=(AB=(A.F,B.F),AC=(A.G,C.F),BC=(B.G,C.G))
    caps=(AB=A.ef⊗B.ef,AC=A.eg⊗C.ef,BC=B.eg⊗C.eg)
    actual=finite_lmps_sectors(seams,caps;depth=1)
    for name in (:Z1,:Z2A,:Z2B,:Z2C,:Z4)
        result=getproperty(actual,name)
        value=result.value*exp(result.logscale)
        reference=getproperty(expected,name)
        @show name value reference value/reference
        @test value ≈ reference rtol=1e-10 atol=1e-12
    end
    @test lmps_sector_entropies(actual).stilde ≈ expected.stilde atol=1e-10
    # Mix degenerate retained states within each parity sector, rather than
    # testing only scalar even/odd phases. Opposite regional ends use dual maps.
    function gauge(V)
        U=id(ComplexF64,V)
        rng=MersenneTwister(741)
        for (_,block) in blocks(U)
            block.=Matrix(qr(randn(rng,ComplexF64,size(block))).Q)
        end
        U
    end
    uA,uB,uC=gauge(space(A.F,1)),gauge(space(B.F,1)),gauge(space(C.F,1))
    transforms=(AB=(uA,uB),AC=(transpose(uA'),uC),BC=(transpose(uB'),transpose(uC')))
    change(T,U)=(U⊗id(ComplexF64,space(T,2))⊗id(ComplexF64,space(T,3)))*T*U'
    newseams=NamedTuple{keys(seams)}(Tuple(
        (change(pair[1],us[1]),change(pair[2],us[2])) for (pair,us) in zip(values(seams),values(transforms))))
    newcaps=NamedTuple{keys(caps)}(Tuple((us[1]⊗us[2])*cap for (us,cap) in zip(values(transforms),values(caps))))
    changed=finite_lmps_sectors(newseams,newcaps;depth=1)
    for (a,b) in zip(values(actual),values(changed))
        @test FermionicPEPS._sector_relative_error(a,b)<1e-10
    end
    @show expected.stilde lmps_sector_entropies(actual).stilde
    flush(stdout)
end

rng=MersenneTwister(891)
V=Vect[FermionParity](0=>1,1=>1)
@testset "physical triangle" begin
    qA=randn(rng,ComplexF64,V⊗V⊗V←one(V))
    qB=randn(rng,ComplexF64,V⊗V'⊗V←one(V))
    qC=randn(rng,ComplexF64,V⊗V'⊗V'←one(V))
    psi=ncon([qA,qB,qC],[[-1,1,2],[-2,1,3],[-3,2,3]])
    expected=occupation_sectors(vec(convert(Array,psi)),[:A,:B,:C])
    check_maps(qA,qB,qC,expected)
end

# Lift a six-mode state to three regional maps without a Gaussian assumption.
# Region B contains its exact coefficient tensor; A/C pass through the outer
# regional Fock legs. This is an independent complete-region-map factorization.
@testset "six-mode physical states" begin
    W=Vect[FermionParity](0=>1,1=>0)
    F=isomorphism(ComplexF64,fuse(V⊗V)←V⊗V)
    for label in (:non_gaussian,:ksvc_3x2,:deformed_ksvc_3x2)
        original=label==:non_gaussian ? randn(rng,ComplexF64,V^6←one(V)) :
            finite_tensor(ksvc_tensor(;bond_weight=label==:ksvc_3x2 ? (1,1) : (0.7,0.8)),3,2)
        original/=norm(original)
        fused=(F⊗F⊗F)*original
        PA,PB,PC=(space(fused,i) for i in 1:3)
        qA=permute(id(ComplexF64,PA),((1,2),()))⊗first_even(W)
        qB=permute(fused,((2,1,3),()))
        c=permute(id(ComplexF64,PC),((1,2),()))⊗first_even(W')
        qC=permute(c,((1,3,2),()))
        reconstructed=ncon([qA,qB,qC],[[-1,1,2],[-2,1,3],[-3,2,3]])
        @test reconstructed ≈ fused atol=1e-12
        expected=occupation_sectors(vec(convert(Array,original)),[:A,:A,:B,:B,:C,:C])
        @show label
        check_maps(qA,qB,qC,expected)
    end
end
