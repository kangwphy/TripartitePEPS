using Test, Random, LinearAlgebra, TensorKit, PEPSKit, MPSKit
include(joinpath(@__DIR__,"..","..","src","FermionicPEPS.jl"))
using .FermionicPEPS

@testset "physical spatial edge to MPS bra" begin
    V=Vect[FermionParity](0=>2,1=>1)
    rng=MersenneTwister(31291)
    peps=ksvc_ipeps(;bond_weight=(.125,.25),bond_gauge=:balanced)
    for orientation in 0:3
        oriented=foldl((p,_)->rotl90(p),1:orientation;init=peps)
        for trial in 1:2
            A=randn(rng,ComplexF64,space(oriented[1]))
            B=randn(rng,ComplexF64,space(A))
            opposite=rot180(oriented)[1]
            Mn=randn(rng,ComplexF64,V⊗dual(space(A,2))⊗space(B,2)←V)
            Ms=randn(rng,ComplexF64,V⊗dual(space(opposite,2))⊗space(opposite,2)←V)
            Mb=FermionicPEPS._spatial_boundary_bra(Ms)
            @test space(Mb)==space(Mn)
            v0=randn(rng,ComplexF64,V←V)
            @test MPSKit.transfer_left(v0,Mn,Mb) ≈ PEPSKit.edge_transfer_left(v0,Mn,Ms)
            # The right environment is dual to the left under the graded cap.
            # Its matrix chart therefore carries parity on the remaining
            # (south) auxiliary leg; using an untransported cap is different.
            @test MPSKit.transfer_right(twist(v0,2),Mn,Mb) ≈
                  twist(PEPSKit.edge_transfer_right(v0,Mn,Ms),2)
            raw=FermionicPEPS._grow_boundary(Mn,A,B)
            Q=permute(PEPSKit.twistdual(raw,(2,3)),((1,2,3,4,5),(6,7,8)))
            Vl=space(Q,1)⊗space(Q,2)⊗space(Q,3); Vr=domain(Q)
            UL=isomorphism(ComplexF64,fuse(Vl)←Vl)
            UR=isomorphism(ComplexF64,fuse(Vr)←Vr)
            Ip=id(ComplexF64,space(Q,4))⊗id(ComplexF64,space(Q,5))
            D=(UL⊗Ip)*Q*UR'
            left=randn(rng,ComplexF64,V⊗dual(space(A,5))⊗space(B,5)←V)
            grown_left=PEPSKit.edge_transfer_left(left,(A,B),Mn,Ms)
            fused_left=permute(left,((1,),(4,2,3)))*UL'
            expected_left=permute(grown_left,((1,),(4,2,3)))*UR'
            @test MPSKit.transfer_left(fused_left,D,Mb) ≈ expected_left
            right=randn(rng,ComplexF64,V⊗dual(space(A,3))⊗space(B,3)←V)
            grown_right=PEPSKit.edge_transfer_right(right,(A,B),Mn,Ms)
            # The three-port physical cap has both the physical cup and the
            # auxiliary closing loop. Transport both before matrix pairing.
            right_chart(r)=twist(PEPSKit.twistdual(r,(2,3)),4)
            @test MPSKit.transfer_right(UR*right_chart(right),D,Mb) ≈
                  UL*right_chart(grown_right)
            @tensor physical_pair = left[a k bra;l]*right[l k bra;a]
            @test dot(permute(left,((1,),(4,2,3)))',right_chart(right)) ≈ physical_pair
        end
    end
end
