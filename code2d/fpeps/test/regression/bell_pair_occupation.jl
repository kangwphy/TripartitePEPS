# Independent CAR and ordinary occupation-replica reference for one Bell
# pair on each of the three regional seams. This verifies the analytic
# zero junction constant used by the infinite Bell-pair LMPS fixture.
using Test, LinearAlgebra
include("../../src/FermionicPEPS.jl")
using .FermionicPEPS
@testset "fermionic Bell-pair triangle occupation reference" begin
    for (wx,wy) in ((.5,.25),(1.0,1.0))
        # Mode order A1,A2,B1,B2,C1,C2; oriented disjoint CAR pair words.
        state=Dict(UInt(0)=>1.0+0im)
        for (i,j,w) in ((0,2,wx),(1,4,wy),(3,5,wy))
            created=FermionicPEPS._apply_word(state,[(i,true),(j,true)])
            state=mergewith(+,state,Dict(mask=>w*c for (mask,c) in created))
        end
        psi=zeros(ComplexF64,64)
        for (mask,c) in state; psi[Int(mask)+1]=c; end
        normalize!(psi)
        exact=occupation_sectors(psi,[:A,:A,:B,:B,:C,:C])
        s2(w)=-log((1+w^4)/(1+w^2)^2)
        expected=[s2(wx)+s2(wy),s2(wx)+s2(wy),2s2(wy)]
        @test exact.S2 ≈ expected atol=1e-12
        @test abs(exact.stilde)<1e-12
        @test real(exact.Z4) ≈ exp(-sum(expected)) atol=1e-12
        println("weights=",(wx,wy)," CAR stilde=",exact.stilde)
    end
end
