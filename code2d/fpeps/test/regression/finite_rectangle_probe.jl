include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, Test

@testset "unequal regions and the original critical fPEPS" begin
    for weight in ((1.,1.),(.125,.125)),cut in (1,2)
        peps=ksvc_ipeps(;bond_weight=weight)
        actual=measure_finite_fpeps_stilde(peps;L=2,width=3,cut)
        regs=[y==2 ? :C : x<=cut ? :A : :B for y in 1:2 for x in 1:3]
        expected=occupation_sectors(finite_fock_state(3,2;bond_weight=weight),regs)
        for name in (:Z1,:Z2A,:Z2B,:Z2C,:Z4)
            z=getproperty(actual.sectors,name)
            @test z.value*exp(z.logscale) ≈ getproperty(expected,name) rtol=1e-10 atol=1e-12
        end
        @test actual.stilde ≈ expected.stilde atol=1e-10
        @show weight cut actual.stilde expected.stilde
        flush(stdout)
    end
    @test_throws ArgumentError measure_finite_fpeps_stilde(;L=2,width=3,cut=3)
end
