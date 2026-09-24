include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, Test

@testset "finite public API and physical gauge" begin
    for weight in ((.125,.125),(.21,.32))
        A=ksvc_tensor(;bond_weight=weight,bond_gauge=:balanced)
        @test finite_tensor_state(A,3,2) ≈ finite_fock_state(3,2;bond_weight=weight) atol=1e-12
    end
    peps=ksvc_ipeps(;bond_weight=(.125,.125),bond_gauge=:balanced)
    result=measure_finite_fpeps_stilde(peps;L=2)
    expected=occupation_sectors(finite_fock_state(2,2;bond_weight=(.125,.125)),[:A,:B,:C,:C])
    @test result.stilde ≈ expected.stilde atol=1e-12
    @test result.regulator==:virtual_vacuum_obc && result.junctions==1
    @test !result.diagnostics.size_converged && !result.diagnostics.chi_converged
    @test result.diagnostics.norm_phase_error<1e-12
    @test_throws ArgumentError measure_finite_fpeps_stilde(peps;L=3)
    @test_throws ArgumentError measure_finite_fpeps_stilde(peps;L=2,chi=0)
    @test_throws ArgumentError measure_finite_fpeps_stilde(peps;L=2,phase_tolerance=Inf)
    @test_throws ArgumentError ksvc_tensor(;bond_gauge=:unknown)
    # This particular original KSVC closure is exactly the zero physical ket.
    @test_throws ErrorException measure_finite_fpeps_stilde(;L=2)
end
