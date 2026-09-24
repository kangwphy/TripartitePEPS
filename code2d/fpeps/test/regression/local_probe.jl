using Test, LinearAlgebra
include("../../src/FermionicPEPS.jl")
using .FermionicPEPS
using TensorKit

@show ksvc_projector()
@show ksvc_ipeps()
for bond_weight in ((1,1),(0.7,0.8)),(nx,ny) in ((1,1),(2,1),(1,2),(3,1),(2,2),(3,2))
    A = ksvc_tensor(;bond_weight)
    reference = finite_fock_state(nx,ny;bond_weight)
    candidate = finite_tensor_state(A,nx,ny)
    @show bond_weight nx ny norm(reference) norm(reference-candidate)
    flush(stdout)
    @test candidate ≈ reference atol=1e-12
    if iszero(norm(reference))
        println("This virtual vacuum closure has exactly zero norm; skip normalized observables.")
        continue
    end
    gamma = majorana_covariance(reference)
    @test gamma*gamma ≈ -I atol=1e-12
    if nx*ny >= 3
        regions = [(:A,:B,:C)[mod1(i,3)] for i in 1:nx*ny]
        sectors = occupation_sectors(reference,regions)
        gaussian = gaussian_sectors(gamma,regions)
        @show sectors gaussian
        @test sectors.S2 ≈ gaussian.S2 atol=1e-12
        @test sectors.Z4/sectors.Z1^4 ≈ gaussian.Z4 atol=1e-12
        @test sectors.stilde ≈ gaussian.stilde atol=1e-12
    end
end
