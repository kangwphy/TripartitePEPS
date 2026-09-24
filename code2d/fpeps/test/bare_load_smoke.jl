using Test
include("../benchmark/wide_bare_direct_scan_point.jl")
@testset "Relocated fermion bare backend loads" begin
    @test isdefined(@__MODULE__, :bare_direct_objects)
    @test isdefined(@__MODULE__, :wide_bare_direct_objects)
    @test isdefined(@__MODULE__, :bare_cardinal_seams)
    @test isdefined(@__MODULE__, :scan_wide_bare)
    @test isdefined(@__MODULE__, :large_bare_fixedpoint)
    @test isdefined(@__MODULE__, :character_lmps_sectors)
    @test isdefined(FermionicPEPS, :finite_occupation_sixr)
end
# Load/dependency check only: no entropy, boundary optimization or acceptance claim.
