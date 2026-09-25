using LMPSVUMPS, Serialization, Test
const test_source=ENV["TFIM_TEST_SOURCE"]
const expected_source=deserialize(test_source)
ENV["TFIM_SELECTED_STATE"]=test_source
ENV["TFIM_ROUTE_CHI"]=get(ENV,"TFIM_TEST_CHI","4")
ENV["TFIM_BARE_OUTPUT"]=ENV["TFIM_TEST_OUTPUT"]
include(joinpath(@__DIR__,"../production/tfim/run_bare_sixr.jl"))
@testset "physical D4 driver and portable provenance" begin
    saved=deserialize(joinpath(ENV["TFIM_TEST_OUTPUT"],"measurement.jls"))
    direct=deserialize(joinpath(ENV["TFIM_TEST_OUTPUT"],"measurement_direct.jls"))
    @test saved.row.backend==get(ENV,"TFIM_BACKEND","cpu")
    @test saved.row.source_converged==expected_source.selected_groundstate.converged
    @test !saved.row.production_release
    @test saved.row.status=="PASS_NUMERICAL_DIAGNOSTIC"
    @test !isempty(saved.row.code_sha256)
    @test isfinite(saved.row.tildeS)
    @test direct.M.north isa Array
    @test direct.orig.G isa Array
    @test length(saved.boundary_trials)==4
    @test all(length(x.trials)==3 for x in saved.boundary_trials)
    @test isfile(joinpath(ENV["TFIM_TEST_OUTPUT"],"measurement.csv"))
end
