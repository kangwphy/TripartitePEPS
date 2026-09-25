using LMPSVUMPS, Serialization, Test
const fixture_dir=mktempdir()
const fixture_A=reshape(ComplexF64[1,0],2,1,1,1,1)
const fixture_peps=pepskit_from_lmps(PEPS(fixture_A))
const fixture_path=joinpath(fixture_dir,"source.jls")
open(io->serialize(io,(selected_groundstate=(peps=fixture_peps,converged=false),config=(D=1,h=0.0,ctm_chi=1))),fixture_path,"w")
ENV["TFIM_SELECTED_STATE"]=fixture_path
ENV["TFIM_ROUTE_CHI"]="1"
ENV["TFIM_BARE_OUTPUT"]=joinpath(fixture_dir,"measurement")
include(joinpath(@__DIR__,"../production/tfim/run_bare_sixr.jl"))
@testset "backend driver and portable provenance" begin
    saved=deserialize(joinpath(ENV["TFIM_BARE_OUTPUT"],"measurement.jls"))
    direct=deserialize(joinpath(ENV["TFIM_BARE_OUTPUT"],"measurement_direct.jls"))
    @test saved.row.backend==get(ENV,"TFIM_BACKEND","cpu")
    @test !saved.row.source_converged
    @test !saved.row.production_release
    @test saved.row.status=="PASS_NUMERICAL_DIAGNOSTIC"
    @test !isempty(saved.row.code_sha256)
    @test abs(saved.row.tildeS)<1e-12
    @test direct.M.north isa Array
    @test direct.orig.G isa Array
    @test length(saved.boundary_trials)==4
    @test all(length(x.trials)==3 for x in saved.boundary_trials)
    @test isfile(joinpath(ENV["TFIM_BARE_OUTPUT"],"measurement.csv"))
end
