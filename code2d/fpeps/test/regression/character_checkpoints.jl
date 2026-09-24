include("../../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, Test, Serialization
include("../../benchmark/character_lmps_measurement.jl")
@testset "character checkpoint identity and phase gates" begin
    W=Vect[FermionParity](0=>1,1=>0)
    rail(chi,ket,bra)=ones(ComplexF64,chi⊗ket⊗bra←chi)
    seams=(AB=(rail(W',W,W'),rail(W,W',W)),
           AC=(rail(W,W,W'),rail(W,W',W)),
           BC=(rail(W',W,W'),rail(W',W',W)))
    caps=solve_seam_caps(seams).caps
    mktempdir() do directory
        args=(;initial_depths=(1,2,4),max_depth=4,checkpoint_dir=directory)
        fresh=measure_character_lmps_adaptive(seams,caps;args...)
        @test fresh.converged && abs(fresh.stilde)<1e-12
        @test length(readdir(directory))==3
        timestamps=[stat(joinpath(directory,f)).mtime for f in readdir(directory)]
        reloaded=measure_character_lmps_adaptive(deepcopy(seams),deepcopy(caps);args...)
        @test reloaded.stilde==fresh.stilde
        @test timestamps==[stat(joinpath(directory,f)).mtime for f in readdir(directory)]
        changed=deepcopy(seams); changed.AB[1].data .*= 2
        @test_throws ErrorException measure_character_lmps_adaptive(changed,caps;args...)
        path=joinpath(directory,"depth_1.jls")
        saved=deserialize(path)
        bad=merge(saved.sectors,(Z4=merge(saved.sectors.Z4,(value=im*saved.sectors.Z4.value,)),))
        serialize(path,merge(saved,(sectors=bad,)))
        @test_throws ErrorException measure_character_lmps_adaptive(seams,caps;args...)
    end
end
