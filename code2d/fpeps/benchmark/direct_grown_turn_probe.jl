# Infinite VUMPS/LMPS diagnostic with explicit row-end turns. Region B uses
# its actual rotated PEPS, as in the CAR-validated finite regional geometry.
include("direct_graded_sixr_probe.jl")

function right_grown_turn(x)
    # Exact contraction of the finite-CAR row-end identity diagram:
    # (L*UR ⊗ I ⊗ I) * permute(id(domain(UR)),((1,2,3,5,6),(4,))).
    # The complete random even-tensor identity was checked in 11216355.
    # Keep the graded permutation, avoiding explicit identity-product work.
    permute(x.direct.L*x.direct.UR,((1,3,4),(2,)))
end

function prepare_grown_turn(input, output;dense_threshold=512)
mkpath(output)
north = deserialize(joinpath(input,"boundary_1.jls"))
south = deserialize(joinpath(input,"boundary_3.jls"))
bfile = joinpath(output,"rotated_boundary.jls")
east = if isfile(bfile)
    deserialize(bfile)
else
    state = solve_boundary(rotl90(north.peps); chi=north.chi,
        tolerance=1e-10, maxiter=2000)
    serialize(bfile,state)
    state
end
println("three oriented VUMPS boundaries ready"); flush(stdout)
regional = map((north,east,south)) do boundary
    result=direct_rails(boundary)
    # Retain only the returned tensors before allocating the next direction's
    # temporary eigensolver/canonicalization workspaces at larger chi.
    GC.gc()
    println("regional rails complete for direction=",boundary.direction); flush(stdout)
    result
end
a,b,c = regional
serialize(joinpath(output,"regional.jls"),regional)
Ha,Hb,Hc = map(x->x.objects.AL,regional)
println("constructing explicit grown turns"); flush(stdout)
Ta,Tb = right_grown_turn(a),right_grown_turn(b)
flip(M) = permute(M,((4,2,3),(1,)))
seams = (; AB=(Ta,flip(Hb)), AC=(flip(Ha),Hc), BC=(Tb,flip(Hc)))
for (name,(X,Y)) in pairs(seams)
    println(name," ",space(X)," | ",space(Y))
    all(i->space(X,i)==dual(space(Y,i)),(2,3)) || error("physical seam mismatch $name")
end
serialize(joinpath(output,"geometry.jls"),(;regional,seams))
# Resolve the complete even spectrum through total chi=32; a single Krylov
# seed can otherwise miss invariant sectors when testing leading isolation.
cap_result = solve_seam_caps(seams;dense_threshold)
serialize(joinpath(output,"prepared.jls"),(;regional,seams,cap_result))
(;regional,seams,cap_result)
end

function run_grown_turn(input, output;sector_builder=finite_lmps_sectors)
p=prepare_grown_turn(input,output)
seams,cap_result=p.seams,p.cap_result
result = measure_lmps_stilde(seams,cap_result.caps; depths=(8,16,32,64),
    accept_unconverged=true,sector_builder,on_sample=s->begin
        println("depth=",s.depth," stilde=",s.entropies.stilde,
                " residual=",s.endpoint_residual); flush(stdout)
    end)
serialize(joinpath(output,"result.jls"),result)
open(joinpath(output,"result.toml"),"w") do io
    TOML.print(io,Dict("stilde"=>result.stilde,
        "length_converged"=>result.converged,
        "physical_replica_benchmark_certified"=>false,
        "construction"=>"three oriented boundaries and explicit grown right turns"))
end
result
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    run_grown_turn(ARGS...)
end
