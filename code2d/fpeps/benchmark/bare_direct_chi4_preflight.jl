include("bare_direct_core.jl")

function bare_preflight(source,output)
    BLAS.set_num_threads(2);mkpath(output)
    names=("boundary_1.jls","boundary_3.jls","rotated_boundary.jls")
    files=[joinpath(source,n) for n in names]
    hashes=Dict(f=>bytes2hex(sha256(read(f))) for f in files)
    n,s,perpendicular_n=deserialize.(files)
    all(b->sum(b.chi)==4 && b.converged,(n,s,perpendicular_n)) || error("converged chi4 sources required")
    objects=bare_direct_objects(n,s)
    serialize(joinpath(output,"bare_objects.jls"),objects)
    open(joinpath(output,"bare_fixedpoints.toml"),"w") do io;TOML.print(io,objects.report);end
    println("BARE fixed points ",objects.report);flush(stdout)
    seams=bare_cardinal_seams(objects,perpendicular_n)
    serialize(joinpath(output,"bare_uncapped_geometry.jls"),(;seams,objects))
    # Independently optimize the opposite perpendicular boundary; never infer
    # this direction from the unproven south-to-rotated W relation.
    target=perpendicular_n.peps
    perpendicular_s=solve_boundary(target;chi=n.chi,direction=3,
        initial_state=n.state,tolerance=1e-12,maxiter=5000,accept_unconverged=true)
    serialize(joinpath(output,"perpendicular_south.jls"),perpendicular_s)
    println("perpendicular residual=",perpendicular_s.galerkin);flush(stdout)
    function radius(a,b)
        # A standard left mixed environment has codomain below and domain above.
        init=zeros(ComplexF64,space(b,1)←space(a,1))
        result=bare_dense_fixedpoint(x->MPSKit.transfer_left(x,a,b),init+id(ComplexF64,space(a,1)))
        abs(result.lambda)
    end
    ref=perpendicular_s.state.AL[1]
    comparisons=Dict{String,Any}[]
    for (name,H) in (("HA",objects.HA),("HB",objects.HB))
        h=MPSKit.InfiniteMPS([H];tol=1e-14,maxiter=1000)
        fidelity=radius(h.AL[1],ref)/sqrt(radius(h.AL[1],h.AL[1])*radius(ref,ref))
        push!(comparisons,Dict("name"=>name,"fidelity_per_site"=>fidelity))
    end
    report=Dict("method"=>"bare_two_and_three_column_fixedpoints",
        "complete"=>true,"chi"=>4,"source_hashes"=>hashes,
        "perpendicular_native_residual"=>perpendicular_s.galerkin,
        "perpendicular_native_converged"=>perpendicular_s.converged,
        "reconstruction"=>comparisons,"fixedpoints"=>objects.report,
        "grown_tensor_used"=>false,"endpoint_maps_used"=>false,"accepted_entropy"=>false)
    open(joinpath(output,"preflight.toml"),"w") do io;TOML.print(io,report);end
    println(report)
end
bare_preflight(ARGS...)
