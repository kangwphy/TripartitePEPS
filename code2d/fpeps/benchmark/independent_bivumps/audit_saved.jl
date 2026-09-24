include("core.jl")
for source in ARGS
    out=joinpath(source,"saved_audit.toml");ispath(out) && error("refusing overwrite")
    files=joinpath.(source,("independent_pair.jls","boundary_1.jls","boundary_3.jls","report.toml"))
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in files)
    pair=deserialize(files[1]);n=deserialize(files[2]);s=deserialize(files[3]);report=TOML.parsefile(files[4])
    report["schema"]==BIVUMPS_SCHEMA || error("wrong solver schema")
    report["core_sha256"]==bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))) || error("changed equations")
    a=bv_audit(pair.R,pair.L,pair.transfer)
    norm(pair.R.AL[1]-n.state.AL[1])<1e-12 || error("native export differs")
    bra=FermionicPEPS._spatial_boundary_bra
    L=MPSKit.InfiniteMPS([bra(s.state.AR[1])];tol=1e-13,maxiter=1000)
    left_export_error=abs(1-abs(dot(pair.L,L)))
    left_export_error<1e-10 || error("south export does not represent the solved independent left state")
    exported=bv_audit(n.state,L,n.transfer)
    residual=max(a.residual,exported.residual)
    threshold=1e-9 # Original environment gate; distinct from optional 1e-12 polishing.
    record=Dict("complete"=>true,"passed"=>residual<threshold,"acceptance_tolerance"=>threshold,
        "outer_residual"=>residual,"solved_pair"=>a.report,"exported_pair"=>exported.report,
        "left_export_fidelity_error"=>left_export_error,"zero_optimization_steps"=>true,
        "requested_solver_tolerance"=>report["tolerance"],"requested_solver_tolerance_met"=>report["converged"],
        "source_hashes"=>hashes,"core_sha256"=>report["core_sha256"],
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))))
    bv_write(out,record);println(source," saved/exported residual=",residual," acceptance=",residual<threshold);flush(stdout)
end
