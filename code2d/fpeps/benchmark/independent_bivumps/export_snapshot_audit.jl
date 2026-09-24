# Export an immutable independent pair as an initial guess, never as accepted data.
include("dominance_response.jl")

function es_prepare_audit(snapshot,template,out)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");mkpath(out);BLAS.set_num_threads(2)
    paths=vcat([snapshot],joinpath.(template,["boundary_1.jls","boundary_3.jls"]))
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in paths)
    for file in ("core.jl","newton_response_fast.jl","newton_metric_response.jl",
                 "branch_response.jl","dominance_response.jl","export_snapshot_audit.jl")
        p=joinpath(@__DIR__,file);hashes[p]=bytes2hex(sha256(read(p)))
        cp(p,joinpath(out,"source_"*file))
    end
    state=deserialize(snapshot);n,s=deserialize.(paths[2:3]);R,L,O=state.R,state.L,state.transfer
    norm(PEPSKit.ket(O[1])-PEPSKit.ket(n.transfer[1]))<1e-12 &&
        norm(PEPSKit.bra(O[1])-PEPSKit.bra(n.transfer[1]))<1e-12 || error("snapshot/template transfer mismatch")
    bra=FermionicPEPS._spatial_boundary_bra
    chart=bv_matrix(bra,s.state.AR[1]);norm(chart'*chart-I)<1e-12 || error("nonunitary bra chart")
    unbra=y->TensorMap(conj.(chart'*y.data),space(s.state.AR[1]))
    raw=unbra(L.AR[1]);norm(bra(raw)-L.AR[1])<1e-12*norm(L.AR[1]) || error("bra inverse failed")
    S=MPSKit.InfiniteMPS([raw];tol=1e-13,maxiter=2000)
    exportedL=MPSKit.InfiniteMPS([bra(S.AR[1])];tol=1e-13,maxiter=2000)
    fidelity=abs(dot(L,exportedL));abs(fidelity-1)<1e-9 || error("snapshot export changed left state")
    before=bd_constraints(R.AL[1],L.AL[1],O;vectors=false)
    after=bd_constraints(R.AL[1],exportedL.AL[1],O;vectors=false)
    bv_write(joinpath(out,"roundtrip_audit.toml"),Dict("complete"=>true,
        "passed"=>maximum(abs.(after.gaps-before.gaps))<1e-10,
        "seed_only"=>true,"accepted_entropy"=>false,
        "source_hashes"=>hashes,"left_fidelity_per_site"=>fidelity,
        "snapshot_gaps"=>before.gaps,"exported_gaps"=>after.gaps,
        "gap_absolute_changes"=>abs.(after.gaps-before.gaps),
        "snapshot_spectra"=>before.reports,"exported_spectra"=>after.reports))
    println("Roundtrip fidelity=",fidelity," before=",before.gaps," after=",after.gaps);flush(stdout)
    minimum(after.gaps)>1.5e-9 || error("exported snapshot lacks required dominance margin")
    maximum(abs.(after.gaps-before.gaps))<1e-10 || error("export changed modulus gaps")
    originalL=MPSKit.InfiniteMPS([bra(s.state.AR[1])];tol=1e-13,maxiter=1000)
    template_gaps=bd_constraints(n.state.AL[1],originalL.AL[1],n.transfer;vectors=false)
    audit=bv_audit(R,exportedL,O)
    for (file,old,psi) in (("boundary_1.jls",n,R),("boundary_3.jls",s,S))
        saved=save_boundary(joinpath(out,file),old,psi,old.transfer,0)
        serialize(joinpath(out,file),merge(saved,(;source=:independent_frozen_snapshot_seed,
            converged=false,bivumps_converged=false)))
    end
    all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("input changed")
    bv_write(joinpath(out,"preparation.toml"),Dict("complete"=>true,"seed_only"=>true,
        "entropy_accepted"=>false,"optimization_steps"=>0,"independent_left_right"=>true,
        "direction_constraint"=>false,"job_id"=>ENV["SLURM_JOB_ID"],"source_hashes"=>hashes,
        "left_fidelity_per_site"=>fidelity,"snapshot_gaps"=>before.gaps,
        "exported_gaps"=>after.gaps,"previous_template_gaps"=>template_gaps.gaps,
        "initial_independent_audit"=>audit.report))
    println("Snapshot seed exported: gaps=",after.gaps," prior-source gaps=",template_gaps.gaps,
        " initial outer=",audit.residual);flush(stdout)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    es_prepare_audit(ARGS...)
end
