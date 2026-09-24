include("core.jl")

function bv_run(source,out;maxiter=60,tol=1e-9,perturb=0.0)
    haskey(ENV,"SLURM_JOB_ID") || error("submit numerical work through Slurm")
    ispath(out) && error("refusing overwrite: $out")
    mkpath(out); BLAS.set_num_threads(2)
    cp(joinpath(@__DIR__,"core.jl"),joinpath(out,"source_core.jl"))
    cp(@__FILE__,joinpath(out,"source_run.jl"))
    paths=joinpath.(source,("boundary_1.jls","boundary_3.jls"))
    n,s=deserialize.(paths);R=deepcopy(n.state)
    # This maps the independently supplied south state into a bra chart;
    # it NEVER constructs the south state from the north state.
    bra=FermionicPEPS._spatial_boundary_bra
    L=MPSKit.InfiniteMPS([bra(s.state.AR[1])];tol=1e-13,maxiter=1000)
    # The graded spatial-bra map is antiunitary, but is NOT an involution:
    # reversing arrows changes which physical strand receives the twist.
    # Construct its inverse in the exact parity-block coefficient basis.
    chart=bv_matrix(bra,s.state.AR[1])
    norm(chart'*chart-I)<1e-12 || error("spatial bra map is not antiunitary")
    unbra=y->TensorMap(conj.(chart'*y.data),space(s.state.AR[1]))
    rt=norm(unbra(bra(s.state.AR[1]))-s.state.AR[1])/norm(s.state.AR[1])
    rt<1e-12 || error("spatial bra inverse failed: $rt")
    initial_L=copy(L.AL[1])
    if perturb>0
        rng=MersenneTwister(1931);a=copy(R.AL[1]);noise=randn(rng,ComplexF64,space(a))
        R=MPSKit.InfiniteMPS([a+perturb*norm(a)/norm(noise)*noise];tol=1e-13,maxiter=1000)
    end
    norm(initial_L-L.AL[1])==0 || error("right perturbation mutated the left state")
    meta=Dict("schema"=>BIVUMPS_SCHEMA,"independent_left_right"=>true,
        "direction_constraint"=>false,"spatial_bra_roundtrip"=>rt,"initial_right_perturbation"=>perturb,
        "source_hashes"=>Dict(p=>bytes2hex(sha256(read(p))) for p in paths),
        "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))),
        "run_sha256"=>bytes2hex(sha256(read(@__FILE__))),"job_id"=>ENV["SLURM_JOB_ID"],
        "chi"=>sum(n.chi),"model_direction"=>n.direction,"accepted_entropy"=>false)
    bv_write(joinpath(out,"input.toml"),meta)
    rows=Dict{String,Any}[];reason="maxiter";audit=bv_audit(R,L,n.transfer)
    push!(rows,merge(audit.report,Dict("iteration"=>0)))
    println("initial ",rows[end]);flush(stdout)
    for it in 1:maxiter
        audit.residual<tol && (reason="outer_residual_passed";break)
        ac=bv_pencil(audit.env.ac,R.AC[1],L.AC[1])
        c=bv_pencil(audit.env.c,R.C[1],L.C[1];target=ac.z/audit.env.q)
        trials=Dict{String,Any}[];accepted=nothing
        for alpha in (1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625)
            trial=Dict{String,Any}("alpha"=>alpha);push!(trials,trial)
            try
                rn=bv_retract(R,ac.r,c.r,alpha);ln=bv_retract(L,ac.l,c.l,alpha)
                a=bv_audit(rn,ln,n.transfer);trial["outer_residual"]=a.residual
                if a.residual<audit.residual*(1-1e-4*alpha) || a.residual<tol
                    accepted=(rn,ln,a,alpha);break
                end
            catch e
                trial["error"]=sprint(showerror,e)
            end
        end
        if accepted===nothing
            reason="no_decreasing_paired_step"
            push!(rows,Dict("iteration"=>it,"accepted"=>false,"trials"=>trials,
                "AC_proposal"=>ac.report,"C_proposal"=>c.report));break
        end
        R,L,audit,alpha=accepted
        row=merge(audit.report,Dict("iteration"=>it,"accepted"=>true,"alpha"=>alpha,
            "trials"=>trials,"AC_proposal"=>ac.report,"C_proposal"=>c.report))
        push!(rows,row);println("iteration=",it," residual=",audit.residual," alpha=",alpha);flush(stdout)
        serialize(joinpath(out,"iterate.jls"),(;R,L,transfer=n.transfer))
        bv_write(joinpath(out,"progress.toml"),merge(meta,Dict("complete"=>false,"rows"=>rows)))
    end
    audit=bv_audit(R,L,n.transfer)
    S=MPSKit.InfiniteMPS([unbra(L.AR[1])];tol=1e-13,maxiter=1000)
    # Save native chart boundaries for unchanged bare/direct contraction.
    for (name,old,state) in (("boundary_1.jls",n,R),("boundary_3.jls",s,S))
        b=save_boundary(joinpath(out,name),old,state,old.transfer,length(rows)-1)
        serialize(joinpath(out,name),merge(b,(;source=:independent_bivumps_v1,
            bivumps_converged=audit.residual<tol,bivumps_residual=audit.residual)))
    end
    serialize(joinpath(out,"independent_pair.jls"),(;R,L,transfer=n.transfer))
    report=merge(meta,Dict("complete"=>true,"converged"=>audit.residual<tol,
        "tolerance"=>tol,"stop_reason"=>reason,"rows"=>rows,"final"=>audit.report,
        "xi_pair"=>bv_xi(R,L),"xi_mps_right"=>bv_xi(R,R),"xi_mps_left"=>bv_xi(L,L)))
    bv_write(joinpath(out,"report.toml"),report)
    println("finished residual=",audit.residual," reason=",reason);flush(stdout)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    bv_run(ARGS[1],ARGS[2];maxiter=length(ARGS)>2 ? parse(Int,ARGS[3]) : 60,
        perturb=length(ARGS)>3 ? parse(Float64,ARGS[4]) : 0.0,
        tol=length(ARGS)>4 ? parse(Float64,ARGS[5]) : 1e-9)
end
