# Isolated v8 Gauss-Newton loop with exact vector derivative actions.
# The mathematical equations/trust/acceptance gates are unchanged. The v8
# small-chi control is inherited evidence, NOT an exact-source v9 control.
# An actual chi16 derivative control AND complete trust-step replay are
# mandatory. This file never overrides or changes the live v8 solver.
include("replay_mixed_vector_step.jl")

function m9_control()
    rng=MersenneTwister(32301);J=randn(rng,20,8);f=randn(rng,20)
    model=(;basis=Matrix{Float64}(I,8,8),H=J,rhs=f,svd=svd(J));rows=[]
    for radius in (0.001,0.2,100.0)
        step=bk_step(model,radius);M=J'*J;g=J'*f
        reference=-(M\g);mu=0.0
        if norm(reference)>radius
            lo,hi=0.0,1.0
            while norm((M+hi*I)\g)>radius;hi*=10;end
            for _ in 1:100
                mid=(lo+hi)/2
                if norm((M+mid*I)\g)>radius;lo=mid;else;hi=mid;end
            end
            mu=hi;reference=-((M+mu*I)\g)
        end
        err=norm(step.delta-reference)/norm(reference)
        kkt=norm(J'*(f+J*step.delta)+mu*step.delta)/norm(g)
        max(err,kkt)<1e-8 || error("rectangular trust control failed")
        push!(rows,Dict("radius"=>radius,"relative_step_error"=>err,"KKT_error"=>kkt))
    end
    Dict("passed"=>true,"rows"=>rows)
end

function m9_roots(audit,R,L)
    rows=[]
    for (label,P,r,l) in (("AC",audit.env.ac,R.AC[1],L.AC[1]),("C",audit.env.c,R.C[1],L.C[1]))
        white=bv_white(P.H,P.N);values=eigvals(white.K)
        z=dot(l.data,P.H*r.data)/dot(l.data,P.N*r.data)
        push!(rows,Dict("center"=>label,"modulus_rank"=>count(abs.(values).>abs(z)*(1+1e-8))+1,
            "eigenvalue_relative_error"=>minimum(abs.(values.-z))/abs(z)))
    end
    rows
end

function m9_run(source,out,controls,replaypath,smallpath,publicpath;maxiter=48,tol=1e-11)
    haskey(ENV,"SLURM_JOB_ID") || error("submit via Slurm")
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    hashes=Dict{String,String}()
    for path in controls
        check=TOML.parsefile(path)
        check["complete"] && check["passed"] && check["nonstationary"] || error("derivative control not passed")
        check["independent_left_right"] && !check["direction_constraint"] || error("control restricts independent sides")
        for (p,h) in check["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("stale derivative control: $p")
            hashes[p]=h
        end
        hashes[path]=bytes2hex(sha256(read(path)))
    end
    paths=joinpath.(source,["boundary_1.jls","boundary_3.jls","independent_pair.jls"])
    for p in paths
        get(hashes,p,"")==bytes2hex(sha256(read(p))) || error("control does not cover input: $p")
    end
    replay=TOML.parsefile(replaypath)
    replay["complete"] && replay["passed"] && replay["chi"]==16 || error("actual chi16 trust replay required")
    for (p,h) in replay["source_hashes"]
        bytes2hex(sha256(read(p)))==h || error("stale replay source: $p")
        haskey(hashes,p) && hashes[p]!=h && error("conflicting source controls")
        hashes[p]=h
    end
    for p in (paths...,joinpath(@__DIR__,"mixed_center_vector_response.jl"),
            joinpath(@__DIR__,"replay_mixed_vector_step.jl"),joinpath(@__DIR__,"krylov_trust.jl"))
        get(replay["source_hashes"],p,"")==bytes2hex(sha256(read(p))) || error("replay does not cover source: $p")
    end
    hashes[replaypath]=bytes2hex(sha256(read(replaypath)))
    small=TOML.parsefile(smallpath);public=TOML.parsefile(publicpath)
    small["complete"] && small["converged"] && small["final"]["outer_residual"]<tol || error("inherited v8 solve failed")
    small["final_branch_dominant"] && small["final_center_dominant"] || error("inherited v8 root check failed")
    v8path=joinpath(@__DIR__,"newton_mixed_center_v8.jl")
    small["run_sha256"]==get(hashes,v8path,"") || error("replay and small control have different original solvers")
    public["complete"] && public["passed"] && public["zero_optimization_steps"] || error("inherited export failed")
    haskey(public,"reference") || error("inherited known-state comparison missing")
    get(public["source_hashes"],smallpath,"")==bytes2hex(sha256(read(smallpath))) || error("export does not cover small control")
    for (p,checked) in ((smallpath,small),(publicpath,public))
        for (f,h) in checked["source_hashes"]
            bytes2hex(sha256(read(f)))==h || error("stale inherited control: $f")
            haskey(hashes,f) && hashes[f]!=h && error("conflicting inherited control")
            hashes[f]=h
        end
        hashes[p]=bytes2hex(sha256(read(p)))
    end
    hashes[@__FILE__]=bytes2hex(sha256(read(@__FILE__)))
    n,s=deserialize.(paths[1:2]);sum(n.chi)==16 || error("this pilot is validated only at chi16")
    pair=deserialize(paths[3]);maps=ba_pair_maps(n,s)
    R,L=deepcopy(pair.R),deepcopy(pair.L)
    meta=Dict{String,Any}("complete"=>false,"accepted_entropy"=>false,"chi"=>sum(n.chi),
        "independent_left_right"=>true,"direction_constraint"=>false,"source_hashes"=>hashes,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>ENV["SLURM_JOB_PARTITION"],
        "run_sha256"=>hashes[@__FILE__],"field"=>"raw_RMS_and_exact_mixed_white",
        "method"=>"dense_real_Gauss_Newton_trust","derivative_controls"=>controls,
        "initial_perturbation"=>0.0,"tolerance"=>tol,"trust_control"=>m9_control(),
        "response_backend"=>"exact_vector_actions_v9","trust_step_replay"=>replaypath,
        "inherited_v8_small_control"=>smallpath,"inherited_v8_export_control"=>publicpath,
        "small_chi_control_scope"=>"inherited original v8 equations and trust loop; not exact-source v9")
    bv_write(joinpath(out,"input.toml"),meta)
    rows=[];best=nothing;radius=0.01;refs=nothing;reason="maxiter"
    for iteration in 0:maxiter
        R,pr=mv_reproject(R,maps.right);L,pl=mv_reproject(L,maps.left)
        audit=bv_audit(R,L,pair.transfer)
        if best===nothing || audit.residual<best.residual
            best=(;R=deepcopy(R),L=deepcopy(L),residual=audit.residual,iteration)
            serialize(joinpath(out,"best_independent_pair.jls"),(;R,L,transfer=pair.transfer))
        end
        row=merge(audit.report,Dict("iteration"=>iteration,"radius"=>radius,
            "reprojection_right"=>pr,"reprojection_left"=>pl));push!(rows,row)
        if iteration==1
            err=abs(audit.residual/replay["original"]["outer_residual"]-1)
            row["first_update_replay_relative_error"]=err
            err<1e-6 || error("solver loop disagrees with validated first-step replay")
        end
        bv_write(joinpath(out,"progress.toml"),merge(meta,Dict("rows"=>rows)))
        println("Mixed-center iteration=",iteration," original residual=",audit.residual);flush(stdout)
        if audit.residual<tol;reason="outer_residual_passed";break;end
        iteration==maxiter && break
        try
            chart=mg_chart(R,L,pair.transfer;refs);refs=chart.base.refs
            sym=at_tangent(R,L,maps);Q=sym.Q;k=size(Q,2)
            row["symmetry"]=sym.report;row["field_norm"]=norm(chart.field)
            row["lift_checks"]=chart.base.lift_checks
            J=zeros(length(chart.field),k);maxresponse=0.0;started=time()
            for j in 1:k
                response=chart.action(Q[:,j]);J[:,j]=response.field
                maxresponse=max(maxresponse,response.cap_response_error)
                j%32==0 && (println("  mixed Jacobian ",j,"/",k);flush(stdout))
            end
            row["Jacobian_seconds"]=time()-started
            maxresponse<1e-8 || error("cap response failed: $maxresponse")
            S=svd(J);row["Jacobian"]=Dict("columns"=>k,"singular_min"=>minimum(S.S),
                "singular_max"=>maximum(S.S),"cap_response_error"=>maxresponse,
                "tensor_rank_truncated"=>false,"gradient_norm"=>norm(J'*chart.field))
            model=(;basis=Matrix{Float64}(I,k,k),H=J,rhs=chart.field,svd=S)
            trials=[];row["trials"]=trials;accepted=nothing
            for attempt in 1:12
                step=bk_step(model,radius);dx=step.delta
                predicted=(norm(chart.field)^2-step.predicted_residual^2)/2
                trial=Dict{String,Any}("radius"=>radius,"predicted_gain"=>predicted,
                    "step_norm"=>norm(dx),"regularization"=>step.eta);push!(trials,trial)
                if predicted>1e-14norm(chart.field)^2
                    try
                        candidate=chart.evaluate(Q*dx)
                        rr,cr=mv_reproject(candidate.R,maps.right)
                        ll,cl=mv_reproject(candidate.L,maps.left)
                        fresh=mw_cache(rr.AL[1],ll.AL[1],pair.transfer;responses=false,refs=candidate.refs)
                        actual=(norm(chart.field)^2-norm(fresh.field)^2)/2
                        ratio=actual/predicted
                        trial["actual_gain"]=actual;trial["gain_ratio"]=ratio
                        trial["field_norm_after"]=norm(fresh.field)
                        trial["right_reprojection"]=cr;trial["left_reprojection"]=cl
                        if actual>0 && ratio>0.1
                            accepted=(rr,ll,fresh.refs)
                            if ratio<0.25;radius/=4;elseif ratio>0.75 && norm(dx)>0.9radius;radius=min(0.1,2radius);end
                            break
                        end
                    catch err
                        err isa ErrorException || rethrow()
                        trial["error"]=sprint(showerror,err)
                    end
                end
                radius/=4
            end
            accepted===nothing && (reason="mixed_trust_step_failed";break)
            R,L,refs=accepted
        catch err
            err isa ErrorException || rethrow()
            row["error"]=sprint(showerror,err);reason="mixed_response_failed";break
        end
    end
    R,L=best.R,best.L;final=bv_audit(R,L,pair.transfer)
    branch=mw_cache(R.AL[1],L.AL[1],pair.transfer;responses=false)
    dominant=all(v["modulus_rank"]==1 for v in values(branch.reports))
    roots=m9_roots(final,R,L);center=all(v["modulus_rank"]==1 && v["eigenvalue_relative_error"]<1e-8 for v in roots)
    converged=final.residual<tol && dominant && center
    south=MPSKit.InfiniteMPS([maps.unbra(L.AR[1])];tol=1e-13,maxiter=1000)
    for (name,old,state) in (("boundary_1.jls",n,R),("boundary_3.jls",s,south))
        b=save_boundary(joinpath(out,name),old,state,old.transfer,best.iteration)
        serialize(joinpath(out,name),merge(b,(;source=:mixed_center_Gauss_Newton_vector_v9,
            bivumps_converged=converged,bivumps_residual=final.residual)))
    end
    serialize(joinpath(out,"independent_pair.jls"),(;R,L,transfer=pair.transfer))
    all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
    report=merge(meta,Dict("complete"=>true,"converged"=>converged,"stop_reason"=>reason,
        "rows"=>rows,"final"=>final.report,"exported_best_iteration"=>best.iteration,
        "final_branch_dominant"=>dominant,"final_center_dominant"=>center,"final_center_roots"=>roots,
        "xi_pair"=>bv_xi(R,L),"xi_mps_right"=>bv_xi(R,R),"xi_mps_left"=>bv_xi(L,L)))
    bv_write(joinpath(out,"report.toml"),report)
    println("Mixed-center finished residual=",final.residual," converged=",converged," reason=",reason);flush(stdout)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    m9_run(ARGS[1],ARGS[2],split(ARGS[3],','),ARGS[4],ARGS[5],ARGS[6];
        maxiter=length(ARGS)>6 ? parse(Int,ARGS[7]) : 48)
end
