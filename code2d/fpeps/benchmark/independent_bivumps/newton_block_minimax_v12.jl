# Independent chi16 max-block trust loop. Same center equations and physical
# gates; the step objective is explicitly different from sum-of-squares GN.
include("minimax_vector_trial.jl")

function m12_roots(audit,R,L)
    rows=[]
    for (label,P,r,l) in (("AC",audit.env.ac,R.AC[1],L.AC[1]),("C",audit.env.c,R.C[1],L.C[1]))
        white=bv_white(P.H,P.N);values=eigvals(white.K)
        z=dot(l.data,P.H*r.data)/dot(l.data,P.N*r.data)
        push!(rows,Dict("center"=>label,"modulus_rank"=>count(abs.(values).>abs(z)*(1+1e-8))+1,
            "eigenvalue_relative_error"=>minimum(abs.(values.-z))/abs(z)))
    end
    rows
end

function m12_run(trialdir,candidateaudit,template,out;maxiter=24,tol=1e-11)
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    haskey(ENV,"SLURM_JOB_ID") || error("submit via Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    trialpath=joinpath(trialdir,"report.toml");trial=TOML.parsefile(trialpath)
    check=TOML.parsefile(candidateaudit);hashes=Dict{String,String}()
    for report in (trial,check)
        report["complete"] && report["passed"] && report["chi"]==16 || error("chi16 prerequisite incomplete")
        report["diagnostic_only"] && !report["accepted_entropy"] || error("wrong prerequisite scope")
        for (p,h) in report["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("stale prerequisite: $p")
            haskey(hashes,p) && hashes[p]!=h && error("conflicting prerequisites")
            hashes[p]=h
        end
    end
    reference=only(filter(r->r["method"]=="block_minimax" && r["radius"]==0.01,trial["trials"]))
    reference["valid"] && reference["passes_minimax_merit_rule"] && reference["original_max_improved"] || error("unsubstantiated initial update")
    reference["dual_solver"]["passed"] && reference["dual_solver"]["tolerance"]<=1e-6 || error("uncertified initial update")
    refpath=reference["candidate_path"]
    bytes2hex(sha256(read(refpath)))==reference["candidate_sha256"] || error("changed verified candidate")
    pairpath=check["candidate_path"];paths=joinpath.(template,["boundary_1.jls","boundary_3.jls"])
    for p in (pairpath,paths...,joinpath(@__DIR__,"minimax_vector_trial.jl"),joinpath(@__DIR__,"minimax_block_trust.jl"))
        get(trial["source_hashes"],p,"")==bytes2hex(sha256(read(p))) || error("trial does not cover input/helper: $p")
    end
    for p in (trialpath,candidateaudit,refpath,@__FILE__);hashes[p]=bytes2hex(sha256(read(p)));end
    pair=deserialize(pairpath);n,s=deserialize.(paths);sum(n.chi)==16 || error("chi16 required")
    R,L,O=pair.R,pair.L,pair.transfer;maps=ba_pair_maps(n,s)
    rows=[];meta=Dict{String,Any}("complete"=>false,"converged"=>false,"chi"=>16,
        "method"=>"maximum_block_trust_v12","accepted_entropy"=>false,"accepted_curve_point"=>false,
        "independent_left_right"=>true,"direction_constraint"=>false,"antiunitary_constraint_retained"=>true,
        "tolerance"=>tol,"dual_gap_tolerance"=>1e-6,"step_objective"=>"maximum eight center-block norms",
        "raw_normalization"=>"smooth RMS upper bound; original audit checked separately",
        "source_hashes"=>hashes,"run_sha256"=>hashes[@__FILE__],"rows"=>rows,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>"preempt")
    bv_write(joinpath(out,"input.toml"),meta)
    best=nothing;radius=0.01;refs=nothing;reason="maxiter"
    for iteration in 0:maxiter
        audit=bv_audit(R,L,O)
        latestpath=joinpath(out,"latest_independent_pair.jls")
        serialize(latestpath*".tmp",(;R,L,transfer=O,iteration,diagnostic_only=true,accepted_entropy=false))
        mv(latestpath*".tmp",latestpath;force=true)
        meta["latest_pair_sha256"]=bytes2hex(sha256(read(latestpath)));meta["latest_pair_iteration"]=iteration
        if best===nothing || audit.residual<best.residual
            best=(;R=deepcopy(R),L=deepcopy(L),residual=audit.residual,iteration)
            bestpath=joinpath(out,"best_independent_pair.jls")
            serialize(bestpath*".tmp",(;R,L,transfer=O));mv(bestpath*".tmp",bestpath;force=true)
        end
        row=merge(audit.report,Dict("iteration"=>iteration,"radius"=>radius));push!(rows,row)
        if iteration==1
            ref=deserialize(refpath)
            errors=[norm(R.AL[1]-ref.R.AL[1])/norm(ref.R.AL[1]),norm(L.AL[1]-ref.L.AL[1])/norm(ref.L.AL[1])]
            residualerr=abs(audit.residual/reference["original_residual"]-1)
            row["first_step_AL_errors"]=errors;row["first_step_residual_relative_error"]=residualerr
            maximum(errors)<1e-7 && residualerr<1e-6 || error("minimax loop does not reproduce its verified first step")
        end
        bv_write(joinpath(out,"progress.toml"),meta)
        println("Block-minimax iteration=",iteration," original=",audit.residual);flush(stdout)
        if audit.residual<tol;reason="outer_residual_passed";break;end
        iteration==maxiter && break
        try
            chart=mg_chart(R,L,O;refs);refs=chart.base.refs
            all(v["modulus_rank"]==1 for v in values(chart.base.reports)) || error("base caps not dominant")
            row["blocks"]=mm_field_check(chart.base,audit);F=chart.field
            sym=at_tangent(R,L,maps);Q=sym.Q;k=size(Q,2);row["symmetry"]=sym.report
            J=zeros(length(F),k);maxresponse=0.0;started=time()
            for j in 1:k
                response=chart.action(Q[:,j]);J[:,j]=response.field
                maxresponse=max(maxresponse,response.cap_response_error)
                j%64==0 && (println("  minimax Jacobian ",j,"/",k);flush(stdout))
            end
            maxresponse<1e-8 || error("inaccurate cap response")
            row["Jacobian_seconds"]=time()-started;row["cap_response_error"]=maxresponse
            lengths=length.(chart.base.fields);half=sum(lengths);offset=0;indices=Vector{Int}[]
            2half==length(F) || error("wrong field layout")
            for width in lengths
                push!(indices,vcat(collect(offset+1:offset+width),collect(half+offset+1:half+offset+width)))
                offset+=width
            end
            blocks=[J[i,:] for i in indices];rhs=[F[i] for i in indices]
            initialmax=maximum(norm,rhs);accepted=nothing;trials=[];row["trials"]=trials
            for attempt in 1:8
                trialrow=Dict{String,Any}("radius"=>radius);push!(trials,trialrow)
                function progress(dualrows)
                    if length(dualrows)%5==1
                        trialrow["dual_progress"]=Dict("rows"=>dualrows)
                        bv_write(joinpath(out,"progress.toml"),meta)
                        println("  minimax dual ",dualrows[end]["iteration"]," gap=",dualrows[end]["relative_duality_gap"]);flush(stdout)
                    end
                end
                solution=mm_solve(blocks,rhs,radius;maxiter=160,callback=progress)
                trialrow["dual_solver"]=solution.report
                if !solution.report["passed"]
                    trialrow["rejection"]="uncertified_linear_subproblem";radius/=4;continue
                end
                dx=solution.delta;predictedmax=solution.report["predicted_block_maximum"]
                predicted=(initialmax^2-predictedmax^2)/2
                trialrow["predicted_gain"]=predicted;trialrow["step_norm"]=norm(dx)
                if predicted>1e-14initialmax^2
                    try
                        candidate=chart.evaluate(Q*dx)
                        rr,cr=mv_reproject(candidate.R,maps.right);ll,cl=mv_reproject(candidate.L,maps.left)
                        fresh=mw_cache(rr.AL[1],ll.AL[1],O;responses=false,refs=candidate.refs)
                        all(v["modulus_rank"]==1 for v in values(fresh.reports)) || error("trial caps not dominant")
                        final=bv_audit(rr,ll,O);blockcheck=mm_field_check(fresh,final)
                        actual=(initialmax^2-blockcheck["block_maximum"]^2)/2;ratio=actual/predicted
                        trialrow["actual_gain"]=actual;trialrow["gain_ratio"]=ratio
                        trialrow["original_residual_after"]=final.residual;trialrow["blocks_after"]=blockcheck
                        trialrow["old_sum_actual_gain"]=(dot(F,F)-dot(fresh.field,fresh.field))/2
                        trialrow["right_reprojection"]=cr;trialrow["left_reprojection"]=cl
                        if actual>0 && ratio>0.1 && final.residual<audit.residual
                            accepted=(rr,ll,fresh.refs)
                            if ratio<0.25;radius/=4;elseif ratio>0.75 && norm(dx)>0.9radius;radius=min(0.1,2radius);end
                            break
                        end
                        trialrow["rejection"]="nonlinear_merit_or_original_maximum"
                    catch err
                        err isa ErrorException || rethrow()
                        trialrow["error"]=sprint(showerror,err)
                    end
                end
                radius/=4
            end
            accepted===nothing && (reason="no_certified_reducing_step";break)
            R,L,refs=accepted
        catch err
            err isa ErrorException || rethrow()
            row["error"]=sprint(showerror,err);reason="minimax_model_failed";break
        end
        bv_write(joinpath(out,"progress.toml"),meta)
    end
    R,L=best.R,best.L;final=bv_audit(R,L,O)
    branch=mw_cache(R.AL[1],L.AL[1],O;responses=false)
    dominant=all(v["modulus_rank"]==1 for v in values(branch.reports))
    roots=m12_roots(final,R,L);center=all(v["modulus_rank"]==1 && v["eigenvalue_relative_error"]<1e-8 for v in roots)
    converged=final.residual<tol && dominant && center
    serialize(joinpath(out,"independent_pair.jls"),(;R,L,transfer=O))
    all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
    report=merge(meta,Dict("complete"=>true,"converged"=>converged,"stop_reason"=>reason,
        "final"=>final.report,"exported_best_iteration"=>best.iteration,
        "final_branch_dominant"=>dominant,"final_center_dominant"=>center,"final_center_roots"=>roots,
        "xi_pair"=>bv_xi(R,L),"xi_mps_right"=>bv_xi(R,R),"xi_mps_left"=>bv_xi(L,L)))
    bv_write(joinpath(out,"report.toml"),report)
    println("Block-minimax finished original=",final.residual," converged=",converged," reason=",reason);flush(stdout)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    m12_run(ARGS[1:4]...;maxiter=length(ARGS)>4 ? parse(Int,ARGS[5]) : 24)
end
