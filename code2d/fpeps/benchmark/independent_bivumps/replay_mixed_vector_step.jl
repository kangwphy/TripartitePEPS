# Replay a measured chi16 v8 trust step with the isolated vector backend.
# No production method is overridden; a passed replay is not convergence.
include("mixed_center_vector_response.jl")
include("krylov_trust.jl")

# Same roundoff-only projection as m8_reproject in the hashed v8 source.
function mv_reproject(state,K)
    symmetry=at_symmetry(state,K);A=state.AL[1]
    raw=(A+symmetry.apply(A))/2;raw=raw*bn_invsqrt(raw'*raw)
    candidate,canonical=bt_canonical(raw)
    change=norm(candidate.AL[1]-A)/norm(A)
    change<=1e-10 || error("symmetry repair is not roundoff-size: $change")
    candidate,Dict("coefficient_change"=>change,"canonical"=>canonical)
end

function mv_replay(source,snapshot,controlpath,checkpointaudit,out)
    haskey(ENV,"SLURM_JOB_ID") || error("submit via Slurm")
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    hashes=Dict{String,String}()
    for p in (controlpath,checkpointaudit)
        report=TOML.parsefile(p)
        report["complete"] && report["passed"] && report["chi"]==16 || error("failed chi16 prerequisite")
        for (f,h) in report["source_hashes"]
            bytes2hex(sha256(read(f)))==h || error("stale prerequisite: $f")
            haskey(hashes,f) && hashes[f]!=h && error("conflicting prerequisites")
            hashes[f]=h
        end
        hashes[p]=bytes2hex(sha256(read(p)))
    end
    vectorpath=joinpath(@__DIR__,"mixed_center_vector_response.jl")
    get(hashes,vectorpath,"")==bytes2hex(sha256(read(vectorpath))) || error("unchecked vector backend")
    pairpath=joinpath(snapshot,"independent_pair.jls")
    progresspath=joinpath(snapshot,"progress.toml")
    paths=joinpath.(source,["boundary_1.jls","boundary_3.jls","independent_pair.jls"])
    for p in (paths...,pairpath,progresspath)
        get(hashes,p,"")==bytes2hex(sha256(read(p))) || error("unchecked replay input: $p")
    end
    saved=TOML.parsefile(progresspath)
    saved["initial_perturbation"]==0 && length(saved["rows"])==2 || error("expected first unperturbed update")
    before,after=saved["rows"]
    before["iteration"]==0 && after["iteration"]==1 || error("wrong reference iterations")
    length(before["trials"])==1 || error("reference did not accept its first trial")
    reftrial=only(before["trials"])
    for p in (@__FILE__,joinpath(@__DIR__,"krylov_trust.jl"))
        hashes[p]=bytes2hex(sha256(read(p)))
    end
    report=Dict{String,Any}("complete"=>false,"passed"=>false,"chi"=>16,
        "accepted_entropy"=>false,"accepted_curve_point"=>false,"diagnostic_only"=>true,
        "optimization_steps"=>1,"source_hashes"=>hashes,"job_id"=>ENV["SLURM_JOB_ID"],
        "partition"=>ENV["SLURM_JOB_PARTITION"],"source_solver_job"=>saved["job_id"],
        "independent_left_right"=>true,"direction_constraint"=>false)
    bv_write(out,report)
    try
        n,s=deserialize.(paths[1:2]);sum(n.chi)==16 || error("chi16 required")
        pair=deserialize(paths[3]);maps=ba_pair_maps(n,s)
        R,_=mv_reproject(pair.R,maps.right);L,_=mv_reproject(pair.L,maps.left)
        initial=bv_audit(R,L,pair.transfer)
        report["initial_residual_relative_error"]=abs(initial.residual/before["outer_residual"]-1)
        report["initial_residual_relative_error"]<1e-6 || error("initial state differs from reference")
        chart=mg_chart(R,L,pair.transfer);sym=at_tangent(R,L,maps);Q=sym.Q;k=size(Q,2)
        report["symmetry"]=sym.report;report["lift_checks"]=chart.base.lift_checks
        J=zeros(length(chart.field),k);maxresponse=0.0;started=time()
        for j in 1:k
            response=chart.action(Q[:,j]);J[:,j]=response.field
            maxresponse=max(maxresponse,response.cap_response_error)
            if j%32==0
                report["Jacobian_columns_complete"]=j
                bv_write(out,report)
                println("Vector replay Jacobian ",j,"/",k);flush(stdout)
            end
        end
        report["Jacobian_seconds"]=time()-started
        maxresponse<1e-8 || error("cap response failed")
        S=svd(J);model=(;basis=Matrix{Float64}(I,k,k),H=J,rhs=chart.field,svd=S)
        step=bk_step(model,before["radius"])
        predicted=(norm(chart.field)^2-step.predicted_residual^2)/2
        candidate=chart.evaluate(Q*step.delta)
        rr,_=mv_reproject(candidate.R,maps.right);ll,_=mv_reproject(candidate.L,maps.left)
        fresh=mw_cache(rr.AL[1],ll.AL[1],pair.transfer;responses=false,refs=candidate.refs)
        actual=(norm(chart.field)^2-norm(fresh.field)^2)/2
        ratio=actual/predicted
        actual>0 && ratio>0.1 || error("replayed first step rejected")
        # v8 repeats the roundoff projection at the start of the next iteration.
        rr,_=mv_reproject(rr,maps.right);ll,_=mv_reproject(ll,maps.left)
        final=bv_audit(rr,ll,pair.transfer);reference=deserialize(pairpath)
        comparisons=Dict{String,Any}()
        for (name,value,target) in (("predicted_gain",predicted,reftrial["predicted_gain"]),
                ("actual_gain",actual,reftrial["actual_gain"]),("gain_ratio",ratio,reftrial["gain_ratio"]),
                ("step_norm",norm(step.delta),reftrial["step_norm"]),
                ("outer_residual",final.residual,after["outer_residual"]),
                ("Jacobian_singular_min",minimum(S.S),before["Jacobian"]["singular_min"]),
                ("Jacobian_singular_max",maximum(S.S),before["Jacobian"]["singular_max"]))
            err=abs(value/target-1)
            comparisons[name]=Dict("value"=>value,"reference"=>target,"relative_error"=>err)
            err<1e-6 || error("full step differs in $name: $err")
        end
        report["comparisons"]=comparisons
        for (label,state,refstate) in (("right",rr,reference.R),("left",ll,reference.L))
            err=norm(state.AL[1]-refstate.AL[1])/norm(refstate.AL[1])
            report[label*"_AL_coefficient_relative_error"]=err
            err<1e-7 || error("replayed $label state differs: $err")
        end
        report["cap_response_error"]=maxresponse;report["original"]=final.report
        report["passed"]=true
    catch err
        report["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        report["complete"]=true;bv_write(out,report)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    mv_replay(ARGS...)
end
