# Test min-max center-block steps at the actual corrected chi16 pair.
# Raw RMS blocks upper-bound original raw residuals; white blocks match
# exactly in norm. Always evaluate the original audit after retraction.
include("curvature_vector_trial.jl")
include("minimax_block_trust.jl")

function mm_field_check(cache,audit)
    expected=[audit.report[p][q] for p in ("AC","C") for q in ("right","left","right_white","left_white")]
    got=norm.(cache.fields);white=[3,4,7,8];raw=[1,2,5,6]
    err=norm(got[white]-expected[white])/norm(expected[white])
    err<1e-6 || error("white block norms disagree with original audit")
    all(got[i]>=expected[i]*(1-1e-7) for i in raw) || error("RMS raw bound violated")
    audit.residual<=maximum(got)*(1+1e-6)+1e-12 || error("original residual not controlled by block maximum")
    Dict("white_relative_error"=>err,"block_norms"=>got,"original_norms"=>expected,
        "block_maximum"=>maximum(got),"original_maximum"=>audit.residual)
end

function mm_trial(crossdir,candidateaudit,template,controlpath,out)
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    haskey(ENV,"SLURM_JOB_ID") || error("submit via Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    crosspath=joinpath(crossdir,"report.toml");cross=TOML.parsefile(crosspath)
    check=TOML.parsefile(candidateaudit);control=TOML.parsefile(controlpath)
    hashes=Dict{String,String}()
    for report in (cross,check,control)
        report["complete"] && report["passed"] || error("prerequisite incomplete")
        for (p,h) in report["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("stale source: $p")
            haskey(hashes,p) && hashes[p]!=h && error("conflicting sources")
            hashes[p]=h
        end
    end
    check["chi"]==16 && check["diagnostic_only"] && !check["accepted_entropy"] || error("wrong candidate")
    control["analytic_control_only"] || error("wrong minimax control")
    helper=joinpath(@__DIR__,"minimax_block_trust.jl")
    get(control["source_hashes"],helper,"")==bytes2hex(sha256(read(helper))) || error("minimax helper not covered")
    pairpath=check["candidate_path"];linpath=joinpath(crossdir,"cross_linearization.jls")
    get(cross["source_hashes"],pairpath,"")==bytes2hex(sha256(read(pairpath))) || error("wrong linearization input")
    bytes2hex(sha256(read(linpath)))==cross["linearization_sha256"] || error("changed linearization")
    lin=deserialize(linpath);lin.source_pair_sha256==hashes[pairpath] || error("wrong saved pair")
    paths=joinpath.(template,["boundary_1.jls","boundary_3.jls"])
    for p in paths
        get(hashes,p,"")==bytes2hex(sha256(read(p))) || error("unchecked template")
    end
    for p in (crosspath,candidateaudit,controlpath,linpath,@__FILE__);hashes[p]=bytes2hex(sha256(read(p)));end
    record=Dict{String,Any}("complete"=>false,"passed"=>false,"diagnostic_only"=>true,
        "accepted_entropy"=>false,"accepted_curve_point"=>false,"chi"=>16,
        "committed_optimization_steps"=>0,"source_hashes"=>hashes,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>"preempt",
        "trial_objective"=>"maximum center-block norm; raw RMS upper bounds original raw residual")
    reportpath=joinpath(out,"report.toml");bv_write(reportpath,record)
    try
        pair=deserialize(pairpath);n,s=deserialize.(paths);maps=ba_pair_maps(n,s)
        chart=mg_chart(pair.R,pair.L,pair.transfer);F=chart.field;J,Q=lin.J,lin.Q
        norm(F-lin.F)/norm(F)<1e-8 || error("changed base field")
        initial=bv_audit(pair.R,pair.L,pair.transfer)
        abs(initial.residual/check["original"]["outer_residual"]-1)<1e-6 || error("changed original residual")
        record["initial_original"]=initial.report;record["initial_blocks"]=mm_field_check(chart.base,initial)
        lengths=length.(chart.base.fields);half=sum(lengths)
        2half==length(F) || error("wrong real field layout")
        indices=Vector{Int}[];offset=0
        for width in lengths
            push!(indices,vcat(collect(offset+1:offset+width),collect(half+offset+1:half+offset+width)))
            offset+=width
        end
        blocks=[J[i,:] for i in indices];rhs=[F[i] for i in indices]
        norm(norm.(rhs)-norm.(chart.base.fields))/norm(F)<1e-12 || error("incorrect block extraction")
        initialmax=maximum(norm,rhs);k=size(J,2)
        GN=(;basis=Matrix{Float64}(I,k,k),H=J,rhs=F,svd=svd(J))
        trials=[];record["trials"]=trials
        for radius in (0.001,0.01,0.04),method in ("original_GN","block_minimax")
            row=Dict{String,Any}("radius"=>radius,"method"=>method,"valid"=>false);push!(trials,row)
            try
                if method=="original_GN"
                    delta=bk_step(GN,radius).delta
                else
                    function progress(rows)
                        if length(rows)%5==1
                            row["dual_solver_progress"]=Dict("rows"=>rows)
                            bv_write(reportpath,record)
                            println("Minimax radius=",radius," dual iteration=",rows[end]["iteration"]," gap=",rows[end]["relative_duality_gap"]);flush(stdout)
                        end
                    end
                    solution=mm_solve(blocks,rhs,radius;callback=progress)
                    row["dual_solver"]=solution.report
                    solution.report["passed"] || error("minimax linear subproblem not certified")
                    delta=solution.delta
                end
                predictedmax=maximum(norm(rhs[i]+blocks[i]*delta) for i in eachindex(rhs))
                predicted=(initialmax^2-predictedmax^2)/2
                candidate=chart.evaluate(Q*delta)
                rr,cr=mv_reproject(candidate.R,maps.right);ll,cl=mv_reproject(candidate.L,maps.left)
                fresh=mw_cache(rr.AL[1],ll.AL[1],pair.transfer;responses=false,refs=candidate.refs)
                all(r["modulus_rank"]==1 for r in values(fresh.reports)) || error("trial left dominant branch")
                audit=bv_audit(rr,ll,pair.transfer);checkfields=mm_field_check(fresh,audit)
                actual=(initialmax^2-checkfields["block_maximum"]^2)/2
                row["valid"]=true;row["minimax_predicted_gain"]=predicted;row["minimax_actual_gain"]=actual
                row["minimax_gain_ratio"]=actual/predicted;row["step_norm"]=norm(delta)
                row["original_residual"]=audit.residual;row["field_norm"]=norm(fresh.field)
                row["old_sum_actual_gain"]=(dot(F,F)-dot(fresh.field,fresh.field))/2
                row["original_max_improved"]=audit.residual<initial.residual
                row["passes_minimax_merit_rule"]=predicted>0 && actual>0 && actual/predicted>0.1
                row["blocks"]=checkfields;row["original"]=audit.report
                row["right_reprojection"]=cr;row["left_reprojection"]=cl
                path=joinpath(out,method*"_r"*string(radius)*".jls")
                serialize(path,(;R=rr,L=ll,transfer=pair.transfer,diagnostic_only=true,accepted_entropy=false))
                row["candidate_path"]=path;row["candidate_sha256"]=bytes2hex(sha256(read(path)))
                println(method," radius=",radius," original=",audit.residual," max-objective gain ratio=",actual/predicted);flush(stdout)
            catch err
                err isa ErrorException || rethrow()
                row["error"]=sprint(showerror,err)
            end
            bv_write(reportpath,record)
        end
        record["passed"]=true # Diagnostic completion; inspect each trial/certificate.
    catch err
        record["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        record["complete"]=true;bv_write(reportpath,record)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    mm_trial(ARGS...)
end
