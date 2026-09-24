# A measured weak-subspace Hessian correction at the failed chi16 pair.
# Single-step diagnostics only; no existing solver method is replaced.
include("anchored_vector_response.jl")

# Global minimizer of a diagonal, possibly indefinite, quadratic on a ball.
# Shift by -minimum(beta) before root finding to avoid cancellation there.
function cv_trust(beta,g,radius)
    radius>0 && length(beta)==length(g) || error("invalid trust problem")
    all(isfinite,beta) && all(isfinite,g) || error("nonfinite trust problem")
    floor=max(0.0,-minimum(beta));d=beta.+floor
    zeroinds=findall(iszero,d);nz=findall(!iszero,d)
    z=zeros(length(g));z[nz]=-g[nz]./d[nz]
    hard=all(iszero,g[zeroinds]);tau=0.0
    if hard && norm(z)<=radius
        if floor>0
            z[first(zeroinds)]=sqrt(max(0.0,radius^2-dot(z,z)))
        end
    else
        coords(t)=-g./(d.+t)
        hi=max(norm(g)/radius,eps(Float64)*maximum(abs,beta),floatmin(Float64))
        while norm(coords(hi))>radius;hi*=2;isfinite(hi) || error("trust bracket overflow");end
        lo=0.0
        for _ in 1:160
            mid=(lo+hi)/2
            if norm(coords(mid))>radius;lo=mid;else;hi=mid;end
        end
        tau=hi;z=coords(tau)
    end
    lambda=floor+tau
    residual=(d.+tau).*z+g
    kkt=norm(residual)/max(norm(g),norm((d.+tau).*z),eps())
    radiuserr=max(0.0,norm(z)/radius-1)
    complementarity=lambda*abs(norm(z)-radius)/max(norm(g),lambda*radius,eps())
    minimum(d.+tau)>=0 && kkt<1e-10 && radiuserr<1e-12 && complementarity<1e-10 || error("trust KKT failed")
    gain=-dot(g,z)-dot(beta,z.^2)/2
    (;z,gain,report=Dict("multiplier"=>lambda,"shift_above_PSD_bound"=>tau,
        "hard_case"=>hard && tau==0 && floor>0,"KKT_relative_error"=>kkt,
        "radius_relative_excess"=>radiuserr,"complementarity_error"=>complementarity,
        "minimum_shifted_eigenvalue"=>minimum(d.+tau),"step_norm"=>norm(z)))
end

function cv_control()
    checks=[]
    for (name,b,g,r,reference) in (
        ("indefinite_hard",[-1.0,2.0],[0.0,1.0],1.0,[sqrt(8/9),-1/3]),
        ("flat_linear",[0.0,0.0],[1.0,0.0],1.0,[-1.0,0.0]),
        ("negative_degenerate",[-1.0,-1.0],[0.0,0.0],1.0,[1.0,0.0]),
        ("positive_interior",[1.0,2.0],[0.1,0.2],1.0,[-0.1,-0.1]))
        step=cv_trust(b,g,r);err=norm(step.z-reference)
        err<1e-12 || error("analytic trust control failed: $name")
        push!(checks,Dict("name"=>name,"step_error"=>err,"KKT"=>step.report))
    end
    H=Diagonal([0.03,0.7,2.0,8.0]);F=[0.1,-0.2,0.3,0.9]
    model=(;basis=Matrix{Float64}(I,4,4),H=Matrix(H),rhs=F,svd=svd(Matrix(H)))
    for radius in (0.01,0.3,20.0)
        step=cv_trust(diag(H).^2,H'*F,radius);ref=bk_step(model,radius)
        err=norm(step.z-ref.delta)/norm(ref.delta)
        err<1e-11 || error("positive trust control differs from original")
        push!(checks,Dict("name"=>"positive_GN","radius"=>radius,"step_error"=>err,"KKT"=>step.report))
    end
    Dict("passed"=>true,"checks"=>checks)
end

function cv_trial(geometry,snapshot,anchorpath,template,out)
    haskey(ENV,"SLURM_JOB_ID") || error("submit via Slurm")
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    input=ca_input(geometry,snapshot);hashes=input.hashes;lin=input.lin;pair=input.pair
    anchor=TOML.parsefile(anchorpath)
    anchor["complete"] && anchor["passed"] && anchor["chi"]==16 || error("anchored response not validated")
    for (p,h) in anchor["source_hashes"]
        bytes2hex(sha256(read(p)))==h || error("stale anchored source")
        haskey(hashes,p) && hashes[p]!=h && error("conflicting sources")
        hashes[p]=h
    end
    paths=joinpath.(template,["boundary_1.jls","boundary_3.jls"])
    for p in paths
        get(hashes,p,"")==bytes2hex(sha256(read(p))) || error("unchecked template")
    end
    for p in (anchorpath,@__FILE__);hashes[p]=bytes2hex(sha256(read(p)));end
    record=Dict{String,Any}("complete"=>false,"passed"=>false,"diagnostic_only"=>true,
        "accepted_entropy"=>false,"accepted_curve_point"=>false,"chi"=>16,
        "committed_optimization_steps"=>0,"source_hashes"=>hashes,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>"preempt")
    reportpath=joinpath(out,"report.toml");bv_write(reportpath,record)
    try
        record["trust_control"]=cv_control();bv_write(reportpath,record)
        chart=ca_chart(pair.R,pair.L,pair.transfer);original=mg_chart(pair.R,pair.L,pair.transfer)
        norm(chart.base.field-lin.F)/norm(lin.F)<1e-8 || error("different base field")
        n,s=deserialize.(paths);maps=ba_pair_maps(n,s)
        audit=bv_audit(pair.R,pair.L,pair.transfer)
        record["initial_original"]=audit.report
        F=lin.F;S=svd(lin.J*lin.Q);k=length(S.S);m=6;weak=k-m+1:k
        V=lin.Q*S.V[:,weak];g=S.S.*(S.U'*F)
        record["weak_dimension"]=m;record["weak_GN_eigenvalues"]=S.S[weak].^2
        record["weak_gradient"]=g[weak]
        stencils=[];matrices=Matrix{Float64}[];record["Hessian_stencils"]=stencils
        for h in (1e-3,3e-4)
            H=zeros(m,m);maxresponse=0.0;started=time()
            for i in 1:m
                plus=chart.point(h*V[:,i]);minus=chart.point(-h*V[:,i])
                all(r["modulus_rank"]==1 for p in (plus,minus) for r in values(p.cache.reports)) || error("curvature stencil left dominant branch")
                for j in 1:m
                    ap=plus.action(V[:,j]);am=minus.action(V[:,j])
                    maxresponse=max(maxresponse,ap.cap_response_error,am.cap_response_error)
                    H[j,i]=(dot(plus.field,ap.field)-dot(minus.field,am.field))/(2h)
                end
                println("Weak Hessian h=",h," column ",i,"/",m);flush(stdout)
            end
            asym=norm(H-H')/norm(H)
            maxresponse<1e-8 && asym<5e-3 || error("inaccurate weak Hessian: antisymmetry=$asym")
            push!(matrices,H)
            push!(stencils,Dict("h"=>h,"seconds"=>time()-started,"antisymmetry_relative"=>asym,
                "cap_response_error"=>maxresponse,"raw_rows"=>[collect(r) for r in eachrow(H)],
                "symmetric_eigenvalues"=>eigvals(Symmetric((H+H')/2))))
            bv_write(reportpath,record)
        end
        disagreement=norm(matrices[1]-matrices[2])/norm(matrices[2])
        record["Hessian_step_relative_difference"]=disagreement
        disagreement<5e-3 || error("weak Hessian lacks step convergence: $disagreement")
        H=matrices[2];E=eigen(Symmetric((H+H')/2))
        beta=vcat(S.S[1:k-m].^2,E.values);gamma=vcat(g[1:k-m],E.vectors'*g[weak])
        record["model"]=Dict("weak_eigenvalues"=>E.values,"full_Newton"=>false,
            "strong_weak_Hessian_cross_terms_included"=>false,"antiunitary_constraint_retained"=>true,
            "physical_tensor_rank_truncated"=>false)
        trials=[];record["trials"]=trials
        model=(;basis=Matrix{Float64}(I,k,k),H=lin.J*lin.Q,rhs=F,svd=S)
        for radius in (0.01,0.02,0.04,0.08),method in ("original_GN","weak_curvature")
            row=Dict{String,Any}("radius"=>radius,"method"=>method,"valid"=>false)
            push!(trials,row)
            try
                if method=="original_GN"
                    step=bk_step(model,radius);delta=step.delta
                    predicted=(dot(F,F)-step.predicted_residual^2)/2
                else
                    step=cv_trust(beta,gamma,radius);z=copy(step.z)
                    z[weak]=E.vectors*z[weak];delta=S.V*z;predicted=step.gain
                    row["trust_KKT"]=step.report
                end
                candidate=original.evaluate(lin.Q*delta)
                rr,rrepair=mv_reproject(candidate.R,maps.right);ll,lrepair=mv_reproject(candidate.L,maps.left)
                fresh=mw_cache(rr.AL[1],ll.AL[1],pair.transfer;responses=false,refs=candidate.refs)
                all(r["modulus_rank"]==1 for r in values(fresh.reports)) || error("trial left dominant branch")
                final=bv_audit(rr,ll,pair.transfer)
                actual=(dot(F,F)-dot(fresh.field,fresh.field))/2
                row["valid"]=true;row["predicted_gain"]=predicted;row["actual_gain"]=actual
                row["gain_ratio"]=actual/predicted;row["step_norm"]=norm(delta)
                row["original_residual"]=final.residual;row["field_norm"]=norm(fresh.field)
                row["original_max_improved"]=final.residual<audit.residual
                row["passes_existing_merit_rule"]=predicted>0 && actual>0 && actual/predicted>0.1
                row["original"]=final.report;row["right_reprojection"]=rrepair;row["left_reprojection"]=lrepair
                tag=method*"_r"*string(radius)
                path=joinpath(out,tag*".jls")
                serialize(path,(;R=rr,L=ll,transfer=pair.transfer,diagnostic_only=true,accepted_entropy=false))
                row["candidate_path"]=path;row["candidate_sha256"]=bytes2hex(sha256(read(path)))
                println(method," radius=",radius," original=",final.residual," gain ratio=",actual/predicted);flush(stdout)
            catch err
                err isa ErrorException || rethrow()
                row["error"]=sprint(showerror,err)
            end
            bv_write(reportpath,record)
        end
        record["passed"]=true # Diagnostic completed, not a convergence claim.
    catch err
        record["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        record["complete"]=true;bv_write(reportpath,record)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    cv_trial(ARGS...)
end
