# Dense real Gauss-Newton pilot for the actual raw + dual-whitened equations.
# Changes no historical solver, rank or final physical/entropy acceptance gate.
include("mixed_center_response.jl")
include("krylov_trust.jl")

function m8_control()
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

function m8_reproject(state,K)
    symmetry=at_symmetry(state,K);A=state.AL[1]
    raw=(A+symmetry.apply(A))/2;raw=raw*bn_invsqrt(raw'*raw)
    candidate,canonical=bt_canonical(raw)
    change=norm(candidate.AL[1]-A)/norm(A)
    change<=1e-10 || error("symmetry repair is not roundoff-size: $change")
    candidate,Dict("coefficient_change"=>change,"canonical"=>canonical)
end

function m8_roots(audit,R,L)
    rows=[]
    for (label,P,r,l) in (("AC",audit.env.ac,R.AC[1],L.AC[1]),("C",audit.env.c,R.C[1],L.C[1]))
        white=bv_white(P.H,P.N);values=eigvals(white.K)
        z=dot(l.data,P.H*r.data)/dot(l.data,P.N*r.data)
        push!(rows,Dict("center"=>label,"modulus_rank"=>count(abs.(values).>abs(z)*(1+1e-8))+1,
            "eigenvalue_relative_error"=>minimum(abs.(values.-z))/abs(z)))
    end
    rows
end

function m8_run(source,out,controls;maxiter=12,perturb=0.0,small_control=nothing,tol=1e-11)
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
    for p in (@__FILE__,joinpath(@__DIR__,"krylov_trust.jl"))
        hashes[p]=bytes2hex(sha256(read(p)))
    end
    n,s=deserialize.(paths[1:2]);pair=deserialize(paths[3]);maps=ba_pair_maps(n,s)
    if sum(n.chi)>4
        small_control!==nothing || error("matching small-chi convergence control required")
        control=TOML.parsefile(small_control)
        control["complete"] && control["converged"] && control["final"]["outer_residual"]<tol || error("small-chi solve failed")
        control["final_branch_dominant"] && control["final_center_dominant"] || error("small-chi root check failed")
        control["run_sha256"]==hashes[@__FILE__] || error("different small-chi solver")
        for (p,h) in control["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("stale small-chi source: $p")
            hashes[p]=h
        end
        hashes[small_control]=bytes2hex(sha256(read(small_control)))
    end
    R,L=deepcopy(pair.R),deepcopy(pair.L)
    if perturb>0
        states=[]
        for (i,state,K) in ((1,R,maps.right),(2,L,maps.left))
            sym=at_symmetry(state,K)
            a=state.AL[1]+perturb*randn(MersenneTwister(32310+i),ComplexF64,space(state.AL[1]))
            a=(a+sym.apply(a))/2;a=a*bn_invsqrt(a'*a)
            candidate,_=bt_canonical(a);push!(states,candidate)
        end
        R,L=states
    end
    meta=Dict{String,Any}("complete"=>false,"accepted_entropy"=>false,"chi"=>sum(n.chi),
        "independent_left_right"=>true,"direction_constraint"=>false,"source_hashes"=>hashes,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>ENV["SLURM_JOB_PARTITION"],
        "run_sha256"=>hashes[@__FILE__],"field"=>"raw_RMS_and_exact_mixed_white",
        "method"=>"dense_real_Gauss_Newton_trust","derivative_controls"=>controls,
        "initial_perturbation"=>perturb,"tolerance"=>tol,"trust_control"=>m8_control())
    bv_write(joinpath(out,"input.toml"),meta)
    rows=[];best=nothing;radius=0.01;refs=nothing;reason="maxiter"
    for iteration in 0:maxiter
        R,pr=m8_reproject(R,maps.right);L,pl=m8_reproject(L,maps.left)
        audit=bv_audit(R,L,pair.transfer)
        if best===nothing || audit.residual<best.residual
            best=(;R=deepcopy(R),L=deepcopy(L),residual=audit.residual,iteration)
            serialize(joinpath(out,"best_independent_pair.jls"),(;R,L,transfer=pair.transfer))
        end
        row=merge(audit.report,Dict("iteration"=>iteration,"radius"=>radius,
            "reprojection_right"=>pr,"reprojection_left"=>pl));push!(rows,row)
        bv_write(joinpath(out,"progress.toml"),merge(meta,Dict("rows"=>rows)))
        println("Mixed-center iteration=",iteration," original residual=",audit.residual);flush(stdout)
        if audit.residual<tol;reason="outer_residual_passed";break;end
        iteration==maxiter && break
        try
            chart=mw_chart(R,L,pair.transfer;refs);refs=chart.base.refs
            sym=at_tangent(R,L,maps);Q=sym.Q;k=size(Q,2)
            row["symmetry"]=sym.report;row["field_norm"]=norm(chart.field)
            J=zeros(length(chart.field),k);maxresponse=0.0
            for j in 1:k
                response=chart.action(Q[:,j]);J[:,j]=response.field
                maxresponse=max(maxresponse,response.cap_response_error)
                j%32==0 && (println("  mixed Jacobian ",j,"/",k);flush(stdout))
            end
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
                        rr,cr=m8_reproject(candidate.R,maps.right)
                        ll,cl=m8_reproject(candidate.L,maps.left)
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
    roots=m8_roots(final,R,L);center=all(v["modulus_rank"]==1 && v["eigenvalue_relative_error"]<1e-8 for v in roots)
    converged=final.residual<tol && dominant && center
    south=MPSKit.InfiniteMPS([maps.unbra(L.AR[1])];tol=1e-13,maxiter=1000)
    for (name,old,state) in (("boundary_1.jls",n,R),("boundary_3.jls",s,south))
        b=save_boundary(joinpath(out,name),old,state,old.transfer,best.iteration)
        serialize(joinpath(out,name),merge(b,(;source=:mixed_center_Gauss_Newton_v8,
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
    m8_run(ARGS[1],ARGS[2],split(ARGS[3],',');maxiter=parse(Int,ARGS[4]),perturb=parse(Float64,ARGS[5]),
        small_control=length(ARGS)>5 ? ARGS[6] : nothing)
end
