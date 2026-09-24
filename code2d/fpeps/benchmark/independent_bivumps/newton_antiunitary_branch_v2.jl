# Independent antiunitary-constrained Newton, conditional on tangent audits.
# Full complex equations and final dominant roots remain the acceptance test.
include("antiunitary_newton_chart.jl")
include("krylov_trust.jl")
include("ellipsoid_trust.jl")

function an_roots(audit,R,L)
    rows=[]
    for (name,P,r,l) in (("AC",audit.env.ac,R.AC[1],L.AC[1]),("C",audit.env.c,R.C[1],L.C[1]))
        w=bv_white(P.H,P.N);values=eigvals(w.K)
        z=dot(l.data,P.H*r.data)/dot(l.data,P.N*r.data)
        push!(rows,Dict("center"=>name,"stationary_modulus_rank"=>count(abs.(values).>abs(z)*(1+1e-8))+1,
            "nearest_eigenvalue_relative_error"=>minimum(abs.(values.-z))/abs(z)))
    end
    rows
end

function an_run(source,out,controls;maxiter=30,tol=1e-11,perturb=0.0,radius=0.01,roundoff_control)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");mkpath(out);BLAS.set_num_threads(2)
    hashes=Dict{String,String}()
    rounding=TOML.parsefile(roundoff_control)
    rounding["complete"] && rounding["passed"] && rounding["outside_absolute"]<1e-12 || error("absolute roundoff control missing")
    for (p,h) in rounding["source_hashes"]
        bytes2hex(sha256(read(p)))==h || error("stale roundoff control")
        hashes[p]=h
    end
    rounding_script=joinpath(@__DIR__,"antiunitary_roundoff_audit.jl")
    rounding["script_sha256"]==bytes2hex(sha256(read(rounding_script))) || error("roundoff script changed")
    hashes[rounding_script]=rounding["script_sha256"]
    hashes[roundoff_control]=bytes2hex(sha256(read(roundoff_control)))
    for path in controls
        check=TOML.parsefile(path)
        check["complete"] && check["passed"] && check["independent_left_right"] || error("tangent control missing")
        for (p,h) in check["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("stale tangent control: $p")
            hashes[p]=h
        end
        hashes[path]=bytes2hex(sha256(read(path)))
    end
    for file in ("newton_antiunitary_branch_v2.jl","krylov_trust.jl","ellipsoid_trust.jl")
        p=joinpath(@__DIR__,file);hashes[p]=bytes2hex(sha256(read(p)))
    end
    paths=joinpath.(source,("boundary_1.jls","boundary_3.jls","independent_pair.jls"))
    for p in paths;hashes[p]=bytes2hex(sha256(read(p)));end
    n,s=deserialize.(paths[1:2]);pair=deserialize(paths[3]);maps=ba_pair_maps(n,s)
    R,L=deepcopy(pair.R),deepcopy(pair.L)
    if perturb>0
        states=[]
        for (i,state,change) in ((1,R,maps.right),(2,L,maps.left))
            symmetry=at_symmetry(state,change)
            raw=state.AL[1]+perturb*randn(MersenneTwister(3030+i),ComplexF64,space(state.AL[1]))
            raw=(raw+symmetry.apply(raw))/2;raw=raw*bn_invsqrt(raw'*raw)
            candidate,_=bt_canonical(raw);push!(states,candidate)
        end
        R,L=states
    end
    meta=Dict{String,Any}("schema"=>BIVUMPS_SCHEMA,"complete"=>false,"accepted_entropy"=>false,
        "independent_left_right"=>true,"direction_constraint"=>false,
        "update_strategy"=>"independent_antiunitary_branch_Newton",
        "roundoff_control"=>roundoff_control,"gradient_symmetry_atol"=>1e-12,"gradient_symmetry_rtol"=>1e-8,
        "chi"=>sum(n.chi),"chi_sectors"=>collect(n.chi),"source_hashes"=>hashes,
        "run_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))),
        "job_id"=>ENV["SLURM_JOB_ID"],"tolerance"=>tol,"initial_perturbation"=>perturb,
        "linear_control"=>bk_control(),"ellipsoid_control"=>be_control(),
        "final_requires_original_dominant_audit"=>true,"final_requires_dominant_center_roots"=>true)
    bv_write(joinpath(out,"input.toml"),meta)
    rows=Dict{String,Any}[];refs=nothing;best=nothing;reason="maxiter"
    for iteration in 0:maxiter
        audit=bv_audit(R,L,pair.transfer)
        if best===nothing || audit.residual<best.residual
            best=(R=deepcopy(R),L=deepcopy(L),residual=audit.residual,iteration=iteration,refs=deepcopy(refs))
            serialize(joinpath(out,"best_independent_pair.jls"),(;R,L,transfer=pair.transfer))
        end
        row=merge(audit.report,Dict("iteration"=>iteration,"radius"=>radius));push!(rows,row)
        bv_write(joinpath(out,"progress.toml"),merge(meta,Dict("rows"=>rows)))
        println("Antiunitary Newton iteration=",iteration," full residual=",audit.residual," radius=",radius);flush(stdout)
        if audit.residual<tol;reason="outer_residual_passed";break;end
        iteration==maxiter && break
        try
            symmetry=at_tangent(R,L,maps);row["tangent_symmetry"]=symmetry.report
            sr=symmetry.report
            errors=[sr["right"]["coefficient_defect"],sr["left"]["coefficient_defect"],
                sr["involution_error"],sr["symmetry_error"],sr["basis_fixed_error"]]
            maximum(errors)<1e-8 || error("symmetry tangent lost: $(maximum(errors))")
            c=btm_chart(R,L,pair.transfer;refs);refs=c.base.refs;Q=symmetry.Q
            g=Q'*c.gradient;leakage=norm(c.gradient-Q*g)/max(norm(c.gradient),eps())
            row["gradient_outside_fixed_fraction"]=leakage
            absolute_leakage=norm(c.gradient-Q*g)
            row["gradient_outside_fixed_absolute"]=absolute_leakage
            row["gradient_norm"]=norm(c.gradient)
            # Near a stationary zero, retain the verified absolute arithmetic
            # floor in this symmetry diagnostic. Full equation tolerance is unchanged.
            absolute_leakage<=1e-12+1e-8norm(c.gradient) || error("full gradient outside symmetry tangent: $absolute_leakage")
            row["branch_caps"]=c.base.reports;row["gradient_norm"]=norm(c.gradient)
            action=v->begin
                raw=Q*v;scale=max(norm(c.pushforward(raw)),eps())
                scale*(Q'*c.action(raw/scale))
            end
            pushforward=v->c.pushforward(Q*v)
            basis=bk_basis(action,g;maxbasis=size(Q,2));row["Krylov"]=basis.report
            geometry=be_geometry(basis,pushforward)
            trials=Dict{String,Any}[];row["trials"]=trials;accepted=nothing
            for attempt in 1:12
                step=be_step(basis,geometry,radius);delta=step.delta
                predicted=action(delta);gain=(sum(abs2,g)-sum(abs2,g+predicted))/2
                t=Dict{String,Any}("radius"=>radius,"linear_residual"=>norm(g+predicted)/norm(g),
                    "raw_step_norm"=>norm(pushforward(delta)));push!(trials,t)
                if gain>1e-14sum(abs2,g)
                    try
                        proposal=c.evaluate(Q*delta)
                        defect=max(norm(symmetry.sr.apply(proposal.R)-proposal.R)/norm(proposal.R),
                            norm(symmetry.sl.apply(proposal.L)-proposal.L)/norm(proposal.L))
                        t["retracted_symmetry_defect"]=defect
                        defect<1e-8 || error("finite update left symmetry manifold")
                        # Include the discarded component in the merit: the
                        # restricted equations alone cannot certify improvement.
                        actual=(sum(abs2,c.gradient)-sum(abs2,proposal.gradient))/2
                        ratio=actual/gain;t["gain_ratio"]=ratio
                        if actual>0 && ratio>0.1
                            rr,gr=bt_canonical(proposal.R);ll,gl=bt_canonical(proposal.L)
                            t["canonical_right"]=gr;t["canonical_left"]=gl
                            accepted=(rr,ll,proposal.refs)
                            if ratio<0.25;radius/=4;elseif ratio>0.75 && step.metricnorm>0.9;radius=min(0.1,2radius);end
                            break
                        end
                    catch err
                        err isa ErrorException || rethrow()
                        t["error"]=sprint(showerror,err)
                    end
                end
                radius/=4
            end
            accepted===nothing && (reason="Newton_trust_step_failed";break)
            R,L,refs=accepted
        catch err
            err isa ErrorException || rethrow()
            row["error"]=sprint(showerror,err);reason="Newton_response_failed";break
        end
    end
    R,L=best.R,best.L;audit=bv_audit(R,L,pair.transfer)
    final_branch=bt_cache(R.AL[1],L.AL[1],pair.transfer;refs=best.refs,responses=false)
    dominant=all(v["modulus_rank"]==1 for v in values(final_branch.reports))
    roots=an_roots(audit,R,L);center_dominant=all(v["stationary_modulus_rank"]==1 for v in roots)
    converged=audit.residual<tol && dominant && center_dominant
    S=MPSKit.InfiniteMPS([maps.unbra(L.AR[1])];tol=1e-13,maxiter=1000)
    for (name,old,state) in (("boundary_1.jls",n,R),("boundary_3.jls",s,S))
        b=save_boundary(joinpath(out,name),old,state,old.transfer,best.iteration)
        serialize(joinpath(out,name),merge(b,(;source=:independent_antiunitary_Newton,
            bivumps_converged=converged,bivumps_residual=audit.residual)))
    end
    serialize(joinpath(out,"independent_pair.jls"),(;R,L,transfer=pair.transfer))
    all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
    report=merge(meta,Dict("complete"=>true,"converged"=>converged,"stop_reason"=>reason,
        "exported_best_iteration"=>best.iteration,"final"=>audit.report,"rows"=>rows,
        "final_branch_dominant"=>dominant,"final_center_dominant"=>center_dominant,
        "final_branch_caps"=>final_branch.reports,"final_center_roots"=>roots,
        "xi_pair"=>bv_xi(R,L),"xi_mps_right"=>bv_xi(R,R),"xi_mps_left"=>bv_xi(L,L)))
    bv_write(joinpath(out,"report.toml"),report)
    println("Antiunitary Newton finished: ",audit.residual," converged=",converged," reason=",reason);flush(stdout)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    an_run(ARGS[1],ARGS[2],split(ARGS[3],',');maxiter=parse(Int,ARGS[4]),perturb=parse(Float64,ARGS[5]),roundoff_control=ARGS[6])
end
