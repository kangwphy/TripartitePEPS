# Independent normalized-center-equation Newton. The merit directly contains
# the four raw AC/C residual vectors (RMS rather than max normalization).
# Derivatives include the self-density inverse square root and normalization;
# original full complex/whitened residual gates remain unchanged.
# Full complex equations and final dominant roots remain the acceptance test.
include("center_gradient_response.jl")
include("point_krylov_trust.jl")

function cn_roots(audit,R,L)
    rows=[]
    for (name,P,r,l) in (("AC",audit.env.ac,R.AC[1],L.AC[1]),("C",audit.env.c,R.C[1],L.C[1]))
        w=bv_white(P.H,P.N);values=eigvals(w.K)
        z=dot(l.data,P.H*r.data)/dot(l.data,P.N*r.data)
        push!(rows,Dict("center"=>name,"stationary_modulus_rank"=>count(abs.(values).>abs(z)*(1+1e-8))+1,
            "nearest_eigenvalue_relative_error"=>minimum(abs.(values.-z))/abs(z)))
    end
    rows
end

# Remove only verified roundoff-size drift from each independently solved state.
# This is a fixed per-iteration operation, not a retry until a guard passes.
function cn_reproject(state,change)
    symmetry=at_symmetry(state,change)
    raw=(state.AL[1]+symmetry.apply(state.AL[1]))/2
    raw=raw*bn_invsqrt(raw'*raw)
    candidate,canonical=bt_canonical(raw)
    difference=norm(candidate.AL[1]-state.AL[1])/norm(state.AL[1])
    difference<=1e-10 || error("symmetry repair is not roundoff-size: $difference")
    report=Dict("coefficient_change"=>difference,"before"=>symmetry.report,
        "after"=>at_symmetry(candidate,change).report,"canonical"=>canonical)
    candidate,report
end

function cn_run(source,out,controls;maxiter=30,tol=1e-11,perturb=0.0,radius=0.01,roundoff_control,projection_control,paired_controls,center_controls,convergence_control=nothing,resume_snapshot=nothing,restart_radius=nothing)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");mkpath(out);BLAS.set_num_threads(2)
    hashes=Dict{String,String}()
    for path in vcat(paired_controls,center_controls)
        control=TOML.parsefile(path)
        control["complete"] && control["passed"] && control["nonstationary"] || error("paired or center-gradient control missing")
        control["independent_left_right"] && !control["direction_constraint"] || error("paired control uses constrained boundaries")
        for (p,h) in control["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("stale paired metric control: $p")
            hashes[p]=h
        end
        hashes[path]=bytes2hex(sha256(read(path)))
    end
    projection=TOML.parsefile(projection_control)
    projection["complete"] && projection["diagnostic_only"] || error("projection control missing")
    original=only(r for r in projection["rows"] if r["mode"]=="original")
    projected=only(r for r in projection["rows"] if r["mode"]=="projected")
    projected["coefficient_change"]<1e-12 && projected["full_residual_change"]<1e-11 || error("projection changes control state")
    original["weighted_outside_absolute"]>1e-12+1e-8original["weighted_gradient_norm"] || error("control does not reproduce original failure")
    projected["weighted_outside_absolute"]<=1e-12+1e-8projected["weighted_gradient_norm"] || error("projection does not resolve control diagnostic")
    for (p,h) in projection["source_hashes"]
        bytes2hex(sha256(read(p)))==h || error("stale projection control: $p")
        hashes[p]=h
    end
    hashes[projection_control]=bytes2hex(sha256(read(projection_control)))
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
    for file in ("newton_antiunitary_center_v6.jl","point_krylov_trust.jl","krylov_trust.jl","ellipsoid_trust.jl")
        p=joinpath(@__DIR__,file);hashes[p]=bytes2hex(sha256(read(p)))
    end
    paths=joinpath.(source,("boundary_1.jls","boundary_3.jls","independent_pair.jls"))
    for p in paths;hashes[p]=bytes2hex(sha256(read(p)));end
    n,s=deserialize.(paths[1:2]);pair=deserialize(paths[3]);maps=ba_pair_maps(n,s)
    if sum(n.chi)>4
        convergence_control!==nothing || error("matching small-chi solver control required")
        control=TOML.parsefile(convergence_control)
        control["complete"] && control["converged"] && control["final"]["outer_residual"]<1e-9 || error("small-chi solver control not converged")
        control["final_branch_dominant"] && control["final_center_dominant"] || error("small-chi root control failed")
        control["run_sha256"]==bytes2hex(sha256(read(@__FILE__))) || error("small-chi solver differs")
        for (p,h) in control["source_hashes"]
            bytes2hex(sha256(read(p)))==h || error("small-chi control input changed")
            hashes[p]=h
        end
        hashes[convergence_control]=bytes2hex(sha256(read(convergence_control)))
    end
    if resume_snapshot!==nothing
        hashes[resume_snapshot]=bytes2hex(sha256(read(resume_snapshot)))
        resumed=deserialize(resume_snapshot)
        norm(PEPSKit.ket(resumed.transfer[1])-PEPSKit.ket(pair.transfer[1]))<1e-12 &&
            norm(PEPSKit.bra(resumed.transfer[1])-PEPSKit.bra(pair.transfer[1]))<1e-12 || error("snapshot transfer mismatch")
        pair=resumed;radius=pair.radius
    end
    restart_radius!==nothing && (radius=restart_radius)
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
        "update_strategy"=>"independent_antiunitary_center_Newton",
        "roundoff_control"=>roundoff_control,"paired_controls"=>paired_controls,
        "center_gradient_controls"=>center_controls,"merit"=>"normalized_AC_C_center_equation_norm",
        "center_jacobian_symmetry_assumed"=>false,
        "convergence_control"=>(convergence_control===nothing ? "" : convergence_control),
        "Newton_coordinates"=>"local_mixed_tangent_positive_normalizers",
        "initial_radius"=>radius,"projection_control"=>projection_control,
        "per_iteration_symmetry_reprojection"=>true,"reprojection_max_coefficient_change"=>1e-10,
        "resumed_snapshot"=>(resume_snapshot===nothing ? "" : resume_snapshot),"gradient_symmetry_atol"=>1e-12,"gradient_symmetry_rtol"=>1e-8,
        "chi"=>sum(n.chi),"chi_sectors"=>collect(n.chi),"source_hashes"=>hashes,
        "run_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))),
        "job_id"=>ENV["SLURM_JOB_ID"],"tolerance"=>tol,"initial_perturbation"=>perturb,
        "linear_control"=>pk_control(),"ellipsoid_control"=>be_control(),
        "final_requires_original_dominant_audit"=>true,"final_requires_dominant_center_roots"=>true)
    bv_write(joinpath(out,"input.toml"),meta)
    rows=Dict{String,Any}[];refs=hasproperty(pair,:refs) ? pair.refs : nothing;best=nothing;reason="maxiter"
    for iteration in 0:maxiter
        R,projectionR=cn_reproject(R,maps.right)
        L,projectionL=cn_reproject(L,maps.left)
        checkpoint=(;R=deepcopy(R),L=deepcopy(L),transfer=pair.transfer,refs=deepcopy(refs),radius,iteration)
        serialize(joinpath(out,"last_iteration_pair.tmp.jls"),checkpoint)
        mv(joinpath(out,"last_iteration_pair.tmp.jls"),joinpath(out,"last_iteration_pair.jls");force=true)
        audit=bv_audit(R,L,pair.transfer)
        if best===nothing || audit.residual<best.residual
            best=(R=deepcopy(R),L=deepcopy(L),residual=audit.residual,iteration=iteration,refs=deepcopy(refs))
            serialize(joinpath(out,"best_independent_pair.jls"),(;R,L,transfer=pair.transfer))
        end
        row=merge(audit.report,Dict("iteration"=>iteration,"radius"=>radius,
            "reprojection_right"=>projectionR,"reprojection_left"=>projectionL));push!(rows,row)
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
            c=cg_chart(R,L,pair.transfer;refs);refs=c.base.refs;Q=symmetry.Q
            nr=length(c.weights.tr.data);nl=length(c.weights.tl.data);count=nr+nl
            JR=complex.(symmetry.S[1:nr,1:nr],symmetry.S[count+1:count+nr,1:nr])
            JL=complex.(symmetry.S[nr+1:count,nr+1:count],symmetry.S[count+nr+1:2count,nr+1:count])
            WR,WL=c.weights.WR,c.weights.WL
            covariance=hypot(norm(JR*conj.(WR)-WR*JR),norm(JL*conj.(WL)-WL*JL))/hypot(norm(WR),norm(WL))
            row["paired_metric"]=c.weights.report;row["paired_symmetry_commutator"]=covariance
            covariance<1e-8 || error("paired metric lost symmetry covariance: $covariance")
            g=Q'*c.gradient;leakage=norm(c.gradient-Q*g)/max(norm(c.gradient),eps())
            row["gradient_outside_fixed_fraction"]=leakage
            absolute_leakage=norm(c.gradient-Q*g)
            row["gradient_outside_fixed_absolute"]=absolute_leakage
            row["gradient_norm"]=norm(c.gradient)
            # Near a stationary zero, retain the verified absolute arithmetic
            # floor in this symmetry diagnostic. Full equation tolerance is unchanged.
            absolute_leakage<=1e-12+1e-8norm(c.gradient) || error("full gradient outside symmetry tangent: $absolute_leakage")
            row["branch_caps"]=c.base.reports;row["gradient_norm"]=norm(c.gradient)
            row["center_field_norm"]=norm(c.field)
            row["self_density_right"]=c.density_right;row["self_density_left"]=c.density_left
            row["density_C_error"]=c.density_error
            images=Vector{Float64}[]
            action=v->begin
                dx=c.pushforward(Q*v);scale=max(norm(dx),eps())
                j=c.action(Q*v/scale);push!(images,scale*j.field)
                scale*(Q'*j.gradient)
            end
            pushforward=v->c.pushforward(Q*v)
            basis=pk_basis(action,g;maxbasis=size(Q,2));row["Krylov"]=basis.report
            model=pk_raw_model(basis,images,c.field)
            geometry=be_geometry(model,pushforward)
            trials=Dict{String,Any}[];row["trials"]=trials;accepted=nothing
            for attempt in 1:12
                step=be_step(model,geometry,radius);delta=step.delta
                predicted=model.H*(basis.basis'*delta)
                gain=(sum(abs2,c.field)-sum(abs2,c.field+predicted))/2
                t=Dict{String,Any}("radius"=>radius,
                    "linear_raw_residual"=>norm(c.field+predicted)/norm(c.field),
                    "raw_step_norm"=>norm(pushforward(delta)));push!(trials,t)
                if gain>1e-14sum(abs2,c.field)
                    try
                        proposal=c.evaluate(Q*delta)
                        defect=max(norm(symmetry.sr.apply(proposal.R)-proposal.R)/norm(proposal.R),
                            norm(symmetry.sl.apply(proposal.L)-proposal.L)/norm(proposal.L))
                        t["retracted_symmetry_defect"]=defect
                        defect<1e-8 || error("finite update left symmetry manifold")
                        # The merit contains both AC and C center equations
                        # on both independent sides, with RMS H/N normalization.
                        # Evaluate the density metric at the actual candidate.
                        actual=(sum(abs2,c.field)-sum(abs2,proposal.field))/2
                        ratio=actual/gain;t["gain_ratio"]=ratio
                        t["center_field_after"]=norm(proposal.field)
                        if actual>0 && ratio>0.1
                            rr,gr=bt_canonical(proposal.R);ll,gl=bt_canonical(proposal.L)
                            t["canonical_right"]=gr;t["canonical_left"]=gl
                            fresh=cg_cache(rr.AL[1],ll.AL[1],pair.transfer;refs=proposal.refs,responses=false)
                            a,b=rr.AL[1],ll.AL[1]
                            rhoR,rhoL=cg_density(a;responses=false),cg_density(b;responses=false)
                            freshnorm=norm(cg_fields(a,b,fresh,rhoR.W,rhoL.W).field)
                            difference=abs(freshnorm-norm(proposal.field))
                            # Use the actual post-canonical field for acceptance;
                            # record its sensitivity to the roundoff-size gauge
                            # alignment rather than assuming a fixed metric.
                            t["canonical_center_field_norm"]=freshnorm
                            t["canonical_merit_norm_difference"]=difference
                            actual=(sum(abs2,c.field)-freshnorm^2)/2
                            ratio=actual/gain;t["canonical_gain_ratio"]=ratio
                            if actual>0 && ratio>0.1
                                accepted=(rr,ll,fresh.refs)
                                if ratio<0.25;radius/=4;elseif ratio>0.75 && step.metricnorm>0.9;radius=min(0.1,2radius);end
                                break
                            end
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
    roots=cn_roots(audit,R,L);center_dominant=all(v["stationary_modulus_rank"]==1 for v in roots)
    converged=audit.residual<tol && dominant && center_dominant
    S=MPSKit.InfiniteMPS([maps.unbra(L.AR[1])];tol=1e-13,maxiter=1000)
    for (name,old,state) in (("boundary_1.jls",n,R),("boundary_3.jls",s,S))
        b=save_boundary(joinpath(out,name),old,state,old.transfer,best.iteration)
        serialize(joinpath(out,name),merge(b,(;source=:independent_antiunitary_center_Newton,
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
    cn_run(ARGS[1],ARGS[2],split(ARGS[3],',');maxiter=parse(Int,ARGS[4]),perturb=parse(Float64,ARGS[5]),roundoff_control=ARGS[6],projection_control=ARGS[7],paired_controls=split(ARGS[8],Char(44)),center_controls=split(ARGS[9],Char(44)),
        convergence_control=length(ARGS)>9 ? ARGS[10] : nothing)
end
