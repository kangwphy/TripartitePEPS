# Gauge-fixed finite-difference least-squares solve of the ORIGINAL Galerkin
# equation. Independent of production VUMPS; accepted checkpoints use its gates.
isdefined(@__MODULE__,:save_boundary) || include("stable_boundary_pilot.jl")

function block_invsqrt(T)
    out=copy(T)
    for (q,b) in blocks(T)
        d=eigen(Hermitian((b+b')/2))
        minimum(d.values)>0 || error("singular chart metric")
        block(out,q) .= d.vectors*Diagonal(1 ./ sqrt.(d.values))*d.vectors'
    end
    out
end

function center_polar_svd(C;inverse_power=1.0)
    isfinite(inverse_power) && 0<=inverse_power<=1 || error("invalid inverse-center power")
    inverse=zero(C*C');polar=zero(C)
    for (q,B) in blocks(C)
        F=svd(B)
        minimum(F.S)>0 || error("singular chart center")
        weights=inverse_power==1 ? 1 ./ F.S : F.S .^ (-inverse_power)
        block(inverse,q).=F.U*Diagonal(weights)*F.U'
        block(polar,q).=F.U*F.Vt
    end
    norm(polar'*polar-id(ComplexF64,domain(polar)))<1e-10 || error("center polar factor not unitary")
    inverse,polar
end

function chart_residual(x,base,N,template,op)
    inner_tolerance=parse(Float64,get(ENV,"FPEPS_NEWTON_INNER_TOL","1e-13"))
    0<inner_tolerance<=1e-10 || error("invalid Newton inner tolerance")
    X=copy(template); n=length(X.data)
    if length(x)==n
        X.data .= x
    elseif length(x)==2n
        X.data .= complex.(x[1:n],x[n+1:2n])
    else
        error("invalid residual-chart parameter count")
    end
    # This polar retraction keeps AL in a fixed, smooth left gauge.
    A=base.AL[1]+N*X
    A=A*block_invsqrt(A'*A)
    state=MPSKit.InfiniteMPS([A],copy(base.C[1]);tol=inner_tolerance,maxiter=1000)
    env=MPSKit.environments(state,op;tol=inner_tolerance)
    ac=MPSKit.AC_projection(1,state,op,state,env); ac/=norm(ac)
    z=dot(state.AC[1],ac); ac*=conj(z)/abs(z)
    R=ac-A*(A'*ac)
    # Remove the arbitrary right-canonical gauge by the polar unitary of C.
    # Right multiplication by Q' preserves exactly the native residual norm.
    # Form the polar unitary directly. Diagonalizing C*C' squares its
    # condition number and corrupts Q in weak Schmidt directions at chi32.
    center_power=parse(Float64,get(ENV,"FPEPS_NEWTON_CENTER_POWER","1.0"))
    C=state.C[1];inverse_center,Q=center_polar_svd(C;inverse_power=center_power)
    R=R*Q'
    f=[real.(R.data);imag.(R.data)]
    tangent=N'*R
    tangent_f=[real.(tangent.data);imag.(tangent.data)]
    # Invertible right preconditioning exposes errors in weak Schmidt modes.
    # Optional powers between zero and one temper their amplification; zero
    # recovers the native norm and one retains the original inverse center.
    # It has the same zeros for a full-rank center; final acceptance still
    # checks the original native Galerkin equation and downstream physics.
    unweighted=R*inverse_center
    unweighted_f=[real.(unweighted.data);imag.(unweighted.data)]
    unweighted_tangent=N'*unweighted
    unweighted_tangent_f=[real.(unweighted_tangent.data);imag.(unweighted_tangent.data)]
    native=MPSKit.calc_galerkin(state,op,state,env)
    abs(norm(f)-native)<1e-10 || error("residual chart does not match native Galerkin: chart=$(norm(f)), native=$native")
    (;f,tangent_f,unweighted_f,unweighted_tangent_f,state,env,native)
end

residual_coordinates(r,coordinates)=coordinates=="unweighted" ? r.unweighted_f :
    coordinates=="unweighted_tangent" ? r.unweighted_tangent_f :
    coordinates=="tangent" ? r.tangent_f : r.f
residual_merit(r,coordinates)=coordinates in ("unweighted","unweighted_tangent") ? norm(r.unweighted_f) : r.native

function residual_jacobian!(J,h,x,base,N,template,op;parallel=false,coordinates="ambient",order=2)
    order in (2,4) || error("finite difference order must be 2 or 4")
    completed=Threads.Atomic{Int}(0)
    function column(j)
        xp=copy(x);xp[j]=h;xm=copy(x);xm[j]=-h
        plus=residual_coordinates(chart_residual(xp,base,N,template,op),coordinates)
        minus=residual_coordinates(chart_residual(xm,base,N,template,op),coordinates)
        J[:,j]=(plus-minus)/(2h)
        if order==4
            xp[j]=h/2;xm[j]=-h/2
            halfplus=residual_coordinates(chart_residual(xp,base,N,template,op),coordinates)
            halfminus=residual_coordinates(chart_residual(xm,base,N,template,op),coordinates)
            J[:,j]=(4*(halfplus-halfminus)/h-J[:,j])/3
        end
        done=Threads.atomic_add!(completed,1)+1
        if size(J,2)>=768 && (done%256==0 || done==size(J,2))
            println("Jacobian h=",h," columns=",done,"/",size(J,2));flush(stdout)
        end
    end
    if parallel
        Threads.@threads for j in axes(J,2)
            column(j)
        end
    else
        foreach(column,axes(J,2))
    end
    J
end

function newton_checkpoint(path,data)
    temporary=path*".tmp_$(getpid())"
    serialize(temporary,data)
    mv(temporary,path;force=true)
end

function schmidt_parameter_map(template,C,power;real_parameters=false)
    0<power<=1 || error("parameter power must be in (0,1]")
    inverse,_=center_polar_svd(C;inverse_power=power)
    n=length(template.data);count=(real_parameters ? 1 : 2)*n
    K=zeros(count,count)
    for j in 1:count
        basis=zero(template);basis.data[mod1(j,n)]=j<=n ? 1 : im
        transformed=basis*inverse
        if real_parameters
            norm(imag.(transformed.data))<1e-10*norm(transformed) ||
                error("Schmidt parameter map leaves the verified real chart")
            K[:,j]=real.(transformed.data)
        else
            K[:,j]=[real.(transformed.data);imag.(transformed.data)]
        end
    end
    K
end

function newton_main(source,name,output,maxiter;
        nullspace_builder=base->TensorKit.left_null(base.AL[1]),real_parameters=false,
        checkpoint_metadata=old->(;chi=old.chi,peps=old.peps,direction=old.direction),boundary_writer=save_boundary)
    mkpath(output); old=deserialize(joinpath(source,name));op=old.transfer
    search=get(ENV,"FPEPS_NEWTON_SEARCH","first")
    search in ("first","best") || error("unknown Newton search strategy")
    parallel=get(ENV,"FPEPS_NEWTON_PARALLEL","0")=="1"
    coordinates=get(ENV,"FPEPS_NEWTON_COORDINATES","ambient")
    center_power=parse(Float64,get(ENV,"FPEPS_NEWTON_CENTER_POWER","1.0"))
    isfinite(center_power) && 0<=center_power<=1 || error("invalid inverse-center power")
    parameter_power=parse(Float64,get(ENV,"FPEPS_NEWTON_PARAMETER_POWER","0"))
    isfinite(parameter_power) && 0<=parameter_power<=1 || error("invalid parameter power")
    inner_tolerance=parse(Float64,get(ENV,"FPEPS_NEWTON_INNER_TOL","1e-13"))
    coordinates in ("ambient","tangent","unweighted","unweighted_tangent") || error("unknown residual coordinates")
    fd_start=parse(Float64,get(ENV,"FPEPS_NEWTON_FD_START",search=="best" ? "1e-4" : "1e-3"))
    fd_interleave=get(ENV,"FPEPS_NEWTON_FD_INTERLEAVE","0")=="1"
    fd_order=parse(Int,get(ENV,"FPEPS_NEWTON_FD_ORDER","2"))
    fd_order in (2,4) || error("finite difference order must be 2 or 4")
    regularization_interleave=get(ENV,"FPEPS_NEWTON_REG_INTERLEAVE","0")=="1"
    jacobian_tolerance=parse(Float64,get(ENV,"FPEPS_NEWTON_JAC_TOL","1e-3"))
    fd_start>0 && 0<jacobian_tolerance<1 || error("invalid finite difference settings")
    native_tolerance=parse(Float64,get(ENV,"FPEPS_NEWTON_NATIVE_TOL","1e-12"))
    0<native_tolerance<=1e-12 || error("native stopping tolerance may only tighten the original gate")
    base=old.state;rows=Dict{String,Any}[];start=time()
    for iteration in 0:maxiter
        N=nullspace_builder(base)
        template=N'*base.AL[1];fill!(template.data,0)
        count=(real_parameters ? 1 : 2)*length(template.data); x=zeros(count)
        current=chart_residual(x,base,N,template,op)
        current_f=residual_coordinates(current,coordinates)
        current_merit=residual_merit(current,coordinates)
        # Tangent coordinates retain the full residual at the chart origin;
        # their trial merit uses the ambient native norm. The explicit
        # unweighted option uses an invertible inverse-center preconditioner
        # for its merit. All options retain the native final convergence gate.
        abs(norm(current_f)-current_merit)<1e-10 || error("residual lost at chart origin")
        println("Newton iteration=",iteration," original residual=",current.native," variables=",count);flush(stdout)
        row=Dict{String,Any}("iteration"=>iteration,"galerkin"=>current.native,"elapsed_seconds"=>time()-start,
            "search_strategy"=>search,"parallel_jacobian"=>parallel,"threads"=>Threads.nthreads(),
            "residual_coordinates"=>coordinates,"merit_residual"=>current_merit,
            "inverse_center_power"=>center_power,
            "parameter_power"=>parameter_power,
            "inner_tolerance"=>inner_tolerance,
            "fd_interleave"=>fd_interleave,
            "fd_order"=>fd_order,
            "regularization_interleave"=>regularization_interleave,
            "real_parameters"=>real_parameters)
        merit_noise=5e-15
        if coordinates in ("unweighted","unweighted_tangent")
            repeat_errors=[norm(chart_residual(x,base,N,template,op).unweighted_f-current.unweighted_f) for _ in 1:2]
            merit_noise=max(merit_noise,5maximum(repeat_errors))
            row["merit_repeat_vector_changes"]=repeat_errors
            println("inverse-center residual=",current_merit," resolved-decrease floor=",merit_noise);flush(stdout)
        end
        lambda=FermionicPEPS._boundary_expectation(current.state,op,current.env)
        row["lambda_real"]=real(lambda);row["lambda_imag"]=imag(lambda)
        if length(current.env.GLs[1].data)<=600
            _,spectrum=dense_candidates(x->x*MPSKit.TransferMatrix(current.state.AL,op,current.state.AL),current.env.GLs[1])
            row["left_environment_spectrum"]=spectrum
            println("environment multiplicity=",spectrum["cluster_size"]," relative gap=",spectrum["relative_modulus_gap"]);flush(stdout)
        else
            # Large-chi residual experiments still require a separate full
            # spectral audit before interpreting a stationary point physically.
            row["environment_multiplicity_checked"]=false
        end
        push!(rows,row)
        newton_checkpoint(joinpath(output,"best_state.jls"),merge((;state=current.state,transfer=op,environments=current.env),checkpoint_metadata(old)))
        if current.native<=native_tolerance || iteration==maxiter
            base=current.state;break
        end
        singular=reduce(vcat,[svdvals(b) for (_,b) in blocks(current.state.C[1])])
        row["schmidt_values"]=singular
        println("Schmidt min/max=",extrema(singular));flush(stdout)
        J=zeros(length(current_f),count)
        rng=MersenneTwister(770+iteration);v=randn(rng,count);v/=norm(v)
        derr=Inf
        steps=[fd_start/10.0^k for k in 0:3]
        # Near weak Schmidt modes, the resolved interval can be narrower
        # than a decade. This opt-in grid avoids skipping that interval.
        if fd_interleave
            append!(steps,[fd_start/(3*10.0^k) for k in 0:2])
            sort!(steps;rev=true)
        end
        for h in steps
            residual_jacobian!(J,h,x,base,N,template,op;parallel,coordinates,order=fd_order)
            # Check a directional derivative independently before using the Jacobian.
            check=(residual_coordinates(chart_residual((h/2)*v,base,N,template,op),coordinates)-
                residual_coordinates(chart_residual(-(h/2)*v,base,N,template,op),coordinates))/h
            if fd_order==4
                quarter=(residual_coordinates(chart_residual((h/4)*v,base,N,template,op),coordinates)-
                    residual_coordinates(chart_residual(-(h/4)*v,base,N,template,op),coordinates))/(h/2)
                check=(4quarter-check)/3
            end
            derr=norm(check-J*v)/max(norm(check),eps())
            row["difference_step"]=h
            println("difference step=",h," directional error=",derr);flush(stdout)
            derr<jacobian_tolerance && break
        end
        row["jacobian_directional_error"]=derr
        open(joinpath(output,"iterations.toml"),"w") do io;TOML.print(io,Dict("iterations"=>rows));end
        row["jacobian_tolerance"]=jacobian_tolerance
        derr<jacobian_tolerance || error("nonsmooth or inaccurate residual Jacobian: $derr")
        # Change parameter units only. The residual, merit, finite-difference
        # check, original-coordinate step cap and every acceptance gate stay
        # unchanged. This differs from weighting the residual by inverse C.
        parameter_map=parameter_power==0 ? nothing :
            schmidt_parameter_map(template,current.state.C[1],parameter_power;real_parameters)
        dec=svd(parameter_map===nothing ? J : J*parameter_map)
        row["jacobian_singular_values"]=parameter_map===nothing ? dec.S : svdvals(J)
        if parameter_map!==nothing
            row["parameter_jacobian_singular_values"]=dec.S
        end
        row["merit_gradient_norm"]=norm(J'*current_f)
        println("Jacobian checked error=",derr," sigma min/max=",extrema(dec.S));flush(stdout)
        accepted=false
        # Levenberg regularization plus actual-residual backtracking.
        # The best strategy compares damping strengths before accepting, so a
        # tiny decreasing Newton step cannot hide a much better damped step.
        best_trial=nothing;best_merit=current_merit;best_meta=nothing
        trials=Dict{String,Any}[]
        regularizations=search=="best" ? (1e-10,1e-7,1e-5,1e-4,1e-3,1e-2,1e-1) : (1e-10,1e-7,1e-5,1e-3)
        if regularization_interleave
            search=="best" || error("interleaved regularization requires full trial comparison")
            # The old grid jumps across the weak Jacobian modes between
            # 1e-10 and 1e-7. Compare intervening damping values using the
            # SAME actual-residual, noise and Armijo gates; defaults unchanged.
            regularizations=sort!(unique!([collect(regularizations);
                [m*10.0^k for k in -10:-6 for m in (1,3)]]))
        end
        for reg in regularizations
            delta=-dec.V*((dec.S ./ (dec.S.^2 .+ (reg*maximum(dec.S))^2)).*(dec.U'*current_f))
            parameter_map!==nothing && (delta=parameter_map*delta)
            norm(delta)>0.2 && (delta*=0.2/norm(delta))
            model_delta=J*delta
            merit_slope=dot(current_f,model_delta)
            # Armijo uses the derivative of 1/2 ||R||^2 for the ACTUAL capped,
            # regularized step. A fixed relative decrease per alpha incorrectly
            # rejects small but accurate corrections near a stationary point.
            alphas=search=="best" ? [2.0^-k for k in 0:8] : [1.,.5,.25,.125,.0625]
            for alpha in alphas
                trial=chart_residual(alpha*delta,base,N,template,op)
                trial_merit=residual_merit(trial,coordinates)
                predicted=norm(current_f+alpha*model_delta)
                push!(trials,Dict("regularization"=>reg,"alpha"=>alpha,"residual"=>trial.native,
                    "predicted_residual"=>predicted,"step_norm"=>alpha*norm(delta),"merit_slope"=>merit_slope,
                    "merit_residual"=>trial_merit))
                sufficient=merit_slope<0 &&
                    trial_merit^2<=current_merit^2+2e-4*alpha*merit_slope &&
                    trial_merit<current_merit-merit_noise
                if search=="best"
                    if sufficient && trial_merit<best_merit
                        best_trial=trial;best_merit=trial_merit;best_meta=last(trials)
                    end
                    continue
                end
                if sufficient
                    base=trial.state;accepted=true
                    row["accepted_residual"]=trial.native;row["alpha"]=alpha;row["regularization"]=reg
                    row["accepted_merit"]=trial_merit
                    println("accepted residual=",trial.native," alpha=",alpha," regularization=",reg);flush(stdout)
                    break
                end
            end
            accepted && break
        end
        if search=="best" && best_trial!==nothing
            base=best_trial.state;accepted=true
            row["accepted_residual"]=best_trial.native;row["accepted_merit"]=best_merit
            for key in ("alpha","regularization","predicted_residual","step_norm")
                row[key]=best_meta[key]
            end
            println("best step native residual=",best_trial.native," merit=",best_merit," alpha=",row["alpha"],
                " regularization=",row["regularization"]," predicted=",row["predicted_residual"]);flush(stdout)
        end
        row["line_search_trials"]=trials
        open(joinpath(output,"iterations.toml"),"w") do io;TOML.print(io,Dict("iterations"=>rows));end
        accepted || (println("No sufficiently decreasing resolved step; retaining best checkpoint");break)
    end
    b=boundary_writer(joinpath(output,"diagnostic_boundary.jls"),old,base,op,length(rows)-1)
    b.converged && serialize(joinpath(output,name),b)
    report=Dict("converged"=>b.converged,"galerkin"=>b.galerkin,"source"=>source,
        "requested_native_tolerance"=>native_tolerance,
        "requested_native_tolerance_reached"=>b.galerkin<=native_tolerance,
        "residual_coordinates"=>coordinates,
        "inverse_center_power"=>center_power,
        "parameter_power"=>parameter_power,
        "inner_tolerance"=>inner_tolerance,
        "fd_interleave"=>fd_interleave,
        "fd_order"=>fd_order,
        "real_parameters"=>real_parameters,
        "boundary_name"=>name,"chi"=>sum(old.chi),"lambda_real"=>real(b.lambda),"lambda_imag"=>imag(b.lambda),
        "xi_even_boundary"=>boundary_even_correlation_length(b),"physical_replica_benchmark_certified"=>false)
    open(joinpath(output,"result.toml"),"w") do io;TOML.print(io,report);end
    open(joinpath(output,"iterations.toml"),"w") do io;TOML.print(io,Dict("iterations"=>rows));end
    println(report);flush(stdout)
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    newton_main(ARGS[1],ARGS[2],ARGS[3],length(ARGS)>=4 ? parse(Int,ARGS[4]) : 8)
end
