# Standalone stability experiment. No production method overrides.
println("loading one-site boundary stability pilot"); flush(stdout)
include("../src/FermionicPEPS.jl")
using .FermionicPEPS, MPSKit, PEPSKit, TensorKit, LinearAlgebra, Random, Serialization, TOML

function diagnostics(state,op)
    env=MPSKit.environments(state,op;tol=1e-13)
    residual=MPSKit.calc_galerkin(state,op,state,env)
    lambda=FermionicPEPS._boundary_expectation(state,op,env)
    (;env,residual,lambda)
end

function dense_candidates(H,old)
    d=length(old.data); d<=600 || error("dense pilot limited to local dimension 600")
    matrix=zeros(ComplexF64,d,d); v=copy(old)
    for j in 1:d
        fill!(v.data,0); v.data[j]=1
        matrix[:,j]=H(v).data
    end
    # Use one Schur decomposition throughout. Comparing eigenvalues from two
    # separately balanced factorizations is unreliable for nonnormal matrices.
    sd=schur(matrix);values=sd.values;radius=maximum(abs,values)
    cluster_tolerance=parse(Float64,get(ENV,"FPEPS_CLUSTER_TOL","1e-10"))
    order=sortperm(abs.(values);rev=true)
    candidates=filter(j->abs(values[j])>=radius*(1-1e-7),order)
    remaining=copy(candidates);best_overlap=-1.;vector=zeros(ComplexF64,d);selected_dimension=0
    while !isempty(remaining)
        chosen=first(remaining)
        selection=abs.(values .- values[chosen]) .< cluster_tolerance*max(radius,eps())
        ordered=ordschur(sd,selection);basis=ordered.Z[:,1:count(selection)]
        trial=basis*(basis'*old.data)
        score=norm(trial)/norm(old)
        if score>best_overlap
            best_overlap=score;vector=norm(trial)>1e-12*norm(old) ? trial : basis[:,1]
            selected_dimension=count(selection)
        end
        filter!(j->!selection[j],remaining)
    end
    vector/=norm(vector)
    overlap=dot(old.data,vector);abs(overlap)>0 && (vector*=conj(overlap)/abs(overlap))
    result=copy(old);result.data[:]=vector
    lambda=dot(vector,matrix*vector)
    inner=norm(matrix*vector-lambda*vector)/max(norm(matrix*vector),eps())
    meta=Dict("lambda_real"=>real(lambda),"lambda_imag"=>imag(lambda),"inner_residual"=>inner,
        "overlap"=>abs(dot(old.data,vector))/norm(old),"cluster_size"=>length(candidates),
        "basis_method"=>"single ordered Schur decomposition","selected_subspace_dimension"=>selected_dimension,
        "complex_eigenvalue_cluster_tolerance"=>cluster_tolerance,
        "relative_modulus_gap"=>1-abs(values[order[2]])/radius,
        "leading_real"=>real.(values[order[1:min(4,d)]]),
        "leading_imag"=>imag.(values[order[1:min(4,d)]]))
    result,meta
end

function tracked_step(state,op,env,alpha;pair_centers=false)
    c,mc=dense_candidates(MPSKit.C_hamiltonian(1,state,op,state,env),state.C[1])
    ac_reference=pair_centers ? state.AL[1]*c : state.AC[1]
    ac,ma=dense_candidates(MPSKit.AC_hamiltonian(1,state,op,state,env),ac_reference)
    ac=alpha*ac+(1-alpha)*state.AC[1]/norm(state.AC[1])
    c=alpha*c+(1-alpha)*state.C[1]/norm(state.C[1])
    tensor=MPSKit.regauge!(ac,c)
    next=MPSKit.InfiniteMPS([tensor],state.C[end];tol=1e-13,maxiter=1000)
    next,Dict("AC"=>ma,"C"=>mc)
end

function save_boundary(path,old,state,op,iteration)
    d=diagnostics(state,op)
    L,R=d.env.GLs[1],d.env.GRs[end]
    lr=FermionicPEPS._scaled_residual(L*MPSKit.TransferMatrix(state.AL,op,state.AL),L)
    rr=FermionicPEPS._scaled_residual(MPSKit.TransferMatrix(state.AR,op,state.AR)*R,R)
    cr=max(norm(state.AC[1]-state.AL[1]*state.C[1]),
           norm(state.AC[1]-MPSKit._mul_front(state.C[0],state.AR[1])))/norm(state.AC[1])
    V=MPSKit.left_virtualspace(state,1);chi=(dim(V,FermionParity(0)),dim(V,FermionParity(1)))
    converged=d.residual<=1e-12 && max(lr,rr,cr)<1e-8
    b=(;state,transfer=op,peps=old.peps,direction=old.direction,chi,
        environments=d.env,galerkin=d.residual,lambda=d.lambda,converged,
        left_residual=lr,right_residual=rr,center_residual=cr,iterations=iteration)
    serialize(path,b)
    b
end

function main(source,name,rank,mode,output,maxiter)
    rank in (10,12,16,24,32) || error("unsupported pilot rank")
    mode in ("fixedLM","fixedLR","track","track02") || error("unknown mode")
    mkpath(output);old=deserialize(joinpath(source,name));op=old.transfer
    alpha=mode=="track02" ? .2 : 1.
    if startswith(mode,"track")
        checked,meta=tracked_step(old.state,op,old.environments,alpha)
        check=diagnostics(checked,op)
        check.residual<1e-8 && abs(check.lambda-old.lambda)<1e-8 || error("known fixed point not preserved")
        println("known chi=",sum(old.chi)," fixed point regression passed ",check.residual);flush(stdout)
    end
    Random.seed!(93471+old.direction)
    extra=(rank÷2,rank÷2).-old.chi
    initial=MPSKit.changebonds(old.state,MPSKit.RandExpand(;trscheme=truncspace(FermionicPEPS._environment_space(extra))))
    before=diagnostics(initial,op)
    println("expanded from ",old.chi," to ",(rank÷2,rank÷2)," initial residual=",before.residual);flush(stdout)
    rows=Dict{String,Any}[];started=time();best=Inf;previous=Ref(copy(initial.AC[1]));used=Ref(0)
    function record(i,state,operator,env;localmeta=nothing)
        # This is a gauge-dependent center overlap, explicitly not a fidelity per site.
        res=MPSKit.calc_galerkin(state,operator,state,env)
        lambda=FermionicPEPS._boundary_expectation(state,operator,env)
        overlap=abs(dot(previous[],state.AC[1]))/(norm(previous[])*norm(state.AC[1]))
        previous[]=copy(state.AC[1]);used[]=i
        singular=reduce(vcat,[svdvals(v) for (_,v) in blocks(state.C[1])])
        row=Dict{String,Any}("iteration"=>i,"galerkin"=>res,"lambda_real"=>real(lambda),
            "lambda_imag"=>imag(lambda),"center_overlap_gauge_dependent"=>overlap,
            "min_schmidt"=>minimum(singular),"elapsed_seconds"=>time()-started)
        localmeta===nothing || (row["local_spectrum"]=localmeta)
        push!(rows,row)
        if res<best
            best=res
            serialize(joinpath(output,"best_state.jls"),(;state=copy(state),peps=old.peps,chi=(rank÷2,rank÷2),direction=old.direction))
        end
        if i<=5 || i%100==0 || res<1e-12
            println("iteration=",i," residual=",res," best=",best," lambda=",lambda);flush(stdout)
            open(joinpath(output,"iterations.toml"),"w") do io;TOML.print(io,Dict("iterations"=>rows));end
        end
        state,env
    end
    record(-1,initial,op,before.env)
    state=initial
    if startswith(mode,"fixed")
        inner=(;tol=1e-13,dynamic_tols=false)
        alg=MPSKit.VUMPS(;tol=1e-12,maxiter,verbosity=0,
            alg_gauge=MPSKit.Defaults.alg_gauge(;inner...),
            alg_environments=MPSKit.Defaults.alg_environments(;inner...),
            alg_eigsolve=MPSKit.Defaults.alg_eigsolve(;ishermitian=false,inner...),
            finalize=function(i,s,o,e)
                record(i,convert(MPSKit.InfiniteMPS,s),op,e[1])
                s,e
            end)
        # Use the native multiline API so eigenvalue selection is explicit.
        ms,ev,res=MPSKit.dominant_eigsolve(convert(MPSKit.MultilineMPO,op),
            convert(MPSKit.MultilineMPS,initial),alg,MPSKit.Multiline([before.env]);
            which=mode=="fixedLM" ? :LM : :LR)
        state=convert(MPSKit.InfiniteMPS,ms)
    else
        env=before.env
        for i in 1:maxiter
            state,meta=tracked_step(state,op,env,alpha)
            env=MPSKit.environments(state,op;tol=1e-13)
            record(i,state,op,env;localmeta=meta)
            rows[end]["galerkin"]<1e-12 && break
        end
    end
    open(joinpath(output,"iterations.toml"),"w") do io;TOML.print(io,Dict("iterations"=>rows));end
    b=save_boundary(joinpath(output,"diagnostic_boundary.jls"),old,state,op,used[])
    b.converged && serialize(joinpath(output,name),b)
    report=Dict("mode"=>mode,"chi"=>rank,"source"=>source,"boundary_name"=>name,
        "converged"=>b.converged,"galerkin"=>b.galerkin,"best_iteration_residual"=>best,
        "lambda_real"=>real(b.lambda),"lambda_imag"=>imag(b.lambda),"iterations"=>used[],
        "physical_replica_benchmark_certified"=>false)
    open(joinpath(output,"result.toml"),"w") do io;TOML.print(io,report);end
    println(report);flush(stdout)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    main(ARGS[1],ARGS[2],parse(Int,ARGS[3]),ARGS[4],ARGS[5],length(ARGS)>=6 ? parse(Int,ARGS[6]) : 3000)
end
