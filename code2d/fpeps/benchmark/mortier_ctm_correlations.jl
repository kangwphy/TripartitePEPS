# Same-tensor CTMRG comparison, including complex fermionic two-point functions.
println("loading CTMRG signed correlation scan");flush(stdout)
include("../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, PEPSKit, MPSKit, LinearAlgebra, Random, Serialization, TOML, Test

function car_two(j)
    out=zeros(ComplexF64,4,4)
    for mask in 0:3
        bit=1<<j
        mask&bit==0 && continue
        out[(mask⊻bit)+1,mask+1]=isodd(count_ones(mask&(bit-1))) ? -1 : 1
    end
    out
end
function signed_operators(V)
    a,b=car_two(0),car_two(1)
    Dict(name=>TensorMap(reshape(m,2,2,2,2),V^2←V^2) for (name,m) in
        ("normal"=>a'*b,"anomalous"=>a*b,"nn"=>(a'*a)*(b'*b)))
end
function measure(peps,env;check=false)
    vals=Dict{String,Any}(); site=CartesianIndex(1,1)
    for (name,O) in signed_operators(space(peps[1],1)), (dir,step) in
        (("x",CartesianIndex(0,1)),("y",CartesianIndex(1,0)))
        targets=[site+r*step for r in 1:32]
        corr=MPSKit.correlator(peps,O,site,targets,env)
        vals[name*"_"*dir*"_real"]=real.(corr)
        vals[name*"_"*dir*"_imag"]=imag.(corr)
        if check
            for r in (1,3)
                # Independent direct local contraction checks MPO transport parity.
                localop=PEPSKit.LocalOperator(PEPSKit.physicalspace(peps),(site,targets[r])=>O)
                direct=PEPSKit.expectation_value(peps,localop,env)
                @test isapprox(corr[r],direct;atol=1e-10,rtol=1e-8)
            end
        end
    end
    vals
end
function measure_xi(peps,env)
    vals=Dict{String,Any}()
    for (label,sector) in (("even",FermionParity(0)),("odd",FermionParity(1)))
        xh,xv,lh,lv=MPSKit.correlation_length(peps,env;sector,num_vals=3,tol=1e-11)
        vals[label*"_xi_x"]=only(xh);vals[label*"_xi_y"]=only(xv)
        for (dir,spectrum) in (("x",only(lh)),("y",only(lv)))
            vals[label*"_lambda_"*dir*"_real"]=real.(spectrum)
            vals[label*"_lambda_"*dir*"_imag"]=imag.(spectrum)
        end
    end
    vals
end
function main()
    w=parse(Float64,ARGS[1]);chi=parse(Int,ARGS[2]);seed=length(ARGS)>2 ? parse(Int,ARGS[3]) : 1730
    algorithm=length(ARGS)>3 ? Symbol(ARGS[4]) : :SequentialCTMRG
    adaptive=length(ARGS)>4 && ARGS[5]=="adaptive"
    trunc=adaptive ? truncrank(chi) : (;alg=:FixedSpaceTruncation)
    suffix=adaptive ? "_adaptive" : ""
    out="data/mortier_ctm_scan/w$(w)_chi$(chi)_seed$(seed)_$(algorithm)$(suffix)";mkpath(out)
    peps=ksvc_ipeps(;bond_weight=(w,w),bond_gauge=:balanced)
    rng=MersenneTwister(seed);draw(T,spaces...)=randn(rng,T,spaces...)
    env=PEPSKit.CTMRGEnv(draw,ComplexF64,peps,FermionicPEPS._environment_space((chi÷2,chi÷2)))
    history=Dict{String,Any}[];tolerance=1e-9;started=time()
    result=Dict{String,Any}("w"=>w,"chi"=>chi,"seed"=>seed,"algorithm"=>string(algorithm),
        "adaptive_sector_dimensions"=>adaptive,"projector"=>"FullInfiniteProjector","tolerance"=>tolerance,"r"=>collect(1:32),
        "xi_definition"=>"opposite CTM edge-pair transfer, trivial-sector normalization; one-site cell")
    local info
    for chunk in 1:5
        env,info=PEPSKit.leading_boundary(env,peps;tol=tolerance,maxiter=200,miniter=10,
            alg=algorithm,trunc,projector_alg=:FullInfiniteProjector,verbosity=2)
        values=measure(peps,env;check=chunk==1)
        push!(history,Dict("iteration_budget"=>200chunk,"residual"=>info.convergence_error,
            "converged"=>info.converged,"elapsed_seconds"=>time()-started,"observables"=>values))
        result["history"]=history;result["converged"]=info.converged
        result["residual"]=info.convergence_error;result["observables"]=values
        open(joinpath(out,"result.toml"),"w") do io;TOML.print(io,result);end
        serialize(joinpath(out,"environment.jls"),(;peps,environment=env,info,chi=(chi÷2,chi÷2),converged=info.converged))
        println("w=",w," chi=",chi," chunk=",chunk," residual=",info.convergence_error,
            " normal1=",complex(values["normal_x_real"][1],values["normal_x_imag"][1]));flush(stdout)
        info.converged && break
    end
    # An additional sweep block checks observable stability even after spectral convergence.
    before=result["observables"]
    env,info2=PEPSKit.leading_boundary(env,peps;tol=tolerance,maxiter=50,miniter=50,
        alg=algorithm,trunc,projector_alg=:FullInfiniteProjector,verbosity=2)
    after=measure(peps,env)
    result["postcheck_observable_drift"]=maximum(maximum(abs.(after[k].-before[k])) for k in keys(after))
    result["postcheck_residual"]=info2.convergence_error
    result["converged"]=info.converged && info2.converged
    result["observables"]=after
    result["xi"]=measure_xi(peps,env)
    result["actual_edge_dimensions"]=[dim(space(env.edges[d,1,1],1)) for d in 1:4]
    result["actual_edge_spaces"]=[string(space(env.edges[d,1,1],1)) for d in 1:4]
    result["elapsed_seconds"]=time()-started
    open(joinpath(out,"result.toml"),"w") do io;TOML.print(io,result);end
    serialize(joinpath(out,"environment.jls"),(;peps,environment=env,info=info2,chi=(chi÷2,chi÷2),converged=result["converged"]))
    println("FINAL ",out," converged=",result["converged"]," xi=",result["xi"]);flush(stdout)
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
