# Bounded D4 chi48 -> chi96 QR continuation using the validated GPU route.
# Checkpoints contain CPU arrays and can be read on either backend.
include(joinpath(@__DIR__, "..", "..", "src", "algorithms", "c4v_gs_cuda_support.jl"))
const STATE_PATH = abspath(ENV["TFIM_SOURCE_STATE"])
using MPSKit

const ROOT = abspath(ENV["TFIM_POINT_ROOT"])
const H = parse(Float64, ENV["TFIM_H"])
const BRANCH = ENV["TFIM_BRANCH"]
const SOURCE_HASH = ENV["TFIM_SOURCE_SHA256"]
const CODE_HASH = ENV["TFIM_CODE_SHA256"]
const JOB = get(ENV, "SLURM_JOB_ID", "manual")
const SEGMENT = parse(Int, get(ENV, "TFIM_SEGMENT_STEPS", "100"))
const RUN_SECONDS = parse(Float64, get(ENV, "TFIM_RUN_SECONDS", "6000"))
const TOL = 5e-5
const OPT_TOL = 4.5e-5
const MAX_STEPS = parse(Int, get(ENV, "TFIM_MAX_STEPS", "500"))
0 < MAX_STEPS <= 500 || error("expected 1..500 steps")
const HOST_NAME = readchomp(`hostname`)
const START_TIME = time()
const BOUNDARY = (;alg=:C4vCTMRG,projector_alg=:C4vQRProjector,tol=1e-9,maxiter=2400,miniter=1,verbosity=-1)
const GRADIENT = (;tol=2e-7,maxiter=200,solver_alg=(;alg=:GMRES),verbosity=1)
const CONFIG = (;format_version=1,D=4,h=H,branch=BRANCH,ctm_chi=96,source_sha256=SOURCE_HASH,
    code_sha256=CODE_HASH,ad_tolerance=TOL,optimizer_stop_tolerance=OPT_TOL,ctm_tolerance=1e-9,gradient_tolerance=2e-7,
    spatial_symmetry="rotatereflect",checkpoint_semantics="PEPS/environment restart; LBFGS memory restarts at segment boundary",maximum_iterations=MAX_STEPS)
const CONFIG_HASH = bytes2hex(SHA.sha256(codeunits(repr(CONFIG))))
const OP_CPU = tfim_hamiltonian(TransverseIsing2D(;J=1.0,h=H,unitcell=(1,1)))
const OP_GPU = adapt_operator(CUDA.CuArray, OP_CPU)
mkpath(ROOT)
bytes2hex(open(SHA.sha256, STATE_PATH)) == SOURCE_HASH || error("source SHA mismatch")
BRANCH in ("ordered","disordered") || error("invalid branch")
SEGMENT > 0 || error("invalid segment")

function save(path, x)
    tmp=path*".tmp."*JOB
    open(io->serialize(io,x),tmp,"w");mv(tmp,path;force=true)
end
function stage(name;kwargs...)
    atomic_csv(joinpath(ROOT,"status.csv"),(;stage=name,D=4,h=H,branch=BRANCH,ctm_chi=96,
        slurm_job_id=JOB,host=HOST_NAME,saved_at=string(Dates.now()),kwargs...))
    println("STAGE ",name," ",kwargs);flush(stdout)
end
function cpu_pair(p,e)
    return adapt_peps(Array,p),adapt_environment(Array,e)
end
function check_dims(p,e)
    all(t->eltype(t.data)==ComplexF64,p.A) || error("PEPS is not ComplexF64")
    all(t->dim(space(t,1))==96 && dim(space(t,2))==96,e.corners) || error("environment is not chi96")
end
function xi(p,e,name;chi=96)
    cp,ce=cpu_pair(p,e)
    xh,xv,_,_=MPSKit.correlation_length(cp,ce;num_vals=3,tol=1e-11)
    a,b=Float64(first(xh)),Float64(first(xv))
    isfinite(a)&&isfinite(b)&&a>0&&b>0 || error("invalid CTMRG xi")
    atomic_csv(joinpath(ROOT,name),(;D=4,h=H,branch=BRANCH,ctm_chi=chi,
        xi_horizontal=a,xi_vertical=b,source_sha256=SOURCE_HASH,method="CTMRG",slurm_job_id=JOB))
    return (;xi_horizontal=a,xi_vertical=b)
end
function refresh(p,e;tol=1e-9,maxiter=2400)
    ee,ii=PEPSKit.leading_boundary(e,p;alg=:C4vCTMRG,projector_alg=:C4vQRProjector,
        tol,maxiter,miniter=1,verbosity=-1)
    ii.converged || error("QR environment did not converge: $(ii.convergence_error)")
    return ee,Float64(ii.convergence_error)
end
function fg(p,e;backend="cuda",tight=false)
    to=backend=="cuda" ? CUDA.CuArray : Array
    pp=adapt_peps(to,PEPSKit.peps_normalize(adapt_peps(Array,p)))
    ee=adapt_environment(to,e)
    b=tight ? merge(BOUNDARY,(;tol=1e-10,maxiter=4800)) : BOUNDARY
    g=tight ? merge(GRADIENT,(;tol=2e-8,maxiter=400)) : GRADIENT
    Random.seed!(20260913);CUDA.synchronize()
    rp,re,en,inf=PEPSKit.fixedpoint(backend=="cuda" ? OP_GPU : OP_CPU,pp,deepcopy(ee);
        boundary_alg=b,gradient_alg=g,optimizer_alg=(;tol=TOL,maxiter=1,verbosity=1),
        symmetrization=PEPSKit.RotateReflect(),hasconverged=(args...)->false,shouldstop=(args...)->true)
    inf.fg_evaluations==1 || error("audit changed the state")
    cp,ce=cpu_pair(rp,re)
    norm(cp.A[1,1]-adapt_peps(Array,pp).A[1,1])<1e-12 || error("audit updated PEPS")
    grad=adapt_peps(Array,inf.last_gradient);PEPSKit.symmetrize!(grad,PEPSKit.RotateReflect())
    return (;peps=cp,environment=ce,energy=Float64(real(en)),gradient=grad,gn=norm(grad.A[1,1]))
end
function paired_audit(p,e,name;tight=false)
    stage(name)
    gpu=fg(p,e;backend="cuda",tight);cpu=fg(p,e;backend="cpu",tight)
    de=abs(cpu.energy-gpu.energy);dg=norm(cpu.gradient.A[1,1]-gpu.gradient.A[1,1])
    rg=dg/max(cpu.gn,eps());passed=de<=1e-8&&dg<=1e-7&&rg<=5e-3
    atomic_csv(joinpath(ROOT,name*".csv"),(;energy_abs_difference=de,gradient_abs_difference=dg,
        gradient_relative_difference=rg,cpu_gradient_norm=cpu.gn,gpu_gradient_norm=gpu.gn,
        passed,tight,source_sha256=SOURCE_HASH,slurm_job_id=JOB))
    passed || error("CPU/GPU audit failed")
    return cpu,gpu
end
function persist(p,e,en,gn,iterations,baseline;reason="accepted_step")
    cp,ce=cpu_pair(p,e);check_dims(cp,ce)
    payload=(;format_version=1,config=CONFIG,config_hash=CONFIG_HASH,peps=cp,environment=ce,
        energy=Float64(real(en)),projected_gradient_norm=Float64(gn),completed_iterations=iterations,
        baseline,source_sha256=SOURCE_HASH,slurm_job_id=JOB,host=HOST_NAME,saved_at=string(Dates.now()),reason)
    save(joinpath(ROOT,"iteration_checkpoint.jls"),payload)
    atomic_csv(joinpath(ROOT,"progress.csv"),(;D=4,h=H,branch=BRANCH,ctm_chi=96,iteration=iterations,
        energy=real(en),projected_gradient_norm=gn,converged=false,reason,slurm_job_id=JOB))
end

function main()
    isfile(joinpath(ROOT,"accepted.csv")) && (println("ALREADY_ACCEPTED");return)
    stage("source_check")
    srcpayload=open(deserialize,STATE_PATH);src=source_groundstate(srcpayload)
    src.D==4 && src.environment_chi==48 || error("expected original D4 chi48 source")
    src.converged || error("original source is not converged")
    srcpayload.config.branch==BRANCH && abs(srcpayload.config.h-H)<1e-12 || error("source point mismatch")
    checkpoint=joinpath(ROOT,"iteration_checkpoint.jls")
    peps=PEPSKit.peps_normalize(deepcopy(src.peps));env=nothing;done=0;baseline=nothing
    if isfile(checkpoint)
        ck=open(deserialize,checkpoint)
        ck.config_hash == CONFIG_HASH || error("checkpoint config mismatch")
        peps=ck.peps;env=ck.environment;done=ck.completed_iterations;baseline=ck.baseline
        oldhash=bytes2hex(open(SHA.sha256,checkpoint))
        mkpath(joinpath(ROOT,"restart_sources"))
        archived=joinpath(ROOT,"restart_sources",oldhash*".jls")
        isfile(archived)||Base.cp(checkpoint,archived)
        atomic_csv(joinpath(ROOT,"restart_"*JOB*".csv"),(;checkpoint_sha256=oldhash,completed_iterations=done,slurm_job_id=JOB))
        stage("resume_environment_refresh";completed_iterations=done)
        peps=adapt_peps(CUDA.CuArray,peps);env=adapt_environment(CUDA.CuArray,env)
        env,_=refresh(peps,env)
    else
        stage("source_ctm48_xi")
        xi(src.peps,src.environment,"ctm48_source_xi.csv";chi=48)
        atomic_csv(joinpath(ROOT,"input_normalization.csv"),(;source_norm=norm(src.peps.A[1,1]),
            normalized_norm=norm(peps.A[1,1]),relative_change=norm(peps.A[1,1]-src.peps.A[1,1])/norm(src.peps.A[1,1]),source_sha256=SOURCE_HASH))
        Random.seed!(20260913)
        env0=complex(PEPSKit.initialize_random_c4v_env(peps,ComplexSpace(96)))
        peps=adapt_peps(CUDA.CuArray,peps);env0=adapt_environment(CUDA.CuArray,env0)
        stage("bootstrap_gpu_qr")
        env,qi=PEPSKit.leading_boundary(env0,peps;alg=:C4vCTMRG,projector_alg=:C4vQRProjector,
            tol=1e-9,maxiter=300,miniter=1,verbosity=-1)
        fallback=!qi.converged
        if fallback
            stage("bootstrap_cpu_eigh_fallback")
            Random.seed!(20260913);cp=adapt_peps(Array,peps)
            ce=complex(PEPSKit.initialize_random_c4v_env(cp,ComplexSpace(96)))
            ce,ei=PEPSKit.leading_boundary(ce,cp;alg=:C4vCTMRG,projector_alg=:C4vEighProjector,
                tol=1e-9,maxiter=2400,miniter=4,verbosity=-1)
            ei.converged||error("bootstrap Eigh failed: $(ei.convergence_error)")
            env=adapt_environment(CUDA.CuArray,ce);env,qrerror=refresh(peps,env;maxiter=2400)
        else
            qrerror=Float64(qi.convergence_error)
        end
        check_dims(peps,env)
        atomic_csv(joinpath(ROOT,"bootstrap.csv"),(;source_chi=48,target_chi=96,fallback_used=fallback,
            qr_error=qrerror,converged=true,peps_unchanged=true,source_sha256=SOURCE_HASH))
        stage("ctm96_before_observables")
        cp,ce=cpu_pair(peps,env);baseline=tfim_peps_observables(cp,OP_CPU,ce)
        xi(cp,ce,"ctm96_before_xi.csv")
        atomic_csv(joinpath(ROOT,"observables_before.csv"),(;D=4,h=H,branch=BRANCH,ctm_chi=96,
            energy=baseline.energy,abs_z=baseline.abs_z,x=baseline.x,source_sha256=SOURCE_HASH))
        cpu,gpu=paired_audit(peps,env,"initial_cpu_gpu_audit")
        persist(peps,env,gpu.energy,gpu.gn,0,baseline;reason="bootstrap_validated")
    end
    check_dims(peps,env)
    offset=done;completed_ref=Ref(done);history=joinpath(ROOT,"optimization_history.csv")
    if !isfile(history)
        open(io->println(io,"iteration,energy,projected_gradient_norm,elapsed_seconds,slurm_job_id"),history,"w")
    end
    stage("optimizing";completed_iterations=done)
    function checkpoint_callback(state,cost,gradient,it)
        completed_ref[]=offset+it
        gn=projected_norm(gradient)
        persist(state[1],state[2],cost,gn,offset+it,baseline)
        open(io->println(io,"$(offset+it),$(real(cost)),$gn,$(time()-START_TIME),$JOB"),history,"a")
        println("ACCEPTED_STEP iteration=$(offset+it) energy=$(real(cost)) projected=$gn");flush(stdout)
        return state,cost,gradient
    end
    stop(state,cost,gradient,numfg,it,timespent)=offset+it>=MAX_STEPS||it>=SEGMENT||time()-START_TIME>=RUN_SECONDS||isfile(joinpath(ROOT,"stop_after_step"))
    p,e,en,info=PEPSKit.fixedpoint(OP_GPU,peps,env;boundary_alg=BOUNDARY,gradient_alg=GRADIENT,
        optimizer_alg=(;tol=TOL,maxiter=min(SEGMENT, MAX_STEPS-done),verbosity=2),symmetrization=PEPSKit.RotateReflect(),
        hasconverged=(s,c,g,n)->false,shouldstop=stop,finalize! = checkpoint_callback)
    steps=max(0,length(info.costs)-1);done+=steps;gn=projected_norm(info.last_gradient)
    persist(p,e,en,gn,done,baseline;reason="segment_endpoint")
    cp,ce=cpu_pair(p,e)
    save(joinpath(ROOT,"endpoint_"*JOB*"_"*string(done)*".jls"),(;config=CONFIG,config_hash=CONFIG_HASH,
        peps=cp,environment=ce,energy=Float64(real(en)),projected_gradient_norm=gn,completed_iterations=done,baseline,
        optimizer_info=merge(info,(;last_gradient=adapt_peps(Array,info.last_gradient))),slurm_job_id=JOB))
    if done<MAX_STEPS
        stage("needs_continuation";completed_iterations=done,projected_gradient_norm=gn)
        return
    end
    stage("endpoint_environment_refresh";completed_iterations=done)
    e,ctmerror=refresh(p,e;tol=1e-10,maxiter=4800)
    cpu,gpu=paired_audit(p,e,"endpoint_cpu_gpu_audit";tight=true)
    strictgn=max(cpu.gn,gpu.gn)
    persist(gpu.peps,gpu.environment,gpu.energy,strictgn,done,baseline;reason="strict_endpoint_audit")
    cp,ce=gpu.peps,gpu.environment
    ob=tfim_peps_observables(cp,OP_CPU,ce)
    abs(ob.energy-gpu.energy)<=1e-8 || error("independent energy disagrees")
    ob.energy-baseline.energy<=1e-8 || error("energy increased from fixed original PEPS at chi96")
    abs(ob.energy_imag)<=1e-9 || error("complex energy")
    stage("endpoint_ctm_xi";completed_iterations=done)
    cxi=xi(cp,ce,"ctm96_final_xi.csv")
    finalinfo=merge(info,(;last_gradient=gpu.gradient))
    gs=VariationalGroundState(cp,ce,OP_CPU,finalinfo,Float64(real(ob.energy)),Float64(strictgn),strictgn<=TOL,
        4,96,TOL,done,:c4v_gpu_continuation,(;source_sha256=SOURCE_HASH,completed_iterations=done,ctmerror))
    final=joinpath(ROOT,"warmup_state.jls")
    save(final,(;format_version=1,config=CONFIG,config_hash=CONFIG_HASH,selected_groundstate=gs,
        observables=ob,baseline,ctm_xi=cxi,cpu_audit_gradient=cpu.gradient,slurm_job_id=JOB))
    fh=bytes2hex(open(SHA.sha256,final))
    row=(;D=4,h=H,branch=BRANCH,ctm_chi=96,energy=ob.energy,energy_imag=ob.energy_imag,
        abs_z=ob.abs_z,z=ob.z,x=ob.x,projected_gradient_norm=strictgn,converged=strictgn<=TOL,
        completed_iterations=done,stopping_reason="step_limit",ctm_convergence_error=ctmerror,cpu_gpu_agreement=true,
        xi_horizontal=cxi.xi_horizontal,xi_vertical=cxi.xi_vertical,
        source_sha256=SOURCE_HASH,checkpoint_sha256=fh,code_sha256=CODE_HASH,config_hash=CONFIG_HASH,slurm_job_id=JOB)
    atomic_csv(joinpath(ROOT,"accepted.csv"),row)
    stage("budget_complete_validated";completed_iterations=done,projected_gradient_norm=strictgn)
    println("BUDGET_COMPLETE_VALIDATED $H $BRANCH $strictgn");flush(stdout)
end
try
    main()
catch err
    stage("failed";error=replace(sprint(showerror,err),'\n'=>' '))
    rethrow()
end
