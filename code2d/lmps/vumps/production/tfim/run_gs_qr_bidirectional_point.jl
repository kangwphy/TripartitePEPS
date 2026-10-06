# Generic opt-in CUDA QR point optimizer. CPU entry points remain unchanged.
include(joinpath(@__DIR__, "..", "..", "src", "algorithms", "c4v_gs_cuda_support.jl"))
using MPSKit

const STATE_PATH = abspath(ENV["TFIM_SOURCE_STATE"])
const SOURCE_HASH = ENV["TFIM_SOURCE_SHA256"]
const ROOT = abspath(ENV["TFIM_POINT_ROOT"])
const H = parse(Float64, ENV["TFIM_H"])
const BRANCH = ENV["TFIM_BRANCH"]
const D = parse(Int, get(ENV, "TFIM_D", "3"))
const CHI = parse(Int, get(ENV, "TFIM_CTM_CHI", "48"))
const TOL = parse(Float64, get(ENV, "TFIM_AD_TOLERANCE", "1e-5"))
const MAX_STEPS = parse(Int, get(ENV, "TFIM_MAX_STEPS", "1000"))
const SEGMENT = parse(Int, get(ENV, "TFIM_SEGMENT_STEPS", "1000"))
const RUN_SECONDS = parse(Float64, get(ENV, "TFIM_RUN_SECONDS", "6000"))
const STOP_FILE = abspath(get(ENV, "TFIM_STOP_FILE", joinpath(ROOT, "stop_after_step")))
const CODE_HASH = ENV["TFIM_CODE_SHA256"]
const BASE_COMMIT = get(ENV, "TFIM_BASE_GIT_COMMIT", "unspecified")
const JOB = get(ENV, "SLURM_JOB_ID", "manual")
const HOST_NAME = readchomp(`hostname`)
const BASE_SEED = parse(Int, get(ENV, "TFIM_BASE_SEED", "20261006"))
const CTM_TOL = parse(Float64, get(ENV, "TFIM_CTM_TOLERANCE", "1e-9"))
const GRAD_TOL = parse(Float64, get(ENV, "TFIM_GRADIENT_TOLERANCE", "2e-7"))
const START_TIME = time()
const BOUNDARY = (; alg=:C4vCTMRG, projector_alg=:C4vQRProjector,
    tol=CTM_TOL, maxiter=2400, miniter=1, verbosity=-1)
const GRADIENT = (; tol=GRAD_TOL, maxiter=200, solver_alg=(;alg=:GMRES), verbosity=1)
const TIGHT_BOUNDARY = merge(BOUNDARY, (;tol=min(CTM_TOL, 1e-10), maxiter=4800))
const TIGHT_GRADIENT = merge(GRADIENT, (;tol=min(GRAD_TOL, 2e-8), maxiter=400))
const CONFIG = (; format_version=1, D, h=H, branch=BRANCH, ctm_chi=CHI,
    source_sha256=SOURCE_HASH, code_sha256=CODE_HASH, base_git_commit=BASE_COMMIT,
    ad_tolerance=TOL, maximum_iterations=MAX_STEPS, ctm_tolerance=CTM_TOL,
    gradient_tolerance=GRAD_TOL, base_seed=BASE_SEED,
    spatial_symmetry="rotatereflect", optimization_projector="C4vQRProjector",
    checkpoint_semantics="PEPS/environment restart; LBFGS memory restarts at segment boundary")
const CONFIG_HASH = bytes2hex(SHA.sha256(codeunits(repr(CONFIG))))
const OP_CPU = tfim_hamiltonian(TransverseIsing2D(;J=1.0, h=H, unitcell=(1,1)))
const OP_GPU = adapt_operator(CUDA.CuArray, OP_CPU)

D > 0 && CHI > 0 || error("D and chi must be positive")
isfinite(H) && isfinite(TOL) && TOL > 0 || error("invalid h/gradient tolerance")
0 < MAX_STEPS <= 1000 && SEGMENT > 0 || error("invalid accepted-update budget")
isfinite(RUN_SECONDS) && RUN_SECONDS > 0 || error("invalid runtime budget")
CTM_TOL > 0 && GRAD_TOL > 0 || error("invalid solver tolerance")
mkpath(ROOT)
sha_file(path) = bytes2hex(open(SHA.sha256, path))
sha_file(STATE_PATH) == SOURCE_HASH || error("input source SHA mismatch")

function save_atomic(path, x)
    tmp=path*".tmp."*JOB
    open(io->serialize(io,x),tmp,"w")
    mv(tmp,path;force=true)
end
function stage(name; kwargs...)
    atomic_csv(joinpath(ROOT,"status.csv"), (;stage=name,D,h=H,branch=BRANCH,
        ctm_chi=CHI,slurm_job_id=JOB,host=HOST_NAME,saved_at=string(Dates.now()),kwargs...))
    println("STAGE ",name," ",kwargs)
    flush(stdout)
end
cpu_pair(p,e) = (adapt_peps(Array,p), adapt_environment(Array,e))
function check_peps(p)
    size(p)==(1,1) || error("expected a one-site PEPS")
    t=p.A[1,1]
    dim(space(t,1))==2 || error("physical space is not spin one-half")
    all(i->dim(space(t,i))==D,2:5) || error("actual PEPS virtual D mismatch")
    eltype(t.data)==ComplexF64 || error("expected ComplexF64 PEPS")
    all(isfinite,t.data) && isfinite(norm(t)) && norm(t)>0 || error("invalid PEPS tensor")
end
function same_chi(e)
    e isa PEPSKit.CTMRGEnv || return false
    return all(t->dim(space(t,1))==CHI && dim(space(t,2))==CHI,e.corners)
end
function check_pair(p,e)
    cp,ce=cpu_pair(p,e)
    check_peps(cp)
    same_chi(ce) || error("environment chi mismatch")
    all(t->all(isfinite,t.data),ce.corners) || error("nonfinite CTM corners")
    all(t->all(isfinite,t.data),ce.edges) || error("nonfinite CTM edges")
    return cp,ce
end
function refresh(p,e; tight=false, maxiter=nothing)
    b=tight ? TIGHT_BOUNDARY : BOUNDARY
    !isnothing(maxiter) && (b=merge(b,(;maxiter)))
    ee,ii=PEPSKit.leading_boundary(e,p; b...)
    ii.converged || error("QR environment failed: $(ii.convergence_error)")
    err=Float64(ii.convergence_error)
    isfinite(err) || error("nonfinite CTM convergence error")
    return ee,err
end
function fg(p,e; backend="cuda",tight=false)
    to=backend=="cuda" ? CUDA.CuArray : Array
    pp=adapt_peps(to,PEPSKit.peps_normalize(adapt_peps(Array,p)))
    ee=adapt_environment(to,e)
    b=tight ? TIGHT_BOUNDARY : BOUNDARY
    g=tight ? TIGHT_GRADIENT : GRADIENT
    Random.seed!(BASE_SEED)
    CUDA.synchronize()
    rp,re,en,inf=PEPSKit.fixedpoint(backend=="cuda" ? OP_GPU : OP_CPU,pp,deepcopy(ee);
        boundary_alg=b,gradient_alg=g,optimizer_alg=(;tol=TOL,maxiter=1,verbosity=1),
        symmetrization=PEPSKit.RotateReflect(),hasconverged=(args...)->false,
        shouldstop=(args...)->true)
    inf.fg_evaluations==1 || error("audit changed PEPS")
    cp,ce=check_pair(rp,re)
    norm(cp.A[1,1]-adapt_peps(Array,pp).A[1,1])<1e-12 || error("audit updated PEPS")
    grad=adapt_peps(Array,inf.last_gradient)
    PEPSKit.symmetrize!(grad,PEPSKit.RotateReflect())
    gn=Float64(norm(grad.A[1,1]))
    all(isfinite,grad.A[1,1].data) && isfinite(gn) && isfinite(real(en)) ||
        error("nonfinite energy/gradient in audit")
    abs(imag(en))<=1e-9 || error("complex audit energy")
    return (;peps=cp,environment=ce,energy=Float64(real(en)),gradient=grad,gn)
end
function paired_audit(p,e,name; tight=false)
    stage(name)
    gpu=fg(p,e;backend="cuda",tight)
    cpu=fg(p,e;backend="cpu",tight)
    de=abs(cpu.energy-gpu.energy)
    dg=Float64(norm(cpu.gradient.A[1,1]-gpu.gradient.A[1,1]))
    rg=dg/max(cpu.gn,eps())
    passed=de<=1e-8 && dg<=1e-7 && rg<=5e-3
    row=(;energy_abs_difference=de,gradient_abs_difference=dg,
        gradient_relative_difference=rg,cpu_gradient_norm=cpu.gn,gpu_gradient_norm=gpu.gn,
        passed,tight,source_sha256=SOURCE_HASH,slurm_job_id=JOB)
    atomic_csv(joinpath(ROOT,name*".csv"),row)
    atomic_csv(joinpath(ROOT,name*"_"*JOB*".csv"),row)
    passed || error("mandatory CPU/GPU audit failed")
    return cpu,gpu
end
function persist(p,e,en,gn,done,baseline,source_h,strict_mode,stop_tol;reason)
    cp,ce=check_pair(p,e)
    isfinite(real(en)) && isfinite(gn) && gn>=0 || error("nonfinite optimization endpoint")
    0<=done<=MAX_STEPS || error("accepted-update budget exceeded")
    payload=(;format_version=1,config=CONFIG,config_hash=CONFIG_HASH,peps=cp,environment=ce,
        energy=Float64(real(en)),projected_gradient_norm=Float64(gn),completed_iterations=done,
        baseline,source_h,source_state=STATE_PATH,source_sha256=SOURCE_HASH,
        strict_mode,optimizer_stop_tolerance=stop_tol,slurm_job_id=JOB,
        host=HOST_NAME,saved_at=string(Dates.now()),elapsed_seconds=time()-START_TIME,reason)
    save_atomic(joinpath(ROOT,"iteration_checkpoint.jls"),payload)
    atomic_csv(joinpath(ROOT,"progress.csv"),(;D,h=H,branch=BRANCH,ctm_chi=CHI,
        iteration=done,completed_iterations=done,energy=real(en),projected_gradient_norm=gn,
        converged=false,reason,strict_mode,optimizer_stop_tolerance=stop_tol,slurm_job_id=JOB))
    return payload
end
function saved_source_h(payload)
    for container in (payload, hasproperty(payload,:config) ? payload.config : nothing,
            hasproperty(payload,:common) ? payload.common : nothing)
        !isnothing(container) && hasproperty(container,:h) && return Float64(container.h)
    end
    haskey(ENV,"TFIM_SOURCE_H") && return parse(Float64,ENV["TFIM_SOURCE_H"])
    error("input lacks actual source h; supply source metadata or TFIM_SOURCE_H")
end
json_string(s)= "\""*replace(string(s),'\\'=>"\\\\",'"'=>"\\\"",'\n'=>"\\n",'\r'=>"\\r",'\t'=>"\\t")*"\""
function write_result(row)
    path=joinpath(ROOT,"point_result.json");tmp=path*".tmp."*JOB
    encode(x)=x isa Bool ? string(x) : x isa Number ? string(x) : json_string(x)
    open(tmp,"w") do io
        println(io,"{")
        kv=collect(pairs(row))
        for (i,(k,v)) in enumerate(kv)
            println(io,"  ",json_string(k),": ",encode(v),i==length(kv) ? "" : ",")
        end
        println(io,"}")
    end
    mv(tmp,path;force=true)
end
function publish_terminal(final_payload)
    final=joinpath(ROOT,"warmup_state.jls")
    final_payload.config_hash==CONFIG_HASH && final_payload.validated_endpoint ||
        error("stale or unvalidated terminal checkpoint")
    check_pair(final_payload.selected_groundstate.peps,final_payload.selected_groundstate.environment)
    sha_file(STATE_PATH)==SOURCE_HASH || error("source changed during optimization")
    fh=sha_file(final)
    row=merge(final_payload.endpoint_summary,(;checkpoint_sha256=fh))
    atomic_csv(joinpath(ROOT,"final_summary.csv"),row)
    atomic_csv(joinpath(ROOT,"accepted.csv"),row)
    write_result((;terminal=true,D,h=H,branch=BRANCH,ctm_chi=CHI,
        stopping_reason=row.stopping_reason,completed_iterations=row.completed_iterations,
        converged=row.converged,projected_gradient_norm=row.projected_gradient_norm,
        checkpoint=final,checkpoint_sha256=fh,source_state=STATE_PATH,
        source_sha256=SOURCE_HASH,code_sha256=CODE_HASH,config_hash=CONFIG_HASH,
        cpu_gpu_agreement=true,ctm_converged=true,slurm_job_id=JOB))
    stage("terminal_validated";completed_iterations=row.completed_iterations,
        stopping_reason=row.stopping_reason,converged=row.converged)
end
interrupted() = time()-START_TIME>=RUN_SECONDS || isfile(STOP_FILE)

function main()
    final=joinpath(ROOT,"warmup_state.jls")
    if isfile(final)
        payload=open(deserialize,final)
        publish_terminal(payload)
        return 0
    end
    (isfile(joinpath(ROOT,"accepted.csv")) || isfile(joinpath(ROOT,"point_result.json"))) &&
        error("terminal marker without final checkpoint")
    stage("source_check")
    srcpayload=open(deserialize,STATE_PATH)
    src=source_groundstate(srcpayload)
    hasproperty(src,:peps) || error("source contains no PEPS")
    check_peps(adapt_peps(Array,src.peps))
    hasproperty(src,:D) && src.D!=D && error("source D metadata mismatch")
    if src isa VariationalGroundState && !src.converged
        (hasproperty(srcpayload,:validated_endpoint) && srcpayload.validated_endpoint) ||
            error("unconverged GS source is not a validated budget endpoint")
    end
    if hasproperty(srcpayload,:reason) && hasproperty(srcpayload,:completed_iterations) &&
            hasproperty(srcpayload,:projected_gradient_norm) &&
            !(hasproperty(srcpayload,:validated_endpoint) && srcpayload.validated_endpoint)
        error("nonterminal iteration checkpoint cannot seed the next h")
    end
    source_h=saved_source_h(srcpayload)
    isfinite(source_h) || error("nonfinite source h")
    checkpoint=joinpath(ROOT,"iteration_checkpoint.jls")
    p=PEPSKit.peps_normalize(PEPSKit.symmetrize!(deepcopy(adapt_peps(Array,src.peps)),PEPSKit.RotateReflect()))
    e=nothing;done=0;baseline=nothing;strict_mode=false;stop_tol=TOL;resume_record=nothing
    if isfile(checkpoint)
        ck=open(deserialize,checkpoint)
        ck.config_hash==CONFIG_HASH || error("resume checkpoint configuration mismatch")
        ck.source_sha256==SOURCE_HASH || error("resume source SHA mismatch")
        p=ck.peps;e=ck.environment;done=Int(ck.completed_iterations);baseline=ck.baseline
        resume_record=ck
        source_h=Float64(ck.source_h);strict_mode=ck.strict_mode;stop_tol=ck.optimizer_stop_tolerance
        0<=done<=MAX_STEPS || error("resume exceeds update budget")
        check_pair(p,e)
        stage("resume_environment_refresh";completed_iterations=done)
        p=adapt_peps(CUDA.CuArray,p);e=adapt_environment(CUDA.CuArray,e)
        e,err=refresh(p,e;tight=strict_mode)
        atomic_csv(joinpath(ROOT,"resume_"*JOB*".csv"),(;completed_iterations=done,
            ctm_converged=true,ctm_convergence_error=err,source_sha256=SOURCE_HASH))
    else
        reused=hasproperty(src,:environment) && same_chi(src.environment)
        Random.seed!(BASE_SEED)
        e0=reused ? deepcopy(src.environment) :
            complex(PEPSKit.initialize_random_c4v_env(p,ComplexSpace(CHI)))
        p=adapt_peps(CUDA.CuArray,p);e0=adapt_environment(CUDA.CuArray,e0)
        stage("bootstrap_gpu_qr";source_environment_reused=reused)
        e,qi=PEPSKit.leading_boundary(e0,p;merge(BOUNDARY,(;maxiter=300))...)
        fallback=!qi.converged
        if fallback
            stage("bootstrap_cpu_eigh_fallback")
            cp=adapt_peps(Array,p);Random.seed!(BASE_SEED)
            ce=complex(PEPSKit.initialize_random_c4v_env(cp,ComplexSpace(CHI)))
            ce,ei=PEPSKit.leading_boundary(ce,cp;alg=:C4vCTMRG,
                projector_alg=:C4vEighProjector,tol=CTM_TOL,maxiter=2400,miniter=4,verbosity=-1)
            ei.converged || error("bootstrap Eigh fallback failed")
            e,err=refresh(p,adapt_environment(CUDA.CuArray,ce))
        else
            err=Float64(qi.convergence_error)
        end
        check_pair(p,e)
        atomic_csv(joinpath(ROOT,"bootstrap.csv"),(;D,h=H,branch=BRANCH,ctm_chi=CHI,
            source_h,source_environment_reused=reused,eigh_fallback_used=fallback,
            ctm_converged=true,ctm_convergence_error=err,ctm_tolerance=CTM_TOL,
            optimization_projector="C4vQRProjector",source_sha256=SOURCE_HASH))
    end
    cpu,gpu=paired_audit(p,e,"initial_cpu_gpu_audit";tight=strict_mode)
    p=adapt_peps(CUDA.CuArray,gpu.peps);e=adapt_environment(CUDA.CuArray,gpu.environment)
    if isnothing(baseline)
        baseline=tfim_peps_observables(cpu.peps,OP_CPU,cpu.environment)
        atomic_csv(joinpath(ROOT,"observables_before.csv"),(;D,h=H,branch=BRANCH,
            ctm_chi=CHI,source_h,energy=baseline.energy,x=baseline.x,z=baseline.z,
            abs_z=baseline.abs_z,projected_gradient_norm=max(cpu.gn,gpu.gn),
            source_sha256=SOURCE_HASH))
    end
    persist(p,e,gpu.energy,gpu.gn,done,baseline,source_h,strict_mode,stop_tol;
        reason="initial_or_resume_validated")
    history=joinpath(ROOT,"optimization_history.csv")
    if !isfile(history)
        open(io->println(io,"iteration,energy,projected_gradient_norm,elapsed_seconds,slurm_job_id"),history,"w")
    end
    if !isnothing(resume_record) && done>0
        lines=readlines(history)
        last_written=length(lines)>1 ? parse(Int,first(split(last(lines),','))) : 0
        last_written<=done || error("history is ahead of persistent checkpoint")
        if last_written<done
            # Checkpoint is saved before append. Recover its exact last update
            # if preemption occurred between the two writes; never invent gaps.
            done-last_written==1 && resume_record.reason=="accepted_step" ||
                error("optimization history has an unrecoverable gap")
            open(history,"a") do io
                println(io,"$done,$(resume_record.energy),$(resume_record.projected_gradient_norm),$(resume_record.elapsed_seconds),$(resume_record.slurm_job_id)")
            end
            stage("recovered_last_history_row";completed_iterations=done)
        end
    end
    segment_start=done
    zero_refinements=0
    last_info=nothing
    while true
        if done<MAX_STEPS && (interrupted() || done-segment_start>=SEGMENT)
            stage("needs_continuation";completed_iterations=done)
            return 42
        end
        if done<MAX_STEPS
            offset=done;accepted=Ref(done)
            b=strict_mode ? TIGHT_BOUNDARY : BOUNDARY
            g=strict_mode ? TIGHT_GRADIENT : GRADIENT
            function checkpoint_callback(state,cost,gradient,it)
                absolute_iteration=offset+it
                0<=absolute_iteration<=MAX_STEPS || error("callback exceeded step budget")
                gn=projected_norm(gradient)
                if absolute_iteration>accepted[]
                    record=persist(state[1],state[2],cost,gn,absolute_iteration,baseline,
                        source_h,strict_mode,stop_tol;reason="accepted_step")
                    open(history,"a") do io
                        println(io,"$absolute_iteration,$(record.energy),$gn,$(record.elapsed_seconds),$JOB")
                    end
                    accepted[]=absolute_iteration
                end
                return state,cost,gradient
            end
            stop(s,c,gr,numfg,it,timespent)=offset+it>=MAX_STEPS ||
                offset+it-segment_start>=SEGMENT || interrupted()
            stage("optimizing";completed_iterations=done,strict_mode,
                optimizer_stop_tolerance=stop_tol)
            p,e,en,info=PEPSKit.fixedpoint(OP_GPU,p,e;boundary_alg=b,gradient_alg=g,
                optimizer_alg=(;tol=stop_tol,maxiter=min(MAX_STEPS-done,SEGMENT-(done-segment_start)),verbosity=2),
                symmetrization=PEPSKit.RotateReflect(),
                hasconverged=(s,c,gr,n)->projected_norm(gr)<=stop_tol,
                shouldstop=stop,finalize! = checkpoint_callback)
            steps=max(0,length(info.costs)-1)
            done=offset+steps
            accepted[]==done || error("optimizer/callback accepted-update count disagrees")
            last_info=info
            gn=projected_norm(info.last_gradient)
            persist(p,e,en,gn,done,baseline,source_h,strict_mode,stop_tol;reason="segment_endpoint")
            if done<MAX_STEPS && gn>stop_tol &&
                    (interrupted() || done-segment_start>=SEGMENT)
                stage("needs_continuation";completed_iterations=done,projected_gradient_norm=gn)
                return 42
            end
        else
            steps=0
        end
        # A loose-gradient stop is provisional until this tight paired audit.
        stage("endpoint_environment_refresh";completed_iterations=done)
        e,_=refresh(p,e;tight=true)
        cpu,gpu=paired_audit(p,e,"endpoint_cpu_gpu_audit";tight=true)
        strictgn=max(cpu.gn,gpu.gn)
        if strictgn>TOL && done<MAX_STEPS
            zero_refinements=steps==0 ? zero_refinements+1 : 0
            zero_refinements<=3 || error("tight refinement repeatedly made zero updates")
            strict_mode=true
            stop_tol=min(stop_tol,0.9*TOL)
            p=adapt_peps(CUDA.CuArray,gpu.peps);e=adapt_environment(CUDA.CuArray,gpu.environment)
            persist(p,e,gpu.energy,strictgn,done,baseline,source_h,strict_mode,stop_tol;
                reason="tight_audit_requires_refinement")
            stage("tight_refinement_required";completed_iterations=done,
                projected_gradient_norm=strictgn,optimizer_stop_tolerance=stop_tol)
            continue
        end
        cp=cpu.peps
        # This is the actual saved CPU CTM environment and its own error.
        ce,ctmerror=refresh(cp,cpu.environment;tight=true)
        check_pair(cp,ce)
        ob=tfim_peps_observables(cp,OP_CPU,ce)
        isfinite(ob.energy) && isfinite(ob.x) && isfinite(ob.z) || error("nonfinite observables")
        abs(ob.energy-gpu.energy)<=1e-8 && abs(ob.energy-cpu.energy)<=1e-8 ||
            error("independent final energy disagrees with paired audit")
        abs(ob.energy_imag)<=1e-9 || error("complex final energy")
        ob.energy-baseline.energy<=1e-8 || error("energy increased from input at target h")
        xh,xv,_,_=MPSKit.correlation_length(cp,ce;num_vals=3,tol=1e-11)
        a,b=Float64(first(xh)),Float64(first(xv))
        all(x->isfinite(x)&&x>0,(a,b)) || error("invalid CTM correlation length")
        cxi=(;xi_horizontal=a,xi_vertical=b)
        converged=strictgn<=TOL
        reason=converged ? "gradient_converged" : "step_limit"
        (!converged && done!=MAX_STEPS) && error("invalid terminal stopping condition")
        info_cpu=isnothing(last_info) ? (;last_gradient=cpu.gradient) :
            merge(last_info,(;last_gradient=cpu.gradient))
        gs=VariationalGroundState(cp,ce,OP_CPU,info_cpu,Float64(real(ob.energy)),Float64(strictgn),
            converged,D,CHI,TOL,done,:c4v_gpu_bidirectional,
            (;source_state=STATE_PATH,source_sha256=SOURCE_HASH,source_h,
                completed_iterations=done,stopping_reason=reason,ctm_convergence_error=ctmerror))
        row=(;D,h=H,branch=BRANCH,ctm_chi=CHI,source_h,energy=ob.energy,
            energy_imag=ob.energy_imag,x=ob.x,z=ob.z,abs_z=ob.abs_z,
            projected_gradient_norm=strictgn,converged,completed_iterations=done,
            maximum_iterations=MAX_STEPS,ad_tolerance=TOL,stopping_reason=reason,
            cpu_gpu_agreement=true,ctm_converged=true,
            ctm_convergence_error=ctmerror,ctm_tolerance=TIGHT_BOUNDARY.tol,
            ctm_maxiter=TIGHT_BOUNDARY.maxiter,ctm_projector="C4vQRProjector",
            ctm_measurement_backend="cpu",ctm_error_scope="saved_final_CPU_QR_refresh",
            xi_horizontal=a,xi_vertical=b,qr_discarded_weight_available=false,
            qr_discarded_weight=missing,source_sha256=SOURCE_HASH,code_sha256=CODE_HASH,
            base_git_commit=BASE_COMMIT,config_hash=CONFIG_HASH,slurm_job_id=JOB)
        payload=(;format_version=1,config=CONFIG,config_hash=CONFIG_HASH,selected_groundstate=gs,
            validated_endpoint=true,observables=ob,baseline,ctm_xi=cxi,
            ctm_converged=true,ctm_tolerance=TIGHT_BOUNDARY.tol,ctm_convergence_error=ctmerror,
            ctm_maxiter=TIGHT_BOUNDARY.maxiter,ctm_projector="C4vQRProjector",
            ctm_measurement_backend="cpu",ctm_error_scope="saved_final_CPU_QR_refresh",
            endpoint_summary=row,completed_iterations=done,stopping_reason=reason,
            source_state=STATE_PATH,source_sha256=SOURCE_HASH,source_h,
            cpu_audit_gradient=cpu.gradient,slurm_job_id=JOB,saved_at=string(Dates.now()))
        sha_file(STATE_PATH)==SOURCE_HASH || error("source changed during optimization")
        save_atomic(final,payload)
        atomic_csv(joinpath(ROOT,"ctm_final_xi.csv"),merge(row,(;checkpoint_sha256=sha_file(final))))
        publish_terminal(payload)
        return 0
    end
end

try
    exit(main())
catch err
    stage("failed";error=replace(replace(sprint(showerror,err),'\n'=>' '),','=>';'))
    rethrow()
end
