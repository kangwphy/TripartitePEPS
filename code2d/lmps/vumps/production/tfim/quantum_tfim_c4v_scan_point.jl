using Dates
using LMPSVUMPS
using LinearAlgebra
using PEPSKit
using Printf
using Random
using Serialization
using SHA
using TensorKit

if VERSION >= v"1.12"
    pepskit_source = dirname(pathof(PEPSKit))
    for relative_source in (
            "algorithms/contractions/ctmrg/enlarge_corner.jl",
            "algorithms/contractions/ctmrg/enlarge_corner_column.jl",
            "algorithms/contractions/ctmrg/projector.jl",
            "algorithms/contractions/ctmrg/halfinf_env.jl",
            "algorithms/contractions/ctmrg/fullinf_env.jl",
            "algorithms/contractions/ctmrg/renormalize_corner.jl",
            "algorithms/contractions/ctmrg/renormalize_edge.jl",
            "algorithms/contractions/ctmrg/contract_site.jl",
            "algorithms/contractions/ctmrg/gaugefix.jl",
        )
        Base.include(PEPSKit, joinpath(pepskit_source, relative_source))
    end
end

const D_BOND = parse(Int, ENV["TFIM_D"])
const H_FIELD = parse(Float64, ENV["TFIM_H"])
const BRANCH = lowercase(strip(ENV["TFIM_BRANCH"]))
const CTM_CHI = parse(Int, ENV["TFIM_CTM_CHI"])
const SOURCE_STATE = abspath(ENV["TFIM_SOURCE_STATE"])
const OUTPUT_ROOT = abspath(ENV["TFIM_C4V_SCAN_ROOT"])
const AD_TOLERANCE = parse(Float64, get(ENV, "TFIM_AD_TOLERANCE", "3e-4"))
const TOTAL_MAXITER = parse(Int, get(ENV, "TFIM_AD_MAXITER", "220"))
const SEGMENT_MAXITER = parse(Int, get(ENV, "TFIM_AD_SEGMENT_MAXITER", "10"))
const CHECKPOINT_EVERY = parse(Int, get(ENV, "TFIM_CHECKPOINT_EVERY", "1"))
const CTM_TOLERANCE = parse(Float64, get(ENV, "TFIM_CTM_TOLERANCE", "1e-9"))
const CTM_MAXITER = parse(Int, get(ENV, "TFIM_CTM_MAXITER", "2400"))
const GRADIENT_TOLERANCE = parse(Float64, get(
    ENV, "TFIM_GRADIENT_TOLERANCE", "2e-7"))
const GRADIENT_MAXITER = parse(Int, get(ENV, "TFIM_GRADIENT_MAXITER", "200"))
const SOURCE_QR_MAXITER = parse(Int, get(ENV, "TFIM_SOURCE_QR_MAXITER", "300"))
const BASE_SEED = parse(Int, get(ENV, "TFIM_BASE_SEED", "20260906"))
const SPATIAL_SYMMETRIZATION = PEPSKit.RotateReflect()
const SCRIPT_SHA256 = bytes2hex(open(SHA.sha256, @__FILE__))

D_BOND > 0 || error("D must be positive")
BRANCH in ("ordered", "disordered") || error("invalid branch $BRANCH")
isfile(SOURCE_STATE) || error("missing source $SOURCE_STATE")
CTM_CHI > 0 || error("chi must be positive")
0 < SEGMENT_MAXITER <= TOTAL_MAXITER || error("invalid segment budget")

sha256_file(path) = bytes2hex(open(SHA.sha256, path))
config_digest(config) = bytes2hex(SHA.sha256(codeunits(repr(config))))

function atomic_serialize(path, payload)
    mkpath(dirname(path))
    temporary = path * ".tmp." * get(ENV, "SLURM_JOB_ID", "manual")
    open(io -> serialize(io, payload), temporary, "w")
    mv(temporary, path; force=true)
end

function atomic_row(path, row)
    mkpath(dirname(path))
    clean(value) = replace(replace(string(value), ',' => ';'), '\n' => '|')
    temporary = path * ".tmp." * get(ENV, "SLURM_JOB_ID", "manual")
    open(temporary, "w") do io
        println(io, join(string.(keys(row)), ','))
        println(io, join((clean(value) for value in values(row)), ','))
    end
    mv(temporary, path; force=true)
end

function source_groundstate(payload)
    hasproperty(payload, :selected_groundstate) && return payload.selected_groundstate
    hasproperty(payload, :groundstate) && return payload.groundstate
    error("source checkpoint contains no selected ground state")
end

function projected_norm(gradient)
    projected = PEPSKit.symmetrize!(deepcopy(gradient), SPATIAL_SYMMETRIZATION)
    return Float64(norm(projected.A[1, 1]))
end

tag = @sprintf("h%.8f_D%d_ctm%d", H_FIELD, D_BOND, CTM_CHI)
point_root = joinpath(OUTPUT_ROOT, "points", BRANCH, tag)
checkpoint_path = joinpath(point_root, "iteration_checkpoint.jls")
final_path = joinpath(point_root, "warmup_state.jls")
summary_path = joinpath(OUTPUT_ROOT, "summary", BRANCH, tag * ".csv")
source_hash = sha256_file(SOURCE_STATE)

# Runtime budgets are intentionally excluded from the identity hash so a
# preempted or under-budget run can be resumed with a larger budget.  All
# numerical choices, source lineage and the exact driver remain immutable.
identity_config = (;
    format_version=1, D=D_BOND, h=H_FIELD, branch=BRANCH, ctm_chi=CTM_CHI,
    source_state=SOURCE_STATE, source_sha256=source_hash,
    ad_tolerance=AD_TOLERANCE, ctm_tolerance=CTM_TOLERANCE,
    ctm_maxiter=CTM_MAXITER, gradient_tolerance=GRADIENT_TOLERANCE,
    gradient_maxiter=GRADIENT_MAXITER, source_qr_maxiter=SOURCE_QR_MAXITER,
    spatial_symmetry="rotatereflect", boundary_route="c4v_source_qr_refresh",
    script_sha256=SCRIPT_SHA256,
)
config_hash = config_digest(identity_config)
config = merge(identity_config, (;
    total_maxiter=TOTAL_MAXITER, segment_maxiter=SEGMENT_MAXITER,
    checkpoint_every=CHECKPOINT_EVERY,
))

if isfile(final_path)
    final = open(deserialize, final_path)
    hasproperty(final, :config_hash) && final.config_hash == config_hash ||
        error("stale final checkpoint $final_path")
    hasproperty(final, :selected_groundstate) && final.selected_groundstate.converged ||
        error("final checkpoint is not converged")
    isfile(summary_path) || error("final checkpoint exists without summary")
    println("restored completed C4v scan point $final_path")
    exit(0)
end

hamiltonian = tfim_hamiltonian(
    TransverseIsing2D(; J=1.0, h=H_FIELD, unitcell=(1, 1)))
peps = nothing
environment = nothing
completed_iterations = 0
source_h = NaN
source_initial_observables = nothing
bootstrap = (; route="checkpoint_resume", qr_converged=true, qr_error=0.0,
    fallback_used=false, eigh_converged=true, eigh_error=0.0)

if isfile(checkpoint_path)
    checkpoint = open(deserialize, checkpoint_path)
    hasproperty(checkpoint, :config_hash) && checkpoint.config_hash == config_hash ||
        error("checkpoint configuration mismatch $checkpoint_path")
    peps = checkpoint.peps
    environment = checkpoint.environment
    completed_iterations = Int(checkpoint.completed_iterations)
    source_h = Float64(checkpoint.source_h)
    source_initial_observables = checkpoint.source_initial_observables
    hasproperty(checkpoint, :bootstrap) && (bootstrap = checkpoint.bootstrap)
    @printf("resuming C4v scan checkpoint at iteration %d\n", completed_iterations)
else
    source_payload = open(deserialize, SOURCE_STATE)
    hasproperty(source_payload, :config) || error("source has no config provenance")
    hasproperty(source_payload, :config_hash) || error("source has no config hash")
    source = source_groundstate(source_payload)
    source.D == D_BOND || error("source D=$(source.D) does not match target D=$D_BOND")
    source.converged || error("source ground state is not converged")
    source.environment isa PEPSKit.CTMRGEnv || error("source has no CTMRG environment")
    source_payload.config.branch == BRANCH || error("source branch mismatch")
    source_payload.config.ctm_chi == CTM_CHI || error("source CTM chi mismatch")
    source_payload.config.spatial_symmetry == "rotatereflect" ||
        error("source is not a RotateReflect state")
    source_h = Float64(source_payload.config.h)
    peps = PEPSKit.peps_normalize(
        PEPSKit.symmetrize!(deepcopy(source.peps), SPATIAL_SYMMETRIZATION))
    environment0 = deepcopy(source.environment)
    environment_qr, qr_info = PEPSKit.leading_boundary(
        environment0, peps; alg=:C4vCTMRG, projector_alg=:C4vQRProjector,
        tol=CTM_TOLERANCE, maxiter=SOURCE_QR_MAXITER, miniter=1, verbosity=-1)
    if qr_info.converged
        environment = environment_qr
        bootstrap = (; route="source_qr_refresh", qr_converged=true,
            qr_error=Float64(qr_info.convergence_error), fallback_used=false,
            eigh_converged=true, eigh_error=0.0)
    else
        Random.seed!(BASE_SEED)
        environment_seed = complex(PEPSKit.initialize_random_c4v_env(
            peps, TensorKit.ComplexSpace(CTM_CHI)))
        environment_eigh, eigh_info = PEPSKit.leading_boundary(
            environment_seed, peps; alg=:C4vCTMRG,
            projector_alg=:C4vEighProjector, tol=CTM_TOLERANCE,
            maxiter=CTM_MAXITER, miniter=4, verbosity=-1)
        eigh_info.converged || error(
            "source Eigh fallback failed: $(eigh_info.convergence_error)")
        environment, qr_retry = PEPSKit.leading_boundary(
            environment_eigh, peps; alg=:C4vCTMRG,
            projector_alg=:C4vQRProjector, tol=CTM_TOLERANCE,
            maxiter=SOURCE_QR_MAXITER, miniter=1, verbosity=-1)
        qr_retry.converged || error(
            "source QR fallback handoff failed: $(qr_retry.convergence_error)")
        bootstrap = (; route="eigh_qr_fallback",
            qr_converged=qr_retry.converged,
            qr_error=Float64(qr_retry.convergence_error), fallback_used=true,
            eigh_converged=eigh_info.converged,
            eigh_error=Float64(eigh_info.convergence_error))
    end
    source_initial_observables = tfim_peps_observables(peps, hamiltonian, environment)
end

completed_iterations <= TOTAL_MAXITER || error("checkpoint exceeds current budget")
remaining = TOTAL_MAXITER - completed_iterations
remaining > 0 || error("iteration budget exhausted without convergence")
segment_iterations = min(SEGMENT_MAXITER, remaining)
boundary_alg = (; alg=:C4vCTMRG, projector_alg=:C4vQRProjector,
    tol=CTM_TOLERANCE, maxiter=CTM_MAXITER, miniter=1, verbosity=-1)
gradient_alg = (; tol=GRADIENT_TOLERANCE, maxiter=GRADIENT_MAXITER,
    solver_alg=(; alg=:GMRES), verbosity=1)
offset = completed_iterations

function projected_hasconverged(state, cost, gradient, raw_norm)
    return projected_norm(gradient) <= AD_TOLERANCE
end

function checkpoint_finalize(state, cost, gradient, numiter)
    absolute_iteration = offset + numiter
    effective_norm = projected_norm(gradient)
    if absolute_iteration % CHECKPOINT_EVERY == 0
        atomic_serialize(checkpoint_path, (;
            format_version=1, config, config_hash,
            completed_iterations=absolute_iteration,
            peps=state[1], environment=state[2], energy=Float64(real(cost)),
            projected_gradient_norm=effective_norm, bootstrap, source_h,
            source_initial_observables,
            slurm_job_id=get(ENV, "SLURM_JOB_ID", "manual"),
            saved_at=string(Dates.now()),
        ))
    end
    return state, cost, gradient
end

seconds = @elapsed peps, environment, energy, info = PEPSKit.fixedpoint(
    hamiltonian, peps, environment;
    boundary_alg, gradient_alg,
    optimizer_alg=(; tol=AD_TOLERANCE, maxiter=segment_iterations, verbosity=1),
    symmetrization=SPATIAL_SYMMETRIZATION,
    hasconverged=projected_hasconverged,
    finalize! = checkpoint_finalize,
)
iterations_taken = max(0, length(info.costs) - 1)
completed_iterations += iterations_taken
effective_norm = projected_norm(info.last_gradient)
raw_norm = isempty(info.gradnorms) ? Inf : Float64(last(info.gradnorms))
atomic_serialize(checkpoint_path, (;
    format_version=1, config, config_hash, completed_iterations,
    peps, environment, energy=Float64(real(energy)),
    projected_gradient_norm=effective_norm, raw_gradient_norm=raw_norm,
    bootstrap, source_h, source_initial_observables,
    slurm_job_id=get(ENV, "SLURM_JOB_ID", "manual"),
    saved_at=string(Dates.now()),
))

if effective_norm > AD_TOLERANCE && completed_iterations < TOTAL_MAXITER
    @printf("C4v scan segment incomplete iterations=%d projected=%.6e raw=%.6e seconds=%.1f\n",
        completed_iterations, effective_norm, raw_norm, seconds)
    exit(42)
end
if effective_norm > AD_TOLERANCE
    error(@sprintf(
        "C4v scan budget exhausted without convergence: iterations=%d projected=%.6e tolerance=%.6e",
        completed_iterations, effective_norm, AD_TOLERANCE))
end

gs = VariationalGroundState(
    peps, environment, hamiltonian, info, Float64(real(energy)), effective_norm,
    true, D_BOND, CTM_CHI, AD_TOLERANCE, TOTAL_MAXITER,
    :c4v_branch_continuation,
    (; source=SOURCE_STATE, source_sha256=source_hash, source_h,
        completed_iterations, segment_iterations, seconds, bootstrap),
)
observables = tfim_groundstate_observables(gs)
energy_delta = observables.energy - source_initial_observables.energy
energy_delta <= 1e-8 || error(@sprintf(
    "optimized target energy increased from source: delta=%.6e", energy_delta))
row = (;
    format_version=1, stage="c4v_scan_point_complete",
    D=D_BOND, h=H_FIELD, branch=BRANCH, ctm_chi=CTM_CHI,
    spatial_symmetry="rotatereflect", boundary_route=bootstrap.route,
    source_state=SOURCE_STATE, source_sha256=source_hash, source_h,
    state_file=final_path, energy=observables.energy,
    energy_imag=observables.energy_imag, x=observables.x, z=observables.z,
    abs_z=observables.abs_z,
    source_initial_energy_at_target_h=source_initial_observables.energy,
    energy_delta_from_source_at_target_h=energy_delta,
    projected_gradient_norm=effective_norm, raw_gradient_norm=raw_norm,
    converged=true, completed_iterations, total_maxiter=TOTAL_MAXITER,
    ad_tolerance=AD_TOLERANCE, config_hash, script_sha256=SCRIPT_SHA256,
    slurm_job_id=get(ENV, "SLURM_JOB_ID", "manual"),
)
atomic_serialize(final_path, (;
    format_version=1, config, config_hash, selected_groundstate=gs,
    observables, source_initial_observables, summary_row=row,
))
atomic_row(summary_path, row)
@printf("C4v scan complete D=%d h=%.8f branch=%s source_h=%.8f iterations=%d E=%.12f projected=%.6e\n",
    D_BOND, H_FIELD, BRANCH, source_h, completed_iterations,
    observables.energy, effective_norm)
