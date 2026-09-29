# Opt-in GPU GS support; never loaded by the CPU module.
using Adapt
using CUDA
using Dates
using LMPSVUMPS
using LinearAlgebra
using PEPSKit
using Printf
using Random
using Serialization
using SHA
using TensorKit
using cuTENSOR

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

CUDA.functional() || error("CUDA.functional() is false")
cuTENSOR.functional() || error("cuTENSOR.functional() is false")
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "cuda_qr_pullback_compat.jl"))
# PEPSKit 0.8.1 constructs the random auxiliary MPS and mixed-transfer
# fixed-point seeds on the CPU inside the C4v gauge-fixing path.  Forward CUDA
# CTMRG does not visit this routine, but implicit differentiation does.  This
# small gauge calculation is outside the differentiated CTMRG iteration, so do
# it coherently on CPU and transfer only the resulting fixed gauge to CUDA.
@eval PEPSKit function compute_relative_phases(
        envfinal::CTMRGEnv{C, T}, envprev::CTMRGEnv{C, T},
        ::ScramblingEnvGaugeC4v) where {C, T}
    Tprev = Main.Adapt.adapt(Array, envprev.edges[1, 1, 1])
    Tfinal = Main.Adapt.adapt(Array, envfinal.edges[1, 1, 1])
    M = _project_hermitian(randn(scalartype(Tfinal), space(Tfinal)))
    eigsolve_alg = Lanczos()
    rho_prev = right_transfermatrix_fixedpoint([Tprev], [M], eigsolve_alg)
    rho_final = right_transfermatrix_fixedpoint([Tfinal], [M], eigsolve_alg)
    Qprev, = left_orth!(rho_prev; positive=true)
    Qfinal, = left_orth!(rho_final; positive=true)
    sigma_cpu = Qprev * Qfinal'
    sigma = envprev.edges[1,1,1].data isa Main.CUDA.CuArray ? Main.Adapt.adapt(Main.CUDA.CuArray, sigma_cpu) : sigma_cpu
    return fill(sigma, (4, 1, 1))
end

function source_groundstate(payload)
    hasproperty(payload, :selected_groundstate) && return payload.selected_groundstate
    hasproperty(payload, :groundstate) && return payload.groundstate
    hasproperty(payload, :peps) && return payload
    error("source contains no PEPS")
end

function source_h(payload, path)
    hasproperty(payload, :common) && hasproperty(payload.common, :h) &&
        return Float64(payload.common.h)
    hasproperty(payload, :config) && hasproperty(payload.config, :h) &&
        return Float64(payload.config.h)
    match_h = match(r"h([0-9]+\.[0-9]+)", path)
    isnothing(match_h) && error("cannot infer field")
    return parse(Float64, only(match_h.captures))
end

adapt_peps(to, peps) = PEPSKit.InfinitePEPS(map(t -> Adapt.adapt(to, t), peps.A))
adapt_environment(to, env) = PEPSKit.CTMRGEnv(
    map(t -> Adapt.adapt(to, t), env.corners),
    map(t -> Adapt.adapt(to, t), env.edges),
)
function adapt_operator(to, operator)
    terms = [copy(sites) => Adapt.adapt(to, term)
        for (sites, term) in operator.terms]
    return PEPSKit.LocalOperator(operator.lattice, terms...)
end

function projected_norm(gradient)
    # Diagnostic only, outside AD/timing: PEPSKit 0.8.1 RotateReflect
    # allocates CPU isomorphisms in _fit_spaces. Project the small PEPS
    # gradient on CPU, just as in the independent reference comparison.
    projected = PEPSKit.symmetrize!(
        adapt_peps(Array, gradient), PEPSKit.RotateReflect())
    return Float64(norm(projected.A[1, 1]))
end

function atomic_csv(path, row)
    mkpath(dirname(path))
    temporary = path * ".tmp." * get(ENV, "SLURM_JOB_ID", "manual")
    open(temporary, "w") do io
        println(io, join(string.(keys(row)), ','))
        println(io, join(string.(values(row)), ','))
    end
    mv(temporary, path; force=true)
end

function PEPSKit.symmetrize!(state::PEPSKit.InfinitePEPS{TensorKit.TensorMap{T,S,N1,N2,A}}, symmetry::PEPSKit.RotateReflect) where {T,S,N1,N2,A<:CUDA.CuArray}
    cpu = adapt_peps(Array, state)
    PEPSKit.symmetrize!(cpu, symmetry)
    for i in eachindex(state.A); state.A[i] = Adapt.adapt(CUDA.CuArray, cpu.A[i]); end
    return state
end
