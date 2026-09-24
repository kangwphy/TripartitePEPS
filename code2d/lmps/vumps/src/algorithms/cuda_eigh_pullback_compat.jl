# Opt-in GPU compatibility for PEPSKit 0.8.1 full Eigh pullback.
import PEPSKit, TensorKit, CUDA
# Preserve CUDA storage and copy only small truncation-index metadata to host.
function PEPSKit.ChainRulesCore.rrule(
    ::typeof(PEPSKit.eigh_trunc!), t::TensorKit.TensorMap{T,S,N1,N2,A},
    alg::PEPSKit.EighAdjoint{F,R}) where {T,S,N1,N2,A<:CUDA.CuArray,
    F<:PEPSKit.MatrixAlgebraKit.TruncatedAlgorithm{<:PEPSKit.MatrixAlgebraKit.Algorithm},R<:PEPSKit.FullPullback}
    D,V = PEPSKit.eigh_full!(t; alg.fwd_alg.alg)
    (Dt,Vt), inds = PEPSKit.truncate(PEPSKit.eigh_trunc!, (D,V), alg.fwd_alg.trunc)
    err = PEPSKit.truncation_error(PEPSKit.diagview(D), inds)
    # Dictionary value type is GPU-specific, so construct a new dictionary.
    hostinds = Dict(c => Array(i) for (c,i) in inds)
    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)
    function back(delta)
        dt = zero(t)
        PEPSKit.eigh_pullback!(dt,t,(D,V),delta,hostinds;
            gauge_atol=gtol(delta),degeneracy_atol=alg.rrule_alg.degeneracy_atol)
        return PEPSKit.ChainRulesCore.NoTangent(),dt,PEPSKit.ChainRulesCore.NoTangent()
    end
    return (Dt,Vt,err),back
end

