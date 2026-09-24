# Opt-in compatibility for PEPSKit 0.8.1. Not loaded by LMPSVUMPS itself.
# The upstream QR rrule allocates zeros(scalartype(t), space(t)), which is CPU
# storage even for a CUDA TensorMap. Keep the formula, specialize only storage.
import PEPSKit, TensorKit, CUDA

function PEPSKit.ChainRulesCore.rrule(
        ::typeof(PEPSKit.left_orth!),
        t::TensorKit.TensorMap{T,S,N1,N2,A},
        alg::PEPSKit.QRAdjoint{F,R},
    ) where {T,S,N1,N2,A<:CUDA.CuArray,
             F<:PEPSKit.MatrixAlgebraKit.Algorithm,R<:PEPSKit.FullPullback}
    QR = PEPSKit.left_orth(t, alg)
    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)
    function gpu_qr_pullback(delta)
        dt = zero(t)
        PEPSKit.MatrixAlgebraKit.qr_pullback!(
            dt, t, QR, PEPSKit.ChainRulesCore.unthunk.(delta);
            gauge_atol=gtol(delta))
        return PEPSKit.ChainRulesCore.NoTangent(), dt, PEPSKit.ChainRulesCore.NoTangent()
    end
    function gpu_qr_pullback(::Tuple{PEPSKit.ChainRulesCore.ZeroTangent,
                                     PEPSKit.ChainRulesCore.ZeroTangent})
        return PEPSKit.ChainRulesCore.NoTangent(), PEPSKit.ChainRulesCore.ZeroTangent(),
               PEPSKit.ChainRulesCore.NoTangent()
    end
    QR, gpu_qr_pullback
end
