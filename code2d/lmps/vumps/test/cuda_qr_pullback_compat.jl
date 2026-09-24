# Execute only in an allocated GPU Slurm job after loading the opt-in shim.
function test_cuda_qr_pullback_compat()
    Random.seed!(20260909)
    t = randn(ComplexF64, TensorKit.ComplexSpace(12) ← TensorKit.ComplexSpace(5))
    alg = PEPSKit.QRAdjoint()
    qc, pc = PEPSKit.ChainRulesCore.rrule(PEPSKit.left_orth!, copy(t), alg)
    tg = Adapt.adapt(CUDA.CuArray, t)
    qg, pg = PEPSKit.ChainRulesCore.rrule(PEPSKit.left_orth!, copy(tg), alg)
    dq = randn(ComplexF64, TensorKit.space(qc[1]))
    dr = randn(ComplexF64, TensorKit.space(qc[2]))
    # Pullback kernels may use mutable work buffers, so give them fresh copies.
    dc = pc((copy(dq), copy(dr)))[2]
    dg = pg((Adapt.adapt(CUDA.CuArray, dq), Adapt.adapt(CUDA.CuArray, dr)))[2]
    CUDA.synchronize()
    dg.data isa CUDA.CuArray || error("CUDA QR cotangent was not allocated on GPU")
    forward_error = max(norm(qc[1]-Adapt.adapt(Array,qg[1]))/norm(qc[1]),
                        norm(qc[2]-Adapt.adapt(Array,qg[2]))/norm(qc[2]))
    gradient_error = norm(dc-Adapt.adapt(Array,dg))/norm(dc)
    direction = randn(ComplexF64, TensorKit.space(t)); direction /= norm(direction)
    loss(x) = begin
        q, r = PEPSKit.left_orth(x, alg)
        real(dot(dq,q) + dot(dr,r))
    end
    step = 1e-6
    numerical = (loss(t+step*direction)-loss(t-step*direction))/(2step)
    analytical = real(dot(dc,direction))
    fd_error = abs(numerical-analytical)/max(1.0,abs(numerical),abs(analytical))
    forward_error <= 1e-10 || error("CPU/GPU QR forward mismatch $forward_error")
    gradient_error <= 1e-9 || error("CPU/GPU QR gradient mismatch $gradient_error")
    fd_error <= 5e-6 || error("QR directional finite difference failed $fd_error")
    z = PEPSKit.ChainRulesCore.ZeroTangent()
    pg((z,z))[2] isa PEPSKit.ChainRulesCore.ZeroTangent || error("zero tangent failed")
    (; forward_error, gradient_error, finite_difference_error=fd_error,
       cuda_cotangent=true, zero_tangent=true)
end
