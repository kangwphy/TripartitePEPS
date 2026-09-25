# Experimental CUDA derivative; see SOURCE_SHA256.txt. CPU source untouched.
function direct_original_lmps_fixedpoints(
        ML::AbstractArray{<:Number,3},
        MR::AbstractArray{<:Number,3},
        a::AbstractArray{<:Number,4};
        nmax::Int=8000,
        tol::Real=1e-12,
        metric_rtol::Real=1e-12,
        initial_G=nothing,
        initial_B=nothing,
        solver::Symbol=:synchronized_power,
    )
    size(ML) == size(MR) || throw(DimensionMismatch(
        "original direct ML/MR shapes differ: $(size(ML)) versus $(size(MR))"))
    chi, dp, chir = size(ML)
    chi == chir || throw(DimensionMismatch("original direct boundary bonds differ"))
    expected_a = (dp, dp, dp, dp)
    size(a) == expected_a || throw(DimensionMismatch(
        "original direct bulk tensor must have shape $expected_a, got $(size(a))"))
    nmax > 0 || throw(ArgumentError("original direct nmax must be positive"))
    tol > 0 || throw(ArgumentError("original direct tol must be positive"))
    solver in (:synchronized_power, :arnoldi) || throw(ArgumentError(
        "original direct solver must be :synchronized_power or :arnoldi"))
    T = promote_type(eltype(ML), eltype(MR), eltype(a), Float64)

    apply_G, apply_B = device_direct_maps(ML, MR, a)
    G0 = initial_G === nothing ? CuArray(Matrix{T}(I, chi, chi)) : CuArray(T.(initial_G))
    size(G0) == (chi, chi) || throw(DimensionMismatch("initial G has wrong shape"))
    G0 ./= max(norm(G0), eps(Float64))

    B0 = initial_B === nothing ? CuArray(T.(ML)) : CuArray(T.(initial_B))
    size(B0) == (chi, dp, chi) || throw(DimensionMismatch("initial B has wrong shape"))
    B0 ./= max(norm(B0), eps(Float64))

    local G, B, lambda_G, lambda_B, iterations_G, iterations_B
    if solver === :arnoldi
        kdim_G = min(max(4, chi^2), 40)
        kdim_B = min(max(4, chi^2 * dp), 60)
        alg_G = KrylovKit.Arnoldi(; tol=Float64(tol), maxiter=nmax,
                                  krylovdim=kdim_G, verbosity=0)
        alg_B = KrylovKit.Arnoldi(; tol=Float64(tol), maxiter=nmax,
                                  krylovdim=kdim_B, verbosity=0)
        vals_G, vecs_G, info_G = KrylovKit.eigsolve(apply_G, G0, 1, :LM, alg_G)
        vals_B, vecs_B, info_B = KrylovKit.eigsolve(apply_B, B0, 1, :LM, alg_B)
        info_G.converged >= 1 || error("original direct G fixed point did not converge")
        info_B.converged >= 1 || error("original direct LTR fixed point did not converge")
        G = vecs_G[1] / norm(vecs_G[1])
        B = vecs_B[1] / norm(vecs_B[1])
        lambda_G = vals_G[1]
        lambda_B = vals_B[1]
        iterations_G = Int(info_G.numiter)
        iterations_B = Int(info_B.numiter)
    else
        # Paired power iterations with physically compatible seeds.  G and B
        # live in different vector spaces, so this is sector tracking, not a
        # claim that they are two cuts of one literal L_n/R_n tail.  Separate
        # Arnoldi solves may choose unrelated vectors inside a degenerate
        # ordered-sector eigenspace even though both residuals are tiny.
        G = copy(G0); B = copy(B0)
        converged_G = false; converged_B = false
        iterations_G = nmax; iterations_B = nmax
        for it in 1:nmax
            raw_G = apply_G(G); raw_B = apply_B(B)
            norm(raw_G) > eps(Float64) || error("original direct G power step vanished")
            norm(raw_B) > eps(Float64) || error("original direct B power step vanished")
            next_G = raw_G / norm(raw_G)
            next_B = raw_B / norm(raw_B)
            overlap_G = dot(vec(G), vec(next_G))
            overlap_B = dot(vec(B), vec(next_B))
            abs(overlap_G) > eps(Float64) &&
                (next_G .*= conj(overlap_G) / abs(overlap_G))
            abs(overlap_B) > eps(Float64) &&
                (next_B .*= conj(overlap_B) / abs(overlap_B))
            delta_G = norm(next_G - G)
            delta_B = norm(next_B - B)
            G, B = next_G, next_B
            if !converged_G && delta_G <= tol
                converged_G = true; iterations_G = it
            end
            if !converged_B && delta_B <= tol
                converged_B = true; iterations_B = it
            end
            converged_G && converged_B && break
        end
        lambda_G = dot(vec(G), vec(apply_G(G))) / dot(vec(G), vec(G))
        lambda_B = dot(vec(B), vec(apply_B(B))) / dot(vec(B), vec(B))
    end
    residual_G = norm(apply_G(G) - lambda_G * G) /
                 max(norm(apply_G(G)), abs(lambda_G) * norm(G), eps(Float64))
    residual_B = norm(apply_B(B) - lambda_B * B) /
                 max(norm(apply_B(B)), abs(lambda_B) * norm(B), eps(Float64))
    Gi, dropped = m2_metric_inverse(G; rtol=metric_rtol)
    H = direct_right_metric(B, Gi)
    (; G, B, H, lambda_G, lambda_B,
       residual_G=Float64(residual_G), residual_B=Float64(residual_B),
       iterations_G, iterations_B, solver=String(solver),
       metric_condition=Float64(maximum(svdvals(G)) / minimum(svdvals(G))), metric_dropped=dropped)
end

function direct_cardinal_regional_rails(
        original_north,
        Mwest::AbstractArray{<:Number,3},
        Meast::AbstractArray{<:Number,3},
        Msouth::AbstractArray{<:Number,3};
        rtol::Real=1e-12,
        reflect_vA::Bool=false,
        reflect_vB::Bool=false,
        reflect_cL::Bool=false,
        reflect_cR::Bool=true,
    )
    chi, dp, chir = size(Mwest)
    chi == chir || throw(DimensionMismatch("west boundary bonds differ"))
    size(Meast) == size(Msouth) == size(Mwest) || throw(DimensionMismatch(
        "cardinal boundary tensors must have the same shape"))
    size(original_north.B) == (chi, dp, chi) || throw(DimensionMismatch(
        "north LTR tensor does not match cardinal boundaries"))
    Gi, dropped = m2_metric_inverse(original_north.G; rtol)
    HA = direct_right_metric(original_north.B, Gi)
    HB = similar(HA)
    for p in 1:dp
        HB[:, p, :] = Gi * (@view original_north.B[:, p, :])
    end

    # Cardinal VUMPS tensors already carry a geometrical rotation.  Keep the
    # remaining bond reflection explicit so it can be audited independently
    # instead of hiding it inside an endpoint-map reshape.
    vA = direct_rail(Mwest, "vL"; reflected=reflect_vA)
    vB = direct_rail(Meast, "vR"; reflected=reflect_vB)
    hA_left = m2_horizontal_left(HA)
    hA_right = m2_horizontal_right(HA)
    hB_left = m2_horizontal_left(HB)
    hB_right = m2_horizontal_right(HB)
    cL = direct_rail(Msouth, "hL"; reflected=reflect_cL)
    cR = direct_rail(Msouth, "hR"; reflected=reflect_cR)

    # Unused entries are still populated because the rail API deliberately
    # requires a complete oriented object. m2_rail selects only A:(hL,vL),
    # B:(hR,vR), and C:(cL,cR) in the AB|C geometry.
    A = (hL=hA_left, hR=hA_right, vL=vA, vR=vA,
         metric_dropped=dropped)
    B = (hL=hB_left, hR=hB_right, vL=vB, vR=vB,
         metric_dropped=dropped)
    C = (hL=cL, hR=cR, vL=vA, vR=vB, cL, cR,
         metric_dropped=dropped)
    (; rails=(A=A, B=B, C=C), HA, HB, Gi, metric_dropped=dropped,
       reflect_vA, reflect_vB, reflect_cL, reflect_cR)
end
function direct_right_metric(B::AbstractArray{<:Number,3}, Gi)
    H = similar(B, promote_type(eltype(B), eltype(Gi)))
    for p in axes(B, 2)
        H[:,p,:] = (@view B[:,p,:]) * Gi
    end
    H
end
function direct_rail(M::AbstractArray{<:Number,3}, ray::AbstractString;
                     reflected::Bool=false)
    data = reflected ? permutedims(M, (3, 2, 1)) : copy(M)
    n, dp, nr = size(data)
    n == nr || throw(DimensionMismatch("direct rail bonds differ: $(size(data))"))
    ITensor(data,
            Index(n, "M2Rail,$ray,L"),
            Index(dp, "M2Rail,$ray,P"),
            Index(n, "M2Rail,$ray,R"))
end