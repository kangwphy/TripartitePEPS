# direct_route.jl -- direct and diagnostic routes for the LMPS block.
# =============================================================================
# The endpoint/enlarged-cut route keeps the complete grown tensor
#
#     Dgrow[(old bond, MPO bond), p, (old bond, MPO bond)]
#
# instead of replacing the enlarged cut by local flattenings.  It contracts
# the same mixed channel to finite depth.  In the infinite-depth
# limit it is the same network represented by the endpoint fixed points,
# provided the local channel, far-boundary sector, gauge transport, and
# normalization are shared.  Its intermediate bond is N = chi * D_eff, while
# the final H has retained bond chi.
#
# The `direct_finite_depth` route below keeps the complete mixed finite-depth
# maps L_n/R_n.  Its one-copy block H=(L_n D R_n)(L_n R_n)^(-1) is an
# alternative representation of the boundary LMPS and is audited by a
# gauge-invariant free-energy test.  L_n/R_n themselves are oblique maps, so
# they are refactorized into orthonormal endpoint charts before the six-R
# solver uses its bare-delta far boundary.  This refactorization preserves the
# direct projector exactly and does not invoke the density-projector subspace.
# The separate two-/three-column fixed points later in this file are a second
# route: G=fp(ML|MR) and B=fp(ML|a|MR) do not literally share L_n/R_n tails.
# Their compatibility must be established by sector tracking and a closed
# LMPS/free-energy gate, not by the two eigen-residuals alone.
# The older `direct_ltr` function later in this file deliberately keeps the
# local-flattening shortcut as a separate audit diagnostic; it is not the same
# object and must not be used as evidence for the finite-depth route.
# =============================================================================

"Copy a rank-three local tensor into a named rail frame."
function direct_rail(M::AbstractArray{<:Number,3}, ray::AbstractString;
                     reflected::Bool=false)
    data = reflected ? permutedims(M, (3, 2, 1)) : Array(M)
    n, dp, nr = size(data)
    n == nr || throw(DimensionMismatch("direct rail bonds differ: $(size(data))"))
    ITensor(data,
            Index(n, "M2Rail,$ray,L"),
            Index(dp, "M2Rail,$ray,P"),
            Index(n, "M2Rail,$ray,R"))
end

"Four independently tagged rails made from the complete grown tensor."
function direct_spatial_rails(Dgrow::AbstractArray{<:Number,3})
    (hL=direct_rail(Dgrow, "hL"),
     hR=direct_rail(Dgrow, "hR"; reflected=true),
     vL=direct_rail(Dgrow, "vL"),
     vR=direct_rail(Dgrow, "vR"; reflected=true))
end

"Relative residual of a dense transfer-channel eigenmatrix."
function direct_channel_residual(D::AbstractArray{<:Number,3}, X,
                                lambda::Real, side::Symbol)
    n, dp, nr = size(D)
    n == nr || throw(DimensionMismatch("direct channel bonds differ"))
    Y = zeros(promote_type(eltype(D), eltype(X)), n, n)
    for p in 1:dp
        Dp = @view D[:, p, :]
        Y .+= side === :L ? Dp' * X * Dp : Dp * X * Dp'
    end
    norm(Y - lambda * X) / max(norm(Y), abs(lambda) * norm(X), eps(Float64))
end

"""
    direct_metric_fixedpoint(ML, MR; nmax, tol)

Legacy Hermitian MPS-overlap diagnostic.  It contracts the second tensor as a
bra and therefore solves

    E_LR(X) = sum_x ML[x] * X * MR[x]' = lambda_G X.

This is *not* the ordinary scalar-network two-column ladder used by
`direct_original_lmps_fixedpoints`, whose explicitly oriented tensors obey
`sum_x transpose(ML[x])*G*MR[x]`.  The two conventions coincide in some real
reflection-symmetric tests but must not be interchanged for a generic complex
quantum PEPS.  This function remains only for the legacy local-flattening
diagnostic below.
"""
function direct_metric_fixedpoint(ML::AbstractArray{<:Number,3},
                                  MR::AbstractArray{<:Number,3};
                                  nmax::Int=4000, tol::Real=1e-12)
    size(ML) == size(MR) ||
        throw(DimensionMismatch("direct mixed metric tensor shapes differ: " *
                                "$(size(ML)) versus $(size(MR))"))
    chi, dp, chir = size(ML)
    chi == chir || throw(DimensionMismatch("direct metric bonds are not square"))
    nmax > 0 || throw(ArgumentError("direct metric nmax must be positive"))
    tol > 0 || throw(ArgumentError("direct metric tol must be positive"))
    T = promote_type(eltype(ML), eltype(MR), Float64)
    apply = X -> begin
        Y = zeros(T, chi, chi)
        for x in 1:dp
            Y .+= (@view ML[:,x,:]) * X * adjoint(@view MR[:,x,:])
        end
        Y
    end
    # KrylovKit works on the matrix directly and preserves a complex left/right
    # metric when the transfer operator is non-Hermitian.
    X0 = Matrix{T}(I, chi, chi); X0 ./= norm(X0)
    alg = KrylovKit.Arnoldi(; tol=Float64(tol), maxiter=nmax,
                            krylovdim=min(max(4, chi^2), 40), verbosity=0)
    vals, vecs, info = KrylovKit.eigsolve(apply, X0, 1, :LM, alg)
    info.converged >= 1 ||
        error("direct mixed metric did not converge in $nmax iterations")
    G = vecs[1]
    G ./= norm(G)
    lambda = vals[1]
    Y = apply(G)
    residual = norm(Y - lambda * G) /
               max(norm(Y), abs(lambda) * norm(G), eps(Float64))
    (; G, lambda, residual, iterations=Int(info.numiter))
end

"""
    direct_original_lmps_fixedpoints(ML, MR, a; nmax, tol)

Solve the `ML | MR` and `ML | a | MR` ladder eigenproblems.  These are two
different vector spaces and do not literally share mixed-channel tails.  A
paired power iteration controls the physical sector heuristically; acceptance
still requires a gauge-invariant LMPS/free-energy closure. With array
convention `M[l,p,r]`, the vertical two-column transfer is

    G'[rL,rR] = sum_{lL,lR,p}
        ML[lL,p,rL] G[lL,lR] MR[lR,p,rR],

and the three-column transfer is

    B'[rL,south,rR] = sum ML[lL,west,rL]
                          a[north,west,south,east]
                          MR[lR,east,rR] B[lL,north,lR].

Even when `ML` and `MR` are oriented consistently, separate dominant-eigenpair
solves can choose unrelated vectors in a degenerate sector.  `H=B*G^+` is a
candidate LMPS tensor; residuals of G and B alone are insufficient to prove
that it is the original boundary ray.

The effective boundary tensor is returned as

    H[p] = B[p] * G^(-1).

No complex conjugation is inserted inside this routine: `ML`, `a`, and `MR`
must already be the oriented local tensors of the scalar double-layer network.
They are the boundary MPS tensors of the two spatial sides of the three-column
ladder, not a bra/ket pair manufactured from one north boundary tensor.  A real
C4-symmetric RK network permits `ML=MR=M`; a generic PEPS requires the actual
west/east (or rotated equivalent) boundary fixed points.
"""
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

    apply_G = X -> begin
        Y = zeros(T, chi, chi)
        for p in 1:dp
            Y .+= transpose(@view ML[:, p, :]) * X * (@view MR[:, p, :])
        end
        Y
    end
    G0 = initial_G === nothing ? Matrix{T}(I, chi, chi) : Array{T}(initial_G)
    size(G0) == (chi, chi) || throw(DimensionMismatch("initial G has wrong shape"))
    G0 ./= max(norm(G0), eps(Float64))

    apply_B = X -> begin
        Y = zeros(T, chi, dp, chi)
        # a is always ordered [north, west, south, east].  The old boundary
        # physical leg enters from north, ML/MR close west/east, and the new
        # boundary physical leg exits south.  A cyclicly shifted assignment is
        # invisible for C4-symmetric RK tensors but wrong for a generic PEPS.
        for north in 1:dp, west in 1:dp, east in 1:dp
            middle = transpose(@view ML[:, west, :]) *
                     (@view X[:, north, :]) * (@view MR[:, east, :])
            for south in 1:dp
                Y[:, south, :] .+= a[north, west, south, east] .* middle
            end
        end
        Y
    end
    B0 = initial_B === nothing ? Array{T}(ML) : Array{T}(initial_B)
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
       metric_condition=Float64(cond(G)), metric_dropped=dropped)
end

"Build one complete L-shaped rail object from the original G/B fixed points."
function direct_original_oriented_rails(original, ML, MR; rtol::Real=1e-12)
    H = original.H
    chi, dp, chir = size(H)
    chi == chir || throw(DimensionMismatch("original direct H bonds differ"))
    size(ML) == size(MR) == (chi, dp, chi) || throw(DimensionMismatch(
        "original direct rail boundary tensors do not match H"))
    Llocal = direct_left_map(ML)
    # ML/MR are already the oriented scalar-network columns.  In particular,
    # a quantum caller passes ML=conj(M), MR=M.  Taking an adjoint here would
    # conjugate MR a second time and turn the intended M^dagger G M ladder into
    # the wrong M^T G M ladder.
    Rlocal = direct_right_map_plain(MR)
    Gi, dropped = m2_metric_inverse(original.G; rtol)
    Rcommon = Rlocal * Gi
    (hL=m2_horizontal_left(H), hR=m2_horizontal_right(H),
     vL=m2_vertical_left(Llocal, chi, dp),
     vR=m2_vertical_right(Rcommon, chi, dp),
     right_metric_dropped=dropped)
end

"""
    direct_cardinal_regional_rails(original_north, Mwest, Meast, Msouth)

Build the three regional rail frames of the global `AB|C` geometry directly
from cardinal boundary MPS tensors.  `original_north.B` has a west-frame left
bond and an east-frame right bond, while `G` maps between those frames.  Hence
the two upper L shapes require different local horizontal tensors,

    H_A[p] = B[p] G^(-1),       H_B[p] = G^(-1) B[p].

The A vertical arm stays in the west frame, the B vertical arm stays in the
east frame, and C is the independently solved south straight boundary.  No
raw bond is identified across regions; the regional six-R closure constructs
the mixed seam maps explicitly.
"""
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

"Flatten a local MPS site as L[a,(l,x)] = M[a,x,l]."
function direct_left_map(M::AbstractArray{<:Number,3})
    chi, dp, chir = size(M)
    chi == chir || throw(DimensionMismatch("left map bonds are not square"))
    L = zeros(eltype(M), chi, chi * dp)
    for a in 1:chi, x in 1:dp, l in 1:chi
        L[a, l + chi * (x - 1)] = M[a, x, l]
    end
    L
end

"Flatten the opposite local MPS site as R[(l,x),a] = M[a,x,l]^*."
function direct_right_map(M::AbstractArray{<:Number,3})
    direct_left_map(M)'
end

"Flatten an already oriented right scalar-network column without conjugating it."
function direct_right_map_plain(M::AbstractArray{<:Number,3})
    transpose(direct_left_map(M))
end

"Contract one active grown tensor through direct local left/right maps." 
function direct_ltr_block(L::AbstractMatrix, Dgrow::AbstractArray{<:Number,3},
                          R::AbstractMatrix)
    size(Dgrow, 1) == size(L, 2) ||
        throw(DimensionMismatch("direct L/Dgrow cut mismatch"))
    size(Dgrow, 3) == size(R, 1) ||
        throw(DimensionMismatch("direct Dgrow/R cut mismatch"))
    chi = size(L, 1); dp = size(Dgrow, 2)
    B = zeros(promote_type(eltype(L), eltype(Dgrow), eltype(R)), chi, dp, chi)
    for p in 1:dp
        B[:,p,:] = L * (@view Dgrow[:,p,:]) * R
    end
    B
end

"Attach G^{-1} on the outgoing block bond, yielding B[p]G^{-1}."
function direct_right_metric(B::AbstractArray{<:Number,3}, Gi)
    H = similar(B, promote_type(eltype(B), eltype(Gi)))
    for p in axes(B, 2)
        H[:,p,:] = (@view B[:,p,:]) * Gi
    end
    H
end

"Build regional rails from the unverified local-M LTR diagnostic."
function direct_ltr_rails(ML, MR, H, L, R;
                          variant::Symbol=:bond_metric)
    chi, dp, chir = size(H)
    chi == chir || throw(DimensionMismatch("direct LTR block bonds differ"))
    size(ML) == size(MR) == (chi, dp, chi) ||
        throw(DimensionMismatch("direct straight rails do not match LTR block"))
    variant in (:bond_metric, :junction_hinge, :seam_local) ||
        throw(ArgumentError("unknown direct LTR variant $variant"))
    if variant === :seam_local
        # RK-Ising comparison frame: A/B use the direct LTR block and the
        # left direct map, while C remains the straight one-copy boundary.
        # The C rails are named explicitly so this does not silently reuse H.
        rails = (hL=m2_horizontal_left(H), hR=m2_horizontal_right(H),
                 vL=m2_vertical_left(L, chi, dp),
                 vR=reference_vertical_right(L, chi, dp),
                 cL=m2_horizontal_left(ML),
                 cR=reference_horizontal_right(MR))
        return (rails=rails, hinges=Dict{String,Matrix{Float64}}())
    end
    if variant === :bond_metric
        A = (hL=m2_horizontal_left(H), hR=m2_horizontal_right(H),
             vL=m2_vertical_left(L, chi, dp),
             vR=m2_vertical_right(R, chi, dp))
        B = (hL=m2_horizontal_left(H), hR=m2_horizontal_right(H),
             vL=m2_vertical_left(L, chi, dp),
             vR=m2_vertical_right(R, chi, dp))
        C = (hL=m2_horizontal_left(ML), hR=m2_horizontal_right(MR),
             vL=m2_vertical_left(L, chi, dp),
             vR=m2_vertical_right(R, chi, dp))
        return (rails=(A=A, B=B, C=C), hinges=Dict{String,Matrix{Float64}}())
    end

    # The original L-shape convention: B is the bare LTR cell and the
    # mixed metric is inserted once at the A/B junction.  The C half-plane is
    # a straight boundary rail.  This is intentionally kept separate from
    # the bond-metric variant because moving G^{-1} changes the finite-chi
    # tensor network unless the endpoint frame is proven to be canonical.
    rails = m2_oriented_rails(H, L, R; rtol=1e-12)
    rails = merge(rails, (cL=m2_horizontal_left(ML),
                          cR=m2_horizontal_right(MR)))
    Gi, _ = m2_metric_inverse(direct_metric_fixedpoint(ML, MR).G;
                              rtol=1e-12)
    (rails=rails, hinges=Dict("A"=>Matrix{Float64}(Gi),
                              "B"=>Matrix{Float64}(Gi)))
end

"""
    direct_ltr_route_closure(Mt, a, Dphysical; ...)

Legacy Hermitian-overlap/local-flattening diagnostic using only the accepted one-copy local boundary
tensor(s), the bulk local tensor `a`, and the mixed metric fixed point.  No
endpoint rectangle eigenspace is used, but this local substitution is not the
same as contracting the complete finite-depth maps `L_n` and `R_n`.  The
diagnostic block is

    B[p] = L * Dgrow[p] * R,   H[p] = B[p] * G^{-1},

while the vertical arms use the local flattenings of ML and MR.  The regional
rail dictionary keeps A, B, and C in their own named frames so the six-R
closure performs the required mixed seam contractions.  Do not use this
function as the generic complex-PEPS collapsed-cardinal route; that route is
`direct_original_lmps_fixedpoints` with explicitly oriented spatial tensors.
"""
function direct_ltr_route_closure(Mt::AbstractArray{<:Number,3},
                                  a::AbstractArray{<:Number,4},
                                  Dphysical::Int;
                                  MR=Mt,
                                  nmax::Int=1200, tol::Real=1e-10,
                                  metric_nmax::Int=4000,
                                  metric_tol::Real=1e-12,
                                  variant::Symbol=:bond_metric,
                                  verbose::Bool=true)
    size(a, 1) == size(a, 2) == size(a, 3) == size(a, 4) ||
        throw(DimensionMismatch("direct LTR requires a uniform local tensor"))
    ML = Array(Mt); MRarr = Array(MR)
    chi, dp, chir = size(ML)
    size(MRarr) == size(ML) || throw(DimensionMismatch("ML/MR sizes differ"))
    chi == chir || throw(DimensionMismatch("direct LTR boundary bonds are not square"))
    size(a, 1) == dp || throw(DimensionMismatch("direct LTR physical leg mismatch"))

    metric = direct_metric_fixedpoint(ML, MRarr;
                                      nmax=metric_nmax, tol=metric_tol)
    Gi, dropped = m2_metric_inverse(metric.G; rtol=1e-12)
    condition = cond(Matrix(metric.G))
    L = direct_left_map(ML)
    R = direct_right_map(MRarr)
    Dgrow = grown(ML, Array(a))
    B = direct_ltr_block(L, Dgrow, R)
    H = variant === :bond_metric ? direct_right_metric(B, Gi) : B
    route_parts = direct_ltr_rails(ML, MRarr, H, L, R; variant)
    rails, hinges = route_parts.rails, route_parts.hinges
    closure_seconds = @elapsed result = m2_contract_shared_six_R(
        rails, Dphysical; nmax, tol, verbose, hinges)
    block_residual = norm(direct_right_metric(B, Gi) - ML) /
                     max(norm(H), norm(ML), eps(Float64))
    (; result..., method="direct_ltr", base_chi=chi, effective_chi=chi,
       direct_variant=String(variant),
       direct_metric=metric.G, direct_metric_lambda=metric.lambda,
       direct_metric_residual=Float64(metric.residual),
       direct_metric_iterations=metric.iterations,
       direct_metric_condition=Float64(condition), direct_metric_dropped=dropped,
       direct_block_residual=Float64(block_residual), direct_norm=Float64(norm(Dgrow)),
       direct_closure_seconds=Float64(closure_seconds))
end

"""
    direct_finite_depth_maps(ML, MR, Dgrow; depth, tol)

Contract the two *finite* mixed ladders down to the active grown cut.  The
maps are deliberately kept as `L[n] : chi x N` and `R[n] : N x chi`, with
`N=chi*D_eff`; no endpoint eigenspace or local flattening is substituted.
The updates are the literal finite-depth contractions

    L[n+1] = sum_p ML[p]' L[n] Dgrow[p],
    R[n+1] = sum_p Dgrow[p] R[n] MR[p]'.

Normalizing each map is harmless because the final block uses the matching
`G[n]=L[n]R[n]` and its inverse.  `depth` is therefore an explicit physical
finite-depth control, while `residual_*` tests the convergence of the two
semi-infinite mixed maps.  In the primitive-sector limit the resulting
one-copy H can represent the same boundary LMPS as the endpoint route.  The
raw maps need not equal its orthonormal endpoint rectangles, and they must not
be attached to a bare-delta replica far boundary without transporting their
overlap metric.
"""
function direct_finite_depth_maps(
        ML::AbstractArray{<:Number,3},
        MR::AbstractArray{<:Number,3},
        Dgrow::AbstractArray{<:Number,3};
        depth::Int=4000,
        tol::Real=1e-10,
        min_depth::Int=1,
        initial_L=nothing,
        initial_R=nothing,
    )
    depth > 0 || throw(ArgumentError("direct finite depth must be positive"))
    min_depth > 0 || throw(ArgumentError("direct minimum depth must be positive"))
    tol > 0 || throw(ArgumentError("direct finite-depth tolerance must be positive"))
    size(ML) == size(MR) || throw(DimensionMismatch(
        "direct finite-depth ML/MR shapes differ: $(size(ML)) versus $(size(MR))"))
    chi, dp, chir = size(ML)
    chi == chir || throw(DimensionMismatch("direct finite-depth boundary bonds are not square"))
    N, dpg, Nr = size(Dgrow)
    N == Nr || throw(DimensionMismatch("direct finite-depth grown bonds are not square"))
    dp == dpg || throw(DimensionMismatch("direct finite-depth physical legs differ"))
    N == chi * dp || throw(DimensionMismatch(
        "direct finite-depth expected grown bond $(chi*dp), got $N"))
    T = promote_type(eltype(ML), eltype(MR), eltype(Dgrow), Float64)
    L = initial_L === nothing ? ones(T, chi, N) : Array{T}(initial_L)
    R = initial_R === nothing ? ones(T, N, chi) : Array{T}(initial_R)
    size(L) == (chi, N) || throw(DimensionMismatch("initial direct L has wrong shape"))
    size(R) == (N, chi) || throw(DimensionMismatch("initial direct R has wrong shape"))
    norm(L) > eps() || throw(ArgumentError("initial direct L is zero"))
    norm(R) > eps() || throw(ArgumentError("initial direct R is zero"))
    L ./= norm(L); R ./= norm(R)
    residualL = Inf; residualR = Inf
    lambdaL = one(T); lambdaR = one(T)
    used = depth
    for it in 1:depth
        rawL = zeros(T, chi, N)
        rawR = zeros(T, N, chi)
        for p in 1:dp
            rawL .+= adjoint(@view ML[:, p, :]) * L * (@view Dgrow[:, p, :])
            rawR .+= (@view Dgrow[:, p, :]) * R * adjoint(@view MR[:, p, :])
        end
        nL = norm(rawL); nR = norm(rawR)
        nL > eps() || error("direct finite-depth left ladder annihilated at depth $it")
        nR > eps() || error("direct finite-depth right ladder annihilated at depth $it")
        lambdaL = nL / max(norm(L), eps()); lambdaR = nR / max(norm(R), eps())
        nextL = rawL / nL; nextR = rawR / nR
        # Fix only the scalar phase/sign relative to the preceding finite map.
        # No bond-space gauge is inserted here; this is a literal ladder update.
        ovL = dot(vec(L), vec(nextL)); ovR = dot(vec(R), vec(nextR))
        abs(ovL) > eps() && (nextL .*= conj(ovL) / abs(ovL))
        abs(ovR) > eps() && (nextR .*= conj(ovR) / abs(ovR))
        residualL = norm(nextL - L) / max(norm(nextL), norm(L), eps())
        residualR = norm(nextR - R) / max(norm(nextR), norm(R), eps())
        L, R = nextL, nextR
        if it >= min_depth && max(residualL, residualR) <= tol
            used = it
            break
        end
    end
    (; L, R, depth=used, residualL=Float64(residualL),
       residualR=Float64(residualR), lambdaL, lambdaR)
end

"""
    direct_mps_transfer_radius(M; nmax=4000, tol=1e-12)

Compute the dominant eigenvalue of the one-site MPS self-transfer channel

    E_M(X) = sum_p M[p] * X * M[p]^dagger.

The returned `radius` fixes the otherwise invisible per-site scale of an MPS
tensor.  This is intentionally different from the scale-invariant PEPS-strip
ratio used by the older free-energy audits: multiplying every `M[p]` by a
scalar changes `radius` and is therefore detected here.
"""
function direct_mps_transfer_radius(M::AbstractArray{<:Number,3};
                                    nmax::Int=4000,
                                    tol::Real=1e-12)
    chi, dp, chir = size(M)
    chi == chir || throw(DimensionMismatch("MPS transfer bonds are not square"))
    nmax > 0 || throw(ArgumentError("MPS transfer nmax must be positive"))
    tol > 0 || throw(ArgumentError("MPS transfer tolerance must be positive"))
    T = promote_type(eltype(M), ComplexF64)
    apply = X -> begin
        Y = zeros(T, chi, chi)
        for p in 1:dp
            Mp = @view M[:, p, :]
            Y .+= Mp * X * adjoint(Mp)
        end
        Y
    end
    X0 = Matrix{T}(I, chi, chi)
    X0 ./= norm(X0)
    alg = KrylovKit.Arnoldi(; tol=Float64(tol), maxiter=nmax,
                            krylovdim=min(max(4, chi^2), 40), verbosity=0)
    values, vectors, info = KrylovKit.eigsolve(apply, X0, 1, :LM, alg)
    info.converged >= 1 || error("MPS self-transfer radius did not converge")
    value = values[1]
    fixedpoint = vectors[1]
    image = apply(fixedpoint)
    residual = norm(image - value * fixedpoint) /
        max(norm(image), abs(value) * norm(fixedpoint), eps(Float64))
    (; radius=Float64(abs(value)), eigenvalue=value,
       residual=Float64(residual), iterations=Int(info.numiter), fixedpoint)
end

"""
    direct_unit_transfer_lmps(Hraw; nmax=4000, tol=1e-12)

Remove the per-site scale from a reconstructed LMPS tensor using only its own
self-transfer channel.  If `Hraw = kappa * M` and `M` is in the usual unit
transfer-radius canonical normalization, then `scale = abs(kappa)` and
`H = Hraw / scale` represents the same normalized infinite MPS as `M` (up to
an irrelevant global phase and bond gauge).
"""
function direct_unit_transfer_lmps(Hraw::AbstractArray{<:Number,3};
                                   nmax::Int=4000,
                                   tol::Real=1e-12)
    transfer = direct_mps_transfer_radius(Hraw; nmax, tol)
    transfer.radius > eps(Float64) || error("reconstructed LMPS has zero radius")
    scale = sqrt(transfer.radius)
    H = Hraw / scale
    check = direct_mps_transfer_radius(H; nmax, tol)
    (; H, scale=Float64(scale), raw_transfer=transfer,
       normalized_transfer=check)
end

"""
    direct_orthonormal_endpoints(L, R; rtol=1e-12)

Refactor the converged oblique semi-infinite maps without changing their
enlarged-cut projector.  Thin SVDs give

    L = C_L * Lhat,      R = Rhat * C_R,

with `Lhat*Lhat' = I` and `Rhat'*Rhat = I`.  Therefore, exactly within the
retained numerical rank,

    R*(L*R)^(-1)*L = Rhat*(Lhat*Rhat)^(-1)*Lhat.

Unlike the raw mixed maps, `Lhat` and `Rhat` are orthonormal endpoint charts;
a bare delta far boundary is meaningful in each chart.  No density matrix is
diagonalized here and no retained subspace is changed.
"""
function direct_orthonormal_endpoints(L::AbstractMatrix,
                                      R::AbstractMatrix;
                                      rtol::Real=1e-12)
    chi, N = size(L)
    size(R) == (N, chi) || throw(DimensionMismatch(
        "direct endpoint maps have incompatible shapes $(size(L)), $(size(R))"))
    rtol > 0 || throw(ArgumentError("endpoint refactorization rtol must be positive"))
    FL = svd(Matrix(L); full=false)
    FR = svd(Matrix(R); full=false)
    cutL = Float64(rtol) * maximum(FL.S)
    cutR = Float64(rtol) * maximum(FR.S)
    rankL = count(>(cutL), FL.S)
    rankR = count(>(cutR), FR.S)
    rankL == chi || error("direct left map rank $rankL is below chi=$chi")
    rankR == chi || error("direct right map rank $rankR is below chi=$chi")
    Lhat = adjoint(FL.V[:, 1:chi])
    Rhat = FR.U[:, 1:chi]
    Ghat = Lhat * Rhat
    Gi_raw, dropped_raw = m2_metric_inverse(L * R; rtol)
    Gi_hat, dropped_hat = m2_metric_inverse(Ghat; rtol)
    Praw = R * Gi_raw * L
    Phat = Rhat * Gi_hat * Lhat
    projector_residual = norm(Praw - Phat) /
                         max(norm(Praw), norm(Phat), eps(Float64))
    left_isometry = norm(Lhat * adjoint(Lhat) - I) / sqrt(chi)
    right_isometry = norm(adjoint(Rhat) * Rhat - I) / sqrt(chi)
    (; L=Lhat, R=Rhat, G=Ghat,
       projector_residual=Float64(projector_residual),
       left_isometry=Float64(left_isometry),
       right_isometry=Float64(right_isometry),
       dropped_raw, dropped=dropped_hat,
       condition=Float64(cond(Ghat)))
end

"""
    direct_balanced_support_endpoints(L, R, Dgrow; rtol=1e-12)

Put the complete generalized L-shape `(B=L*Dgrow*R, G=L*R)` in a balanced
minimal-support chart.  If `G=U*S*V'`, define

    Ls = S^(-1/2) * U' * L,
    Rs = R * V * S^(-1/2),
    Hs[p] = Ls * Dgrow[p] * Rs.

Then `Ls*Rs=I` on the retained numerical support, and `Hs` has exactly the
same closed products as `...B[p]G^+B[q]G^+...`.  Both endpoint maps and the
horizontal tensor are transported together; this is therefore a gauge/support
change of the complete L-shape, unlike independently orthogonalizing only the
two endpoint maps.
"""
function direct_balanced_support_endpoints(
        L::AbstractMatrix, R::AbstractMatrix,
        Dgrow::AbstractArray{<:Number,3}; rtol::Real=1e-12)
    chi, N = size(L)
    size(R) == (N, chi) || throw(DimensionMismatch(
        "balanced endpoints have incompatible shapes $(size(L)), $(size(R))"))
    size(Dgrow, 1) == size(Dgrow, 3) == N || throw(DimensionMismatch(
        "balanced grown tensor does not match endpoint cut"))
    rtol > 0 || throw(ArgumentError("balanced support rtol must be positive"))
    Graw = L * R
    F = svd(Matrix(Graw))
    isempty(F.S) && error("balanced support metric is empty")
    cutoff = Float64(rtol) * F.S[1]
    keep = findall(s -> s > cutoff, F.S)
    isempty(keep) && error("balanced support is empty at rtol=$rtol")
    U = F.U[:, keep]
    V = F.V[:, keep]
    Sinvhalf = Diagonal(inv.(sqrt.(F.S[keep])))
    Ls = Sinvhalf * adjoint(U) * L
    Rs = R * V * Sinvhalf
    Gs = Ls * Rs
    Hs = direct_ltr_block(Ls, Dgrow, Rs)
    rank = length(keep)
    identity_residual = norm(Gs - I) / sqrt(rank)
    Gi_raw, dropped_raw = m2_metric_inverse(Graw; rtol)
    Praw = R * Gi_raw * L
    Ps = Rs * Ls
    projector_residual = norm(Praw - Ps) /
                         max(norm(Praw), norm(Ps), eps(Float64))
    (; L=Ls, R=Rs, G=Gs, B=Hs, H=Hs, rank, cutoff,
       raw_condition=Float64(cond(Graw)), dropped_raw,
       identity_residual=Float64(identity_residual),
       projector_residual=Float64(projector_residual))
end

"""
    direct_finite_depth_route_closure(Mt, a, Dphysical; ...)

Direct semi-infinite-map route for the production selector.  It evaluates a finite-depth
approximation to the original definitions `G_n=L_n R_n` and
`B_n[p]=L_n Dgrow[p] R_n`, then closes the six-R network with
`H_n=B_n G_n^{-1}`.  The closure keeps the oblique maps in the same
mixed-fixed-point chart as the old boundary MPS.  A separate thin-SVD
projector audit verifies the retained enlarged-cut subspace, but that
refactor is not inserted into only part of the L-shaped object.  The one-copy
H is gauge-audited separately.  This is intentionally separate from the old
`direct_ltr` local shortcut; the latter remains available as an explicitly
unverified diagnostic.
"""
function direct_finite_depth_route_closure(
        Mt::AbstractArray{<:Number,3},
        a::AbstractArray{<:Number,4},
        Dphysical::Int;
        nmax::Int=1200,
        tol::Real=1e-10,
        direct_depth::Int=4000,
        direct_map_tol::Real=1e-10,
        direct_min_depth::Int=4,
        verbose::Bool=true,
    )
    ML = Array(Mt); MR = Array(Mt)
    Dgrow = grown(ML, Array(a))
    # The far end is the natural local closure of the already accepted
    # boundary MPS, not an unrelated all-ones rectangle.  In a primitive
    # channel both seeds reach the same ray.  At a critical/ordered
    # near-degeneracy this physical seed selects the sector continuously
    # connected to Mt; an arbitrary seed can converge with a tiny residual to
    # a different projector and corrupt only the replica constants.
    maps = direct_finite_depth_maps(ML, MR, Dgrow;
                                    depth=direct_depth,
                                    tol=direct_map_tol,
                                    min_depth=direct_min_depth,
                                    initial_L=direct_left_map(ML),
                                    initial_R=direct_right_map(MR))
    # Keep the maps in the original mixed-fixed-point chart.  This is the same
    # chart used by the endpoint construction and retains the relation between
    # the new enlarged cut and the old boundary-MPS bond.  A thin-SVD
    # refactor preserves P=R(LR)^(-1)L but, unless every old-bond rail is
    # transported as well, it is not a gauge transformation of the complete
    # L-shaped object.  `direct_orthonormal_endpoints` remains available as a
    # projector diagnostic, but is deliberately not inserted here.
    canonical = direct_orthonormal_endpoints(maps.L, maps.R; rtol=1e-12)
    G = maps.L * maps.R
    Gi, dropped = m2_metric_inverse(G; rtol=1e-12)
    B = direct_ltr_block(maps.L, Dgrow, maps.R)
    H = direct_right_metric(B, Gi)
    # Keep this raw scalar-rescaled mismatch only as a representation
    # diagnostic.  It is not gauge invariant: H and Mt can describe the same
    # boundary LMPS through a strongly non-orthogonal bond transform while this
    # number remains O(1).  `benchmarks/direct_lmps_free_energy.jl` supplies the
    # decisive one-copy free-energy and normalized-spectrum checks.
    Mt0 = Array(Mt)
    # Use the Hermitian Frobenius product.  The density-projector RK path is
    # real, but a VUMPS checkpoint generally carries a complex bond gauge even
    # for a real PEPS.  The old elementwise product was therefore not a valid
    # scale fit once this route was applied to saved VUMPS boundaries.
    denominator = real(dot(vec(Mt0), vec(Mt0)))
    kappa = dot(vec(Mt0), vec(H)) / max(denominator, eps(Float64))
    block_residual = norm(H .- kappa .* Mt0) /
                     max(norm(H), abs(kappa) * norm(Mt0), eps(Float64))
    rails = m2_oriented_rails(H, maps.L, maps.R; rtol=1e-12)
    closure_seconds = @elapsed result = m2_contract_shared_six_R(
        rails, Dphysical; nmax, tol, verbose)
    (; result..., method="direct_finite_depth", base_chi=size(ML, 1),
       effective_chi=size(ML, 1), direct_depth=maps.depth,
       direct_map_residual_left=maps.residualL,
       direct_map_residual_right=maps.residualR,
       direct_lambda_left=maps.lambdaL, direct_lambda_right=maps.lambdaR,
       direct_block_scale=Float64(abs(kappa)),
       direct_block_residual=Float64(block_residual),
       direct_metric=G, direct_metric_condition=Float64(cond(G)),
       direct_metric_dropped=dropped,
       direct_projector_refactor_residual=canonical.projector_residual,
       direct_left_isometry_residual=canonical.left_isometry,
       direct_right_isometry_residual=canonical.right_isometry,
       direct_block_norm=Float64(norm(B)),
       direct_closure_seconds=Float64(closure_seconds))
end

"Solve the enlarged-bond diagnostic closure from an already accepted boundary tensor."
function direct_route_closure(Mt::AbstractArray{<:Number,3},
                             Dgrow::AbstractArray{<:Number,3},
                             a::AbstractArray{<:Number,4}, Dphysical::Int;
                             nmax::Int=1200, tol::Real=1e-10,
                             verbose::Bool=true)
    size(Dgrow, 1) == size(Dgrow, 3) ||
        throw(DimensionMismatch("direct grown tensor must have square bonds"))
    size(a, 1) == size(a, 2) == size(a, 3) == size(a, 4) ||
        throw(DimensionMismatch("direct route requires a uniform local MPO"))
    size(Dgrow, 2) == size(a, 1) ||
        throw(DimensionMismatch("direct physical leg and MPO leg differ"))
    N = size(Dgrow, 1)
    N > 0 || throw(ArgumentError("direct effective bond must be positive"))

    # Normalize only the tensor representation.  The six-R scalar combination
    # removes this common one-copy scale; retaining it here prevents a hidden
    # normalization difference from being mistaken for an endpoint identity.
    Ddirect = Array(Dgrow) / max(norm(Dgrow), eps(Float64))
    rails = direct_spatial_rails(Ddirect)

    # These are diagnostics for the actual enlarged transfer, not endpoint-map
    # residuals.  They catch a bad full-bond channel even when the six-R solve
    # happens to return a finite scalar.
    rhoL, lamL = channel_fp(Ddirect, :L; nmax, tol)
    rhoR, lamR = channel_fp(Ddirect, :R; nmax, tol)
    resL = direct_channel_residual(Ddirect, rhoL, lamL, :L)
    resR = direct_channel_residual(Ddirect, rhoR, lamR, :R)

    closure_seconds = @elapsed result = m2_contract_shared_six_R(
        rails, Dphysical; nmax, tol, verbose)
    (; result..., method="direct_uncompressed_diagnostic",
       base_chi=size(Mt, 1), effective_chi=N,
       direct_channel_left_residual=Float64(resL),
       direct_channel_right_residual=Float64(resR),
       direct_lambda_left=Float64(lamL),
       direct_lambda_right=Float64(lamR),
       direct_norm=Float64(norm(Dgrow)),
       direct_closure_seconds=Float64(closure_seconds))
end
