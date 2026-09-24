# Audited local VUMPS LTR objects and the mixed-transfer six-R bridge.
#
# MPSKit's GL/GR are weighted effective environments. They give the exact
# local contractions
#
#   B = L * Dgrow(AC) * R,          G = L * (C on old bond) * R,
#   H_L = B * G^(-1) ~= AL,         H_R = G^(-1) * B ~= AR,
#
# but they are not the unweighted oblique-projector rectangles V_L^dagger/V_R
# required by method 2. Directly closing them gives gauge-dependent S2/S3 even
# when the final linear combination accidentally looks plausible. The public
# audit below preserves the valid local construction.  The physical six-R
# bridge separately solves the two mixed-transfer endpoint rectangles between
# the absorbed boundary and the left-canonical VUMPS fixed point.

struct VUMPSLTRDiagnostics
    block_axis_residual::Float64
    metric_axis_residual::Float64
    left_fixedpoint_residual::Float64
    right_fixedpoint_residual::Float64
    metric_condition::Float64
    metric_dropped::Int
    chi::Int
    ket_dimension::Int
    bra_dimension::Int
    compatible::Bool
end

"Copy a CPU or CUDA TensorMap to a dense host array with explicit factors."
_host_array(tensor) = convert(Array, Adapt.adapt(Array, tensor))

function _checked_rank4(name::Symbol, tensor)
    ndims(tensor) == 4 || throw(DimensionMismatch(
        "$name must have [chi,ket,bra,chi] axes, got $(size(tensor))"
    ))
    all(isfinite, tensor) || throw(ArgumentError("$name contains non-finite entries"))
    tensor
end

"Explicit norm site a[U,L,D,R], fused as ket + D*(bra-1)."
function _dense_norm_site(source::PEPS)
    D = source.D
    a = zeros(ComplexF64, D^2, D^2, D^2, D^2)
    fuse(ket, bra) = ket + D * (bra - 1)
    A = source.A
    for uk in 1:D, ub in 1:D, lk in 1:D, lb in 1:D,
        dk in 1:D, db in 1:D, rk in 1:D, rb in 1:D
        value = zero(ComplexF64)
        for s in 1:source.d
            value += A[s, uk, lk, dk, rk] * conj(A[s, ub, lb, db, rb])
        end
        a[fuse(uk, ub), fuse(lk, lb), fuse(dk, db), fuse(rk, rb)] = value
    end
    a
end

"Dgrow[(l,xL),p,(r,xR)] = sum_u AC[l,u,r] a[u,xL,p,xR]."
function _grow_vumps_row(M, a)
    chi, dp, chir = size(M)
    chi == chir || throw(DimensionMismatch("boundary bonds differ"))
    size(a) == (dp, dp, dp, dp) || throw(DimensionMismatch("row legs differ"))
    Dgrow = zeros(promote_type(eltype(M), eltype(a)), chi * dp, dp, chi * dp)
    for l in 1:chi, xL in 1:dp, p in 1:dp, r in 1:chi, xR in 1:dp
        value = zero(eltype(Dgrow))
        for u in 1:dp
            value += M[l, u, r] * a[u, xL, p, xR]
        end
        Dgrow[l + chi * (xL - 1), p, r + chi * (xR - 1)] = value
    end
    Dgrow
end

"GL[a,k,b,l] -> L[a,(l,k,b)], with l fastest on the composite cut."
function _left_environment_map(GL)
    chi, Dket, Dbra, oldchi = size(GL)
    chi == oldchi || throw(DimensionMismatch("GL retained bonds differ"))
    L = zeros(eltype(GL), chi, chi * Dket * Dbra)
    fuse(k, b) = k + Dket * (b - 1)
    for a in 1:chi, l in 1:chi, k in 1:Dket, b in 1:Dbra
        x = fuse(k, b)
        L[a, l + chi * (x - 1)] = GL[a, k, b, l]
    end
    L
end

"GR[r,k,b,a] -> R[(r,k,b),a], with r fastest on the composite cut."
function _right_environment_map(GR)
    oldchi, Dket, Dbra, chi = size(GR)
    oldchi == chi || throw(DimensionMismatch("GR retained bonds differ"))
    R = zeros(eltype(GR), chi * Dket * Dbra, chi)
    fuse(k, b) = k + Dket * (b - 1)
    for r in 1:chi, k in 1:Dket, b in 1:Dbra, a in 1:chi
        x = fuse(k, b)
        R[r + chi * (x - 1), a] = GR[r, k, b, a]
    end
    R
end

"Insert mixed-canonical C on R's old retained bond only."
function _center_dress_right(R, C, dp)
    chi = size(C, 1)
    size(C) == (chi, chi) || throw(DimensionMismatch("C must be square"))
    size(R) == (chi * dp, chi) || throw(DimensionMismatch("R/C mismatch"))
    RC = zeros(promote_type(eltype(R), eltype(C)), size(R))
    for l in 1:chi, x in 1:dp, a in 1:chi, r in 1:chi
        RC[l + chi * (x - 1), a] +=
            C[l, r] * R[r + chi * (x - 1), a]
    end
    RC
end

function _vumps_local_block(L, Dgrow, R)
    chi, dp = size(L, 1), size(Dgrow, 2)
    B = zeros(promote_type(eltype(L), eltype(Dgrow), eltype(R)), chi, dp, chi)
    for p in 1:dp
        B[:, p, :] = L * (@view Dgrow[:, p, :]) * R
    end
    B
end

function _right_metric(B, Gi)
    H = similar(B)
    for p in axes(B, 2)
        H[:, p, :] = (@view B[:, p, :]) * Gi
    end
    H
end

function _left_metric(Gi, B)
    H = similar(B)
    for p in axes(B, 2)
        H[:, p, :] = Gi * (@view B[:, p, :])
    end
    H
end

function _array_scaled_residual(image, target)
    target_norm2 = real(dot(target, target))
    target_norm2 > eps(Float64) || return Inf
    lambda = dot(target, image) / target_norm2
    Float64(norm(image - lambda * target) /
            max(norm(image), abs(lambda) * norm(target), eps(Float64)))
end

"""
    vumps_ltr_objects(boundary; require_converged=true, tolerance=1e-8)

Construct the part of `LTR` determined unambiguously by MPSKit:

    AC[oldL,u,oldR]                   chi x D^2 x chi
    Dgrow[(oldL,xL),p,(oldR,xR)]     chi*D^2 x D^2 x chi*D^2
    L[newL,(oldL,xL)]                chi x chi*D^2
    R[(oldR,xR),newR]                chi*D^2 x chi
    Rcenter[(oldL,xR),newR]          C inserted on R's old bond
    B[p] = L*Dgrow[p]*R              chi x D^2 x chi
    G    = L*Rcenter                 chi x chi
    Hleft[p]  = B[p]*G^(-1)          reproduces AL
    Hright[p] = G^(-1)*B[p]          reproduces AR

`B` and `G` are checked against PEPSKit's own AC/C effective contractions.
The return value is diagnostic: L/R are weighted GL/GR environments, not the
unweighted method-2 projector rectangles.
"""
function vumps_ltr_objects(
        boundary::VUMPSBoundary;
        require_converged::Bool=true,
        tolerance::Real=1e-8,
        inverse_rtol::Real=1e-12,
    )
    require_converged && !boundary.diagnostics.converged &&
        error("cannot audit LTR from an unconverged VUMPS boundary")
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))

    AL = _checked_rank4(:AL, _host_array(boundary.state.AL[1]))
    AR = _checked_rank4(:AR, _host_array(boundary.state.AR[1]))
    AC = _checked_rank4(:AC, _host_array(boundary.state.AC[1]))
    GL = _checked_rank4(:GL, _host_array(boundary.environments.GLs[1]))
    GR = _checked_rank4(:GR, _host_array(boundary.environments.GRs[end]))
    C = _host_array(boundary.state.C[1])

    chi, Dket, Dbra, chir = size(AC)
    chi == chir || throw(DimensionMismatch("AC retained bonds differ"))
    Dket == Dbra == boundary.source.D || throw(DimensionMismatch(
        "VUMPS ket/bra dimensions $Dket/$Dbra do not match source D=$(boundary.source.D)"
    ))
    size(C) == (chi, chi) || throw(DimensionMismatch("C has shape $(size(C))"))
    dp = Dket * Dbra

    Mcenter = reshape(AC, chi, dp, chi)
    Dgrow = _grow_vumps_row(Mcenter, _dense_norm_site(boundary.source))
    L = _left_environment_map(GL)
    R = _right_environment_map(GR)
    Rcenter = _center_dress_right(R, C, dp)
    B = _vumps_local_block(L, Dgrow, R)
    G = L * Rcenter
    Gi, dropped = m2_metric_inverse(G; rtol=inverse_rtol)
    Hleft = _right_metric(B, Gi)
    Hright = _left_metric(Gi, B)

    hac = MPSKit.AC_hamiltonian(
        1, boundary.state, boundary.transfer, boundary.state, boundary.environments
    )
    hc = MPSKit.C_hamiltonian(
        1, boundary.state, boundary.transfer, boundary.state, boundary.environments
    )
    B_package = reshape(_host_array(hac(boundary.state.AC[1])), chi, dp, chi)
    G_package = _host_array(hc(boundary.state.C[1]))
    block_axis = _array_scaled_residual(B, B_package)
    metric_axis = _array_scaled_residual(G, G_package)
    left_fixed = _array_scaled_residual(Hleft, reshape(AL, chi, dp, chi))
    right_fixed = _array_scaled_residual(Hright, reshape(AR, chi, dp, chi))
    compatible = boundary.diagnostics.converged &&
                 block_axis <= tolerance && metric_axis <= tolerance &&
                 left_fixed <= tolerance && right_fixed <= tolerance
    diagnostics = VUMPSLTRDiagnostics(
        block_axis, metric_axis, left_fixed, right_fixed, cond(G), dropped,
        chi, Dket, Dbra, compatible,
    )
    require_converged && !compatible && error(
        "VUMPS local LTR audit failed: B=$(block_axis), G=$(metric_axis), " *
        "HL=$(left_fixed), HR=$(right_fixed)"
    )
    (; Dgrow, L, R, Rcenter, B, G, Hleft, Hright, diagnostics)
end

struct VUMPSRectangleDiagnostics
    mixed_left_residual::Float64
    mixed_right_residual::Float64
    local_fixedpoint_residual::Float64
    projector_p2_residual::Float64
    projector_left_residual::Float64
    projector_right_residual::Float64
    metric_condition::Float64
    metric_dropped::Int
    mixed_left_iterations::Int
    mixed_right_iterations::Int
    chi::Int
    effective_physical_dimension::Int
    compatible::Bool
end

"Normalize a mixed fixed point and remove its arbitrary scalar phase."
function _phase_normalize_mixed!(next, previous)
    magnitude = norm(next)
    isfinite(magnitude) && magnitude > eps(Float64) ||
        error("mixed transfer produced a zero or non-finite iterate")
    next ./= magnitude
    overlap = dot(previous, next)
    abs(overlap) > eps(Float64) && (next .*= conj(overlap) / abs(overlap))
    next
end

"""
Solve the two opened half-infinite maps between `Dgrow=T*ML` and the natural
mixed-canonical target tensors `ML` and `MR`:

    sum_p ML[p]^dag L Dgrow[p] = lambda_L L,
    sum_p Dgrow[p] R MR[p]^dag = lambda_R R.

The maps have shapes `L: chi x (chi*D_eff)` and
`R: (chi*D_eff) x chi`.  They are the endpoint rectangles used in
`P=R(LR)^(-1)L`; they are not PEPSKit's raw weighted `GL/GR` arrays.
"""
function _mixed_endpoint_rectangles(
        Dgrow, ML, MR=ML;
        nmax::Integer=20_000,
        tolerance::Real=1e-12,
    )
    nmax > 0 || throw(ArgumentError("nmax must be positive"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    N, dp, Nr = size(Dgrow)
    size(ML) == size(MR) || throw(DimensionMismatch(
        "left/right target boundary shapes differ: $(size(ML)) versus $(size(MR))"))
    chi, dpm, chir = size(ML)
    N == Nr || throw(DimensionMismatch("grown boundary bonds differ"))
    chi == chir || throw(DimensionMismatch("target boundary bonds differ"))
    dp == dpm || throw(DimensionMismatch("grown/target physical legs differ"))
    T = promote_type(eltype(Dgrow), eltype(ML), eltype(MR))
    L = ones(T, chi, N); L ./= norm(L)
    R = ones(T, N, chi); R ./= norm(R)
    residualL = Inf; residualR = Inf
    iterationsL = nmax; iterationsR = nmax
    for iteration in 1:nmax
        next = zeros(T, chi, N)
        for p in 1:dp
            next .+= adjoint(@view(ML[:, p, :])) * L *
                     @view(Dgrow[:, p, :])
        end
        residualL = _array_scaled_residual(next, L)
        _phase_normalize_mixed!(next, L)
        L = next
        if residualL <= tolerance
            iterationsL = iteration
            break
        end
    end
    for iteration in 1:nmax
        next = zeros(T, N, chi)
        for p in 1:dp
            next .+= @view(Dgrow[:, p, :]) * R *
                     adjoint(@view(MR[:, p, :]))
        end
        residualR = _array_scaled_residual(next, R)
        _phase_normalize_mixed!(next, R)
        R = next
        if residualR <= tolerance
            iterationsR = iteration
            break
        end
    end
    residualL <= tolerance || error(
        "left mixed rectangle did not converge in $nmax steps: $residualL"
    )
    residualR <= tolerance || error(
        "right mixed rectangle did not converge in $nmax steps: $residualR"
    )
    (; L, R, residualL=Float64(residualL), residualR=Float64(residualR),
       iterationsL, iterationsR)
end

"""
    vumps_projector_objects(boundary; ...)

Construct the factorized projector from the converged VUMPS boundary.  `AL`
is mandatory here: the mixed equation maps an absorbed row back to the
left-canonical target.  Using `AR` or `AC` in this equation fails the local
fixed-point gate by order one.
"""
function vumps_projector_objects(
        boundary::VUMPSBoundary;
        require_converged::Bool=true,
        tolerance::Real=1e-7,
        mixed_tolerance::Real=1e-12,
        mixed_nmax::Integer=20_000,
        inverse_rtol::Real=1e-12,
    )
    require_converged && !boundary.diagnostics.converged &&
        error("cannot build rectangles from an unconverged VUMPS boundary")
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    AL = _checked_rank4(:AL, _host_array(boundary.state.AL[1]))
    chi, Dket, Dbra, chir = size(AL)
    chi == chir || throw(DimensionMismatch("AL retained bonds differ"))
    Dket == Dbra == boundary.source.D || throw(DimensionMismatch(
        "AL ket/bra dimensions $Dket/$Dbra do not match source D=$(boundary.source.D)"
    ))
    dp = Dket * Dbra
    M = reshape(AL, chi, dp, chi)
    Dgrow = _grow_vumps_row(M, _dense_norm_site(boundary.source))
    mixed = _mixed_endpoint_rectangles(
        Dgrow, M; nmax=mixed_nmax, tolerance=mixed_tolerance,
    )
    L, R = mixed.L, mixed.R
    G = L * R
    Gi, dropped = m2_metric_inverse(G; rtol=inverse_rtol)
    B = _vumps_local_block(L, Dgrow, R)
    H = _right_metric(B, Gi)
    fixedpoint = _array_scaled_residual(H, M)

    P = R * Gi * L
    p2 = norm(P * P - P) / max(norm(P), eps(Float64))
    lp = norm(L * P - L) / max(norm(L), eps(Float64))
    pr = norm(P * R - R) / max(norm(R), eps(Float64))
    compatible = boundary.diagnostics.converged && dropped == 0 &&
                 mixed.residualL <= mixed_tolerance &&
                 mixed.residualR <= mixed_tolerance &&
                 fixedpoint <= tolerance && p2 <= tolerance &&
                 lp <= tolerance && pr <= tolerance
    diagnostics = VUMPSRectangleDiagnostics(
        mixed.residualL, mixed.residualR, fixedpoint,
        Float64(p2), Float64(lp), Float64(pr), Float64(cond(G)), dropped,
        mixed.iterationsL, mixed.iterationsR, chi, dp, compatible,
    )
    require_converged && !compatible && error(
        "VUMPS rectangle audit failed: mixedL=$(mixed.residualL), " *
        "mixedR=$(mixed.residualR), H=$(fixedpoint), P2=$(p2), " *
        "LP=$(lp), PR=$(pr), dropped=$dropped"
    )
    (; M, Dgrow, L, R, G, B, H, diagnostics)
end

"Four globally oriented rails built from one audited oblique projector."
function vumps_sixr_rails(boundary::VUMPSBoundary; kwargs...)
    objects = vumps_projector_objects(boundary; kwargs...)
    m2_oriented_rails(objects.H, objects.L, objects.R)
end

function solve_vumps_sixr(
        source::PEPS, chi::Integer;
        backend::Symbol=:cpu,
        boundary_kwargs=(;),
        bridge_kwargs=(;),
        sixr_kwargs=(;),
    )
    local boundary, objects, rails, result
    boundary_seconds = @elapsed boundary = solve_vumps_boundary(
        source, chi; backend, boundary_kwargs...
    )
    rectangle_seconds = @elapsed begin
        objects = vumps_projector_objects(boundary; bridge_kwargs...)
        rails = m2_oriented_rails(objects.H, objects.L, objects.R)
    end
    closure_seconds = @elapsed result = m2_contract_shared_six_R(
        rails, source.D; sixr_kwargs...
    )
    (; boundary, objects, rails, result,
       boundary_seconds, rectangle_seconds, closure_seconds,
       total_seconds=boundary_seconds + rectangle_seconds + closure_seconds)
end
