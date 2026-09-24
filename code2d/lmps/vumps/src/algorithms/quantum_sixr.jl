# Generic quantum-PEPS bridge to the ITensor six-R closure.
#
# The VUMPS boundary state is a one-copy boundary of the PEPS norm transfer
# operator.  Its local physical space is already the fused ket/bra virtual
# space (D^2).  We deliberately keep that fused leg intact while constructing
# the LMPS rails; only `m2_split_boundary` opens it again for the replica
# labels.  No spin-flip, reflection, or rotation symmetry is assumed here.
# A, B, and C retain independent boundary gauges all the way to the one-copy
# mixed seam solves K_AB, K_AC, and K_BC.

"Flatten a TensorKit boundary tensor to [left, physical, right]."
function _quantum_boundary_array(tensor)
    raw = _host_array(tensor)
    if ndims(raw) == 3
        chi, dp, chir = size(raw)
        chi == chir || throw(DimensionMismatch(
            "quantum boundary bonds differ: $(size(raw))"
        ))
        return raw
    elseif ndims(raw) == 4
        # Older MPSKit/PEPSKit combinations expose the ProductSpace physical
        # leg as two explicit factors [left, ket, bra, right].
        chi, dket, dbra, chir = size(raw)
        chi == chir || throw(DimensionMismatch(
            "quantum boundary bonds differ: $(size(raw))"
        ))
        return reshape(raw, chi, dket * dbra, chi)
    end
    throw(DimensionMismatch(
        "quantum boundary tensor must have 3 or 4 dense axes, got $(size(raw))"
    ))
end

"Return the one-site boundary tensor as [left, fused ket/bra, right]." 
function _quantum_boundary_tensor(boundary::QuantumVUMPSBoundary)
    nsites = length(boundary.transfer)
    nsites == 1 && return _quantum_boundary_array(boundary.state.AL[1])
    throw(ArgumentError(
        "quantum six-R currently requires a one-site PEPS; got row period $nsites. " *
        "A multi-site cell must use multiline VUMPS and a full two-dimensional block."
    ))
end

"Dense double-layer MPO site in the row convention a[north,west,south,east]."
function _quantum_transfer_site(boundary::QuantumVUMPSBoundary)
    top, bottom = boundary.transfer[1]
    # Do not use `mpotensor` here: that helper inserts TensorKit fuse
    # isomorphisms and changes the physical ProductSpace basis.  The VUMPS
    # boundary itself is initialized on the *unfused* north ProductSpace, so
    # the local double layer must be assembled in that same basis first.
    A = _host_array(top)
    B = _host_array(bottom)
    ndims(A) == 5 && ndims(B) == 5 || throw(DimensionMismatch(
        "PEPS site tensors must have [physical,north,east,south,west] axes"
    ))
    size(A) == size(B) || throw(DimensionMismatch("ket and bra site shapes differ"))
    d, n, e, s, w = size(A)
    n == e == s == w || throw(DimensionMismatch(
        "six-R currently requires a uniform virtual bond, got $(size(A))"
    ))
    dp = n^2
    a = zeros(promote_type(eltype(A), eltype(B)), dp, dp, dp, dp)
    fuse(i, j) = i + n * (j - 1)
    for nk in 1:n, nb in 1:n, ek in 1:n, eb in 1:n,
        sk in 1:n, sb in 1:n, wk in 1:n, wb in 1:n
        value = zero(eltype(a))
        for physical in 1:d
            value += A[physical, nk, ek, sk, wk] *
                     conj(B[physical, nb, eb, sb, wb])
        end
        a[fuse(nk, nb), fuse(wk, wb), fuse(sk, sb), fuse(ek, eb)] = value
    end
    all(isfinite, a) || throw(ArgumentError("double-layer site contains non-finite entries"))
    a
end

"Apply one PEPS norm row to a boundary tensor in TensorKit."
function _quantum_grow_boundary_tensor(boundary::QuantumVUMPSBoundary, A)
    nsites = length(boundary.transfer)
    nsites == 1 || throw(ArgumentError(
        "quantum six-R currently requires a one-site PEPS; got row period $nsites"
    ))
    O = boundary.transfer[1]
    # A has the exact MPSKit/PEPSKit form
    #   A[left; north_ket north_bra right].
    # The output retains the old-left and west ket/bra factors on its left,
    # and the south ket/bra plus old-right/east ket/bra factors on its right.
    # Keeping this contraction typed is essential: TensorKit inserts the dual
    # twists that a dense ProductSpace reshape cannot represent.
    @tensor grown[left west_ket west_bra;
                  south_ket south_bra right east_ket east_bra] :=
        A[left north_ket north_bra right] *
        PEPSKit.ket(O)[physical;
            north_ket east_ket south_ket west_ket] *
        conj(PEPSKit.bra(O)[physical;
            north_bra east_bra south_bra west_bra])
    raw = _host_array(grown)
    ndims(raw) == 8 || throw(DimensionMismatch(
        "TensorKit T*M contraction must have eight explicit axes, got $(size(raw))"
    ))
    chi, dwk, dwb, dsk, dsb, chir, dek, deb = size(raw)
    chi == chir || throw(DimensionMismatch("grown boundary old bonds differ"))
    dwk == dwb == dsk == dsb == dek == deb || throw(DimensionMismatch(
        "grown boundary has nonuniform virtual factors: $(size(raw))"
    ))
    reshape(raw, chi * dwk * dwb, dsk * dsb, chi * dek * deb)
end

"Apply one PEPS norm row to the left-canonical boundary tensor."
_quantum_grow_boundary_tensor(boundary::QuantumVUMPSBoundary) =
    _quantum_grow_boundary_tensor(boundary, boundary.state.AL[1])

"""
Construct one common-chart generalized factorization of the converged VUMPS
LMPS.

The mixed-canonical identities are `AC = AL*C = C*AR`.  MPSKit computes the
left half-infinite environment in the `AL-O-AL` channel and the right one in
the `AR-O-AR` channel.  Absorbing `C` into the enlarged old bond gives

    Rend = kron(C, I_dp) * Renv,
    B[p] = Lenv * Dgrow(AL)[p] * Rend,
    G    = Lenv * Rend,
    B[p] ~= kappa * AL[p] * G.

Here `kappa` is the dominant one-row transfer eigenvalue in the normalization
used for the endpoint maps.  Strip free-energy ratios are insensitive to this
overall scale, but reconstructing the same normalized LMPS tensor requires
`Hreconstructed = (B*G^+)/kappa`.

These endpoint maps solve the mixed-canonical direct ladders

    L' = sum_p AL[p]^dagger L Dgrow(AL)[p],
    R' = sum_p Dgrow(AL)[p] R AR[p]^dagger.

Forcing the right reference tensor to `AL` instead of `AR` is a different
all-left-canonical channel.  It requires an additional output-bond gauge
factor, worsens the metric conditioning, and is not the factorization used by
the oriented rails below.  `m2_oriented_rails` transports the resulting bridge
metric `G : AR -> AL` into one common retained-bond frame.

The last relation is the numerically stable statement.  Forming `B/G` can
amplify the finite VUMPS residual by `cond(G)` and therefore need not recover
the saved full-chi `AL` array in weak Schmidt directions.  `AL` is retained as
the stable representative; `B/G` is exposed only as a reconstruction audit.

It is important that `B` and `G` use the *same* endpoint maps.  The equivalent
center-chart contraction `Lenv*Dgrow(AC)*Renv` is retained as an independent
axis and center-placement audit.
"""
function quantum_vumps_ltr_objects(
        boundary::QuantumVUMPSBoundary;
        require_converged::Bool=true,
        tolerance::Real=1e-7,
        inverse_rtol::Real=1e-12,
        polish_endpoints::Bool=true,
        endpoint_polish_nmax::Integer=2_000,
        endpoint_polish_tolerance::Real=1e-12,
    )
    require_converged && !boundary.converged &&
        error("cannot build quantum LTR objects from an unconverged boundary")
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    endpoint_polish_nmax > 0 || throw(ArgumentError(
        "endpoint_polish_nmax must be positive"))
    endpoint_polish_tolerance > 0 || throw(ArgumentError(
        "endpoint_polish_tolerance must be positive"))

    AL = _quantum_boundary_array(boundary.state.AL[1])
    AR = _quantum_boundary_array(boundary.state.AR[1])
    AC = _quantum_boundary_array(boundary.state.AC[1])
    chi, dp, chir = size(AC)
    chi == chir || throw(DimensionMismatch("quantum AC bonds differ"))
    size(AL) == size(AR) == size(AC) || throw(DimensionMismatch(
        "quantum AL/AR/AC dense shapes differ"))

    # GL and GR are the two complete half-infinite contractions returned by
    # the same VUMPS solve.  They must be kept together with AC and C; solving
    # unrelated mixed endpoint channels is a different finite-chi object.
    GL = _host_array(boundary.environments.GLs[1])
    GR = _host_array(boundary.environments.GRs[end])
    L = _left_environment_map(GL)
    R = _right_environment_map(GR)
    C = _host_array(boundary.state.C[1])
    size(C) == (chi, chi) || throw(DimensionMismatch(
        "quantum center matrix has shape $(size(C)), expected $((chi, chi))"))

    Dgrow = _quantum_grow_boundary_tensor(boundary, boundary.state.AL[1])
    size(Dgrow) == (chi * dp, dp, chi * dp) || throw(DimensionMismatch(
        "quantum AL-grown tensor has shape $(size(Dgrow)), " *
        "expected $((chi * dp, dp, chi * dp))"))
    size(L) == (chi, chi * dp) || throw(DimensionMismatch(
        "quantum left environment map has shape $(size(L))"))
    size(R) == (chi * dp, chi) || throw(DimensionMismatch(
        "quantum right environment map has shape $(size(R))"))

    Lraw = L
    Renv = R
    Rendraw = _center_dress_right(Renv, C, dp)

    # GL/GR are already environment fixed points, but only to the tolerance of
    # the saved boundary solve.  The six-R closure can amplify a 1e-8 tail
    # error into a visibly different component.  Polish the same AL/AR mixed
    # equations before using the endpoint maps; retain the raw maps and their
    # residuals below so this numerical refinement remains auditable.
    raw_endpoint_probe = direct_finite_depth_maps(
        AL, AR, Dgrow;
        depth=1, min_depth=1, tol=eps(Float64),
        initial_L=Lraw, initial_R=Rendraw,
    )
    polished = polish_endpoints ? direct_finite_depth_maps(
        AL, AR, Dgrow;
        depth=Int(endpoint_polish_nmax), min_depth=1,
        tol=endpoint_polish_tolerance,
        initial_L=Lraw, initial_R=Rendraw,
    ) : (; L=Lraw, R=Rendraw, depth=0,
          residualL=raw_endpoint_probe.residualL,
          residualR=raw_endpoint_probe.residualR)
    L = polished.L
    Rend = polished.R
    B = _vumps_local_block(L, Dgrow, Rend)
    G = L * Rend
    Gi, dropped = m2_metric_inverse(G; rtol=inverse_rtol)
    Hinverse = _right_metric(B, Gi)
    AL_G = _right_metric(AL, G)
    generalized_scale = dot(AL_G, B) / dot(AL_G, AL_G)
    abs(generalized_scale) > eps(Float64) || error(
        "quantum VUMPS generalized LMPS scale vanished")
    Hreconstructed = Hinverse / generalized_scale
    support_projector = G * Gi
    complement = Matrix{eltype(support_projector)}(
        I, chi, chi) - support_projector
    Hcomplete = Hreconstructed + _right_metric(AL, complement)
    inverse_left_fixedpoint_residual = _array_scaled_residual(Hinverse, AL)
    reconstructed_left_fixedpoint_residual =
        _array_scaled_residual(Hreconstructed, AL)
    complete_left_fixedpoint_residual = _array_scaled_residual(Hcomplete, AL)
    common_bg_residual = _array_scaled_residual(B, _right_metric(AL, G))
    endpoint_probe = direct_finite_depth_maps(
        AL, AR, Dgrow;
        depth=1, min_depth=1, tol=eps(Float64),
        initial_L=L, initial_R=Rend,
    )

    # `B=kappa*AL*G` is the rank-safe generalized equation.  Do not overwrite AL by
    # explicitly applying a pseudoinverse to tiny Schmidt directions: those
    # directions are numerically arbitrary but can badly pollute the full
    # transfer spectrum while leaving every closed scalar unchanged.
    Hleft = AL

    # Equivalent center-chart expression.  This is a stringent axis and
    # center-placement gate for AC = AL*C on the grown right bond.
    Dgrow_center = _quantum_grow_boundary_tensor(boundary, boundary.state.AC[1])
    B_center_chart = _vumps_local_block(Lraw, Dgrow_center, Renv)
    Braw = _vumps_local_block(Lraw, Dgrow, Rendraw)
    Graw = Lraw * Rendraw
    center_chart_residual = _array_scaled_residual(B_center_chart, Braw)

    # Authoritative contraction of the same infinite network.  This keeps
    # TensorKit's ProductSpace dual arrows intact and therefore gates the
    # manual dense GL/GR flattening above.
    hac = MPSKit.AC_hamiltonian(
        1, boundary.state, boundary.transfer, boundary.state,
        boundary.environments)
    hc = MPSKit.C_hamiltonian(
        1, boundary.state, boundary.transfer, boundary.state,
        boundary.environments)
    B_package_tensor = hac(boundary.state.AC[1])
    G_package_tensor = hc(boundary.state.C[1])
    # MPSKit's canonical identity is AC = AL * C.  TensorMap right division
    # is the corresponding BG^{-1}; converting B and G separately to Array
    # first loses which virtual space is a domain and which is a dual codomain.
    # Even this correctly oriented division is not a stable full-chi inverse
    # when G is ill-conditioned: it is a diagnostic, not the production AL.
    Hleft_package_tensor = B_package_tensor / G_package_tensor
    B_package = _quantum_boundary_array(B_package_tensor)
    G_package = _host_array(G_package_tensor)
    Hleft_package = _quantum_boundary_array(Hleft_package_tensor)
    size(B_package) == size(AC) || throw(DimensionMismatch(
        "MPSKit AC image has shape $(size(B_package))"))
    size(G_package) == (chi, chi) || throw(DimensionMismatch(
        "MPSKit C image has shape $(size(G_package))"))
    _, package_dropped = m2_metric_inverse(G_package; rtol=inverse_rtol)
    package_left_fixedpoint_residual =
        _array_scaled_residual(Hleft_package, AL)
    block_axis_residual = _array_scaled_residual(Braw, B_package)
    metric_axis_residual = _array_scaled_residual(Graw, G_package)
    # A rank-deficient center is legitimate: the VUMPS tensor is defined on
    # its Schmidt support, and `m2_metric_inverse` records the discarded null
    # complement.  Requiring zero discarded directions would reject exact
    # product/low-rank states for purely representational reasons.
    compatible = boundary.converged &&
                 common_bg_residual <= tolerance &&
                 endpoint_probe.residualL <= tolerance &&
                 endpoint_probe.residualR <= tolerance &&
                 center_chart_residual <= tolerance &&
                 block_axis_residual <= tolerance &&
                 metric_axis_residual <= tolerance
    diagnostics = (;
        common_bg_residual=Float64(common_bg_residual),
        raw_direct_left_fixedpoint_residual=
            Float64(raw_endpoint_probe.residualL),
        raw_direct_right_fixedpoint_residual=
            Float64(raw_endpoint_probe.residualR),
        direct_left_fixedpoint_residual=Float64(endpoint_probe.residualL),
        direct_right_fixedpoint_residual=Float64(endpoint_probe.residualR),
        endpoint_polish_depth=polished.depth,
        inverse_left_fixedpoint_residual=
            Float64(inverse_left_fixedpoint_residual),
        generalized_scale=generalized_scale,
        reconstructed_left_fixedpoint_residual=
            Float64(reconstructed_left_fixedpoint_residual),
        complete_left_fixedpoint_residual=
            Float64(complete_left_fixedpoint_residual),
        center_chart_residual=Float64(center_chart_residual),
        package_left_fixedpoint_residual=Float64(package_left_fixedpoint_residual),
        block_axis_residual=Float64(block_axis_residual),
        metric_axis_residual=Float64(metric_axis_residual),
        metric_condition=Float64(cond(G)), metric_dropped=dropped,
        package_metric_condition=Float64(cond(G_package)),
        package_metric_dropped=package_dropped,
        compatible,
    )
    require_converged && !compatible && error(
        "quantum VUMPS LTR audit failed: package HL=" *
        "$(package_left_fixedpoint_residual), dropped=$package_dropped")
    a = _quantum_transfer_site(boundary)
    (; M=AL, a, AL, AR, AC, C, Dgrow, Dgrow_center,
       L, R=Rend, Rend, Lraw, Renv, Rendraw, polished,
       B, Braw, B_center_chart, G, Graw, H=Hleft, Hleft, Hinverse,
       Hreconstructed, Hcomplete, support_projector, generalized_scale,
       B_package, G_package, Hleft_package, diagnostics)
end

"Build the generic endpoint rectangles from a quantum VUMPS fixed point."
function quantum_vumps_projector_objects(
        boundary::QuantumVUMPSBoundary;
        require_converged::Bool=true,
        tolerance::Real=1e-7,
        mixed_tolerance::Real=1e-10,
        mixed_nmax::Integer=20_000,
        inverse_rtol::Real=1e-12,
    )
    require_converged && !boundary.converged &&
        error("cannot build quantum rectangles from an unconverged VUMPS boundary")
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    mixed_tolerance > 0 || throw(ArgumentError("mixed_tolerance must be positive"))

    AL = _quantum_boundary_array(boundary.state.AL[1])
    AR = _quantum_boundary_array(boundary.state.AR[1])
    size(AL) == size(AR) || throw(DimensionMismatch(
        "quantum AL/AR dense shapes differ"))
    M = AL
    chi, dp, chir = size(AL)
    dvirt = isqrt(dp)
    dvirt^2 == dp || throw(DimensionMismatch("fused boundary dimension $dp is not a square"))
    a = _quantum_transfer_site(boundary)
    size(a) == (dp, dp, dp, dp) || throw(DimensionMismatch(
        "boundary physical leg $dp does not match effective site $(size(a))"
    ))
    Dgrow = _quantum_grow_boundary_tensor(boundary, boundary.state.AL[1])
    size(Dgrow) == (chi * dvirt^2, dp, chi * dvirt^2) ||
        throw(DimensionMismatch("TensorKit grown boundary has size $(size(Dgrow)); " *
            "expected old bonds multiplied by the PEPS virtual pair"))
    mixed = _mixed_endpoint_rectangles(
        Dgrow, AL, AR; nmax=mixed_nmax, tolerance=mixed_tolerance,
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
    compatible = boundary.converged && dropped == 0 &&
                 mixed.residualL <= mixed_tolerance &&
                 mixed.residualR <= mixed_tolerance && fixedpoint <= tolerance &&
                 p2 <= tolerance && lp <= tolerance && pr <= tolerance
    diagnostics = (; mixed_left_residual=Float64(mixed.residualL),
        mixed_right_residual=Float64(mixed.residualR),
        local_fixedpoint_residual=Float64(fixedpoint),
        projector_p2_residual=Float64(p2),
        projector_left_residual=Float64(lp),
        projector_right_residual=Float64(pr),
        metric_condition=Float64(cond(G)), metric_dropped=dropped,
        chi, effective_physical_dimension=dp, compatible)
    require_converged && !compatible && error(
        "quantum rectangle audit failed: mixedL=$(mixed.residualL), " *
        "mixedR=$(mixed.residualR), H=$fixedpoint, P2=$p2, " *
        "LP=$lp, PR=$pr, dropped=$dropped"
    )
    (; M, AL, AR, a, Dgrow, L, R, G, B, H, diagnostics)
end

"""Construct four rails from the audited common-chart VUMPS factorization.

The legacy `mixed_tolerance` and `mixed_nmax` keywords are accepted so old job
files remain readable, but no independent mixed endpoint eigenproblem is
solved.  `route=:legacy_mixed` is available only for explicit comparisons.
"""
function quantum_vumps_sixr_rails(
        boundary::QuantumVUMPSBoundary;
        route::Symbol=:common_chart,
        mixed_tolerance::Real=1e-10,
        mixed_nmax::Integer=20_000,
        kwargs...,
    )
    objects = if route === :common_chart
        quantum_vumps_ltr_objects(boundary; kwargs...)
    elseif route === :legacy_mixed
        quantum_vumps_projector_objects(
            boundary; mixed_tolerance, mixed_nmax, kwargs...)
    else
        throw(ArgumentError(
            "quantum rail route must be :common_chart or :legacy_mixed"))
    end
    rails = m2_oriented_rails(objects.H, objects.L, objects.R)
    (; boundary, objects, rails)
end

"Solve one region repeatedly and retain the fixed point with the best diagnostics."
function _best_quantum_region_boundary(
        peps, chi, side, base_seed::Integer, trials::Integer, boundary_kwargs;
        boundary_accept=nothing,
    )
    trials > 0 || throw(ArgumentError("boundary_trials must be positive"))
    # A retry must return its diagnostics instead of throwing immediately.
    kwargs = merge(boundary_kwargs, (; accept_unconverged=true))
    best = nothing
    best_score = Inf
    for attempt in 0:(trials - 1)
        candidate = solve_quantum_vumps_boundary(
            peps, chi; side, seed=base_seed + attempt, kwargs...
        )
        score = maximum((candidate.galerkin_residual,
            candidate.left_environment_residual,
            candidate.right_environment_residual,
            candidate.center_residual))
        accepted = boundary_accept === nothing || boundary_accept(candidate)
        if accepted && score < best_score
            best, best_score = candidate, score
        end
        accepted && candidate.converged && return candidate
    end
    best === nothing && error(
        "none of the $trials boundary candidates passed the requested " *
        "sector/branch filter for side=$side chi=$chi")
    best
end

"""
    solve_quantum_vumps_sixr(peps, chi; ...)

Solve the quantum six-R geometry with independent LMPS fixed points for the
three spatial regions.  The default layout is A=north and B=north (two
independent solves of the upper half-plane transfer) and C=south (the lower
straight boundary).  These three solves audit directions and symmetry sectors.

With `assume_spatial_symmetry=false` (the default), the three regional rail
frames are kept distinct.  The ordinary one-copy seam fixed points are the
inter-boundary maps K_AB, K_AC, and K_BC; their tensor products initialize the
six twisted two-copy R tensors.  No retained bond of one region is identified
with a different region by a bare delta.

With `assume_spatial_symmetry=true`, the PEPS is declared to have the required
rotation/reflection symmetry.  The boundary and rectangle are then solved only
once and the historical shared-rail closure is used.  This is the Ising-style
fast path; it deliberately does not repeat equivalent A/B/C solves.

This is the generic quantum route.  It accepts a PEPSKit `InfinitePEPS`
already prepared by a ground-state algorithm; it does not assume that the
PEPS tensor is real, reflection symmetric, or spin-flip invariant.
"""
function solve_quantum_vumps_sixr(
        peps::PEPSKit.InfinitePEPS, chi::Integer;
        A_side::Symbol=:north, B_side::Symbol=:north, C_side::Symbol=:south,
        boundary_seeds::Tuple{Int,Int,Int}=(10_001, 20_001, 30_001),
        boundary_trials::Integer=1,
        assume_spatial_symmetry::Bool=false,
        boundary_accept=nothing,
        boundary_kwargs=(;), bridge_kwargs=(;), sixr_kwargs=(;),
    )
    local A, B, C, ao, bo, co, rails, closure_rails, result
    local boundary_seconds, rectangle_seconds
    if assume_spatial_symmetry
        boundary_seconds = @elapsed A = _best_quantum_region_boundary(
            peps, chi, A_side, boundary_seeds[1], boundary_trials,
            boundary_kwargs; boundary_accept,
        )
        B = C = A
        rectangle_seconds = @elapsed ao = quantum_vumps_sixr_rails(
            A; bridge_kwargs...
        )
        bo = co = ao
        rails = (A=ao.rails, B=ao.rails, C=ao.rails)
    else
        boundary_seconds = @elapsed begin
            # Every region is a separate fixed-point problem. This includes A
            # and B even when they happen to use the same north-facing MPO.
            A = _best_quantum_region_boundary(
                peps, chi, A_side, boundary_seeds[1], boundary_trials,
                boundary_kwargs; boundary_accept,
            )
            B = _best_quantum_region_boundary(
                peps, chi, B_side, boundary_seeds[2], boundary_trials,
                boundary_kwargs; boundary_accept,
            )
            C = _best_quantum_region_boundary(
                peps, chi, C_side, boundary_seeds[3], boundary_trials,
                boundary_kwargs; boundary_accept,
            )
        end
        rectangle_seconds = @elapsed begin
            ao = quantum_vumps_sixr_rails(A; bridge_kwargs...)
            bo = quantum_vumps_sixr_rails(B; bridge_kwargs...)
            co = quantum_vumps_sixr_rails(C; bridge_kwargs...)
            dp = size(ao.objects.M, 2)
            size(bo.objects.M, 2) == dp && size(co.objects.M, 2) == dp ||
                throw(DimensionMismatch(
                    "A/B/C boundaries have different fused physical dimensions"
                ))
            rails = (A=ao.rails, B=bo.rails, C=co.rails)
        end
    end
    D2 = size(ao.objects.a, 1)
    D = isqrt(D2)
    D^2 == D2 || throw(DimensionMismatch("double-layer virtual dimension $D2 is not a square"))
    local closure_seconds
    if assume_spatial_symmetry
        closure_rails = ao.rails
        closure_seconds = @elapsed result = m2_contract_shared_six_R(
            closure_rails, D; sixr_kwargs...
        )
    else
        # Keep every octahedron edge in its owning regional frame. The k=1
        # mixed solves construct K_AB, K_AC, and K_BC between those gauges.
        closure_rails = rails
        closure_seconds = @elapsed result = m2_contract_regional_six_R(
            closure_rails, D; sixr_kwargs...
        )
    end
    (; boundaries=(A=A, B=B, C=C), objects=(A=ao.objects, B=bo.objects, C=co.objects),
       rails, closure_rails, result, boundary_seconds, rectangle_seconds, closure_seconds,
       total_seconds=boundary_seconds + rectangle_seconds + closure_seconds,
       D, fused_virtual_dimension=D2, assume_spatial_symmetry)
end
