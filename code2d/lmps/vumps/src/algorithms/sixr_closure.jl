# sixr_closure.jl -- named-index six-R replica closure engine.
# =============================================================================
# This file contains only the replica machinery used by production. It accepts
# either four shared one-copy rails (hL,hR,vL,vR), or three independently
# gauged regional rail sets (A,B,C). It solves exactly the six rank-four
# endpoints in SIXR_DESCRIPTORS and reuses them in Z4 and the three Z2 sectors.
# It does not choose a spatial LMPS ansatz.
# =============================================================================
isdefined(@__MODULE__, :SIXR_DESCRIPTORS) ||
    include(joinpath(@__DIR__, "sixr.jl"))
using ITensors, KrylovKit, LinearAlgebra, Printf, Random

const M2_SIGMA_A = [1, 2, 3, 4]
const M2_SIGMA_B = [2, 1, 4, 3]
const M2_SIGMA_C = [3, 4, 1, 2]

struct M2Region
    name::String
    sigma::Vector{Int}
end

# -----------------------------------------------------------------------------
# Four oriented local rails
# -----------------------------------------------------------------------------

function m2_horizontal_left(M::AbstractArray{<:Number,3})
    chi, dp, chir = size(M)
    chi == chir || throw(DimensionMismatch("horizontal rail bonds differ"))
    ITensor(M, Index(chi, "M2Rail,hL,L"), Index(dp, "M2Rail,hL,P"),
            Index(chi, "M2Rail,hL,R"))
end

function m2_horizontal_right(M::AbstractArray{<:Number,3})
    chi, dp, chir = size(M)
    chi == chir || throw(DimensionMismatch("horizontal rail bonds differ"))
    data = permutedims(M, (3, 2, 1))
    ITensor(data, Index(chi, "M2Rail,hR,L"), Index(dp, "M2Rail,hR,P"),
            Index(chi, "M2Rail,hR,R"))
end

"L[a,(l,x)] becomes the left vertical rail V_L[l,x,a]."
function m2_vertical_left(L::AbstractMatrix{<:Number}, chi::Int, dp::Int)
    size(L) == (chi, chi * dp) ||
        throw(DimensionMismatch("left rail expected $(chi)x$(chi*dp), got $(size(L))"))
    il = Index(chi, "M2Rail,vL,L")
    ip = Index(dp, "M2Rail,vL,P")
    ir = Index(chi, "M2Rail,vL,R")
    V = ITensor(zeros(eltype(L), chi, dp, chi), il, ip, ir)
    for l in 1:chi, x in 1:dp, a in 1:chi
        V[il => l, ip => x, ir => a] = L[a, l + chi * (x - 1)]
    end
    V
end

"`Rcommon[(l,x),a]` becomes the right rail in the horizontal retained basis."
function m2_vertical_right(Rcommon::AbstractMatrix{<:Number}, chi::Int, dp::Int)
    size(Rcommon) == (chi * dp, chi) ||
        throw(DimensionMismatch("right rail expected $(chi*dp)x$(chi), got $(size(Rcommon))"))
    il = Index(chi, "M2Rail,vR,L")
    ip = Index(dp, "M2Rail,vR,P")
    ir = Index(chi, "M2Rail,vR,R")
    V = ITensor(zeros(eltype(Rcommon), chi, dp, chi), il, ip, ir)
    # `Rcommon` has already been transported from the independently retained
    # right basis to the L/Mt basis by Rcommon = R*(L*R)^(-1).  Its two array
    # axes therefore have the same meaning as the bonds of the horizontal Mt.
    # Exchanging l and a here would apply the spatial reflection twice.
    for l in 1:chi, x in 1:dp, a in 1:chi
        V[il => l, ip => x, ir => a] = Rcommon[l + chi * (x - 1), a]
    end
    V
end

"Truncated SVD inverse used only to put the two endpoint maps in one basis."
function m2_metric_inverse(G::AbstractMatrix{<:Number}; rtol::Real=1e-12)
    size(G, 1) == size(G, 2) ||
        throw(DimensionMismatch("endpoint metric must be square, got $(size(G))"))
    rtol > 0 || throw(ArgumentError("rtol must be positive, got $rtol"))
    F = svd(Matrix(G))
    isempty(F.S) && error("cannot invert an empty endpoint metric")
    cutoff = Float64(rtol) * maximum(F.S)
    keep = F.S .> cutoff
    any(keep) || error("endpoint metric has no singular value above $cutoff")
    Sinv = map((s, k) -> k ? inv(s) : 0.0, F.S, keep)
    F.V * Diagonal(Sinv) * F.U', count(!, keep)
end

"""
    m2_oriented_rails(Mt, L, R; rtol=1e-12)

Construct four rails in one *global* retained-bond frame.  `L` and `R` come
from independent channel eigenspaces, so a raw `R` leg cannot be joined to an
`Mt = (LDR)(LR)^(-1)` leg.  The right endpoint must first be transported:

    G = L*R,             Rcommon = R*G^(-1).

The frozen Ising reference route uses seam-local outward charts instead and is
implemented by `production/reference_rails.jl`; it deliberately does not call
this global-frame constructor.
"""
function m2_oriented_rails(Mt, L, R; rtol::Real=1e-12)
    chi, dp = size(Mt, 1), size(Mt, 2)
    size(Mt, 3) == chi || throw(DimensionMismatch("Mt bonds differ"))
    size(L) == (chi, chi * dp) ||
        throw(DimensionMismatch("L expected $(chi)x$(chi*dp), got $(size(L))"))
    size(R) == (chi * dp, chi) ||
        throw(DimensionMismatch("R expected $(chi*dp)x$(chi), got $(size(R))"))
    Gi, dropped = m2_metric_inverse(L * R; rtol)
    Rcommon = R * Gi
    (hL=m2_horizontal_left(Mt), hR=m2_horizontal_right(Mt),
     vL=m2_vertical_left(L, chi, dp),
     vR=m2_vertical_right(Rcommon, chi, dp),
     right_metric_dropped=dropped)
end

"""
    m2_rail(seam, region, rails)

Select the oriented rail incident on one seam. `rails` may either be one
shared four-rail object (the historical isotropic route), or a regional tuple

    (A=rails_A, B=rails_B, C=rails_C)

whose entries were built independently. In the regional route every logical
edge of the octahedron remains in its owning region's retained-bond frame;
the one-copy seam fixed points subsequently computed by `m2_specs` are the
mixed maps `K_AB`, `K_AC`, and `K_BC` between those frames.
"""
function m2_rail(seam::String, region::String, rails)
    regional = (hasproperty(rails, :A), hasproperty(rails, :B),
                hasproperty(rails, :C))
    any(regional) && !all(regional) && error(
        "regional rails must provide all of A, B, and C"
    )
    local r = all(regional) ? getproperty(rails, Symbol(region)) : rails
    all(hasproperty(r, ray) for ray in (:hL, :hR, :vL, :vR)) || error(
        "region $region does not provide hL, hR, vL, and vR rails"
    )
    if seam == "AB"
        region == "A" && return r.vL
        region == "B" && return r.vR
    elseif seam == "AC"
        region == "A" && return r.hL
        region == "C" && return hasproperty(r, :cL) ? r.cL : r.hL
    elseif seam == "BC"
        region == "B" && return r.hR
        region == "C" && return hasproperty(r, :cR) ? r.cR : r.hR
    end
    error("region $region is not incident on seam $seam")
end

# -----------------------------------------------------------------------------
# Replica wiring with explicit ket and bra legs
# -----------------------------------------------------------------------------

function m2_split_boundary(M::ITensor, region::String, c::Int, d::Int, D::Int)
    oldL = theind(M, "M2Rail", "L")
    oldP = theind(M, "M2Rail", "P")
    oldR = theind(M, "M2Rail", "R")
    chi, dp = ITensors.dim(oldL), ITensors.dim(oldP)
    ITensors.dim(oldR) == chi || error("rail has unequal retained bonds")
    dp == D^2 || error("rail physical leg $dp is not D^2=$(D^2)")
    ray_hits = [ray for ray in ("hL", "hR", "vL", "vR")
                if hastags(oldP, ray)]
    length(ray_hits) == 1 ||
        error("rail physical leg must carry exactly one Ray orientation tag; " *
              "found $(ray_hits) in $(tags(oldP))")
    ray = only(ray_hits)
    il = Index(chi, "Chi,$region,$(pairtag(c,d)),L")
    ir = Index(chi, "Chi,$region,$(pairtag(c,d)),R")
    ik = Index(D, "Virt,$(ket(c)),Ray,$ray,s1_1")
    ib = Index(D, "Virt,$(bra(d)),Ray,$ray,s1_1")
    T = promote_type(eltype(M), Float64)
    X = ITensor(zeros(T, chi, D, D, chi), il, ik, ib, ir)
    for l in 1:chi, k in 1:D, b in 1:D, r in 1:chi
        X[il => l, ik => k, ib => b, ir => r] =
            M[oldL => l, oldP => k + D * (b - 1), oldR => r]
    end
    X
end

function m2_boundary_map(region::String, rail::ITensor,
                         sigma::Vector{Int}, D::Int)
    Dict((region, c) => m2_split_boundary(rail, region, c, sigma[c], D)
         for c in eachindex(sigma))
end

function m2_relative_permutation(pX::Vector{Int}, pY::Vector{Int})
    length(pX) == length(pY) ||
        throw(DimensionMismatch("sector permutations have different lengths"))
    k = length(pX)
    sort(pX) == collect(1:k) || throw(ArgumentError("invalid permutation pX=$pX"))
    sort(pY) == collect(1:k) || throw(ArgumentError("invalid permutation pY=$pY"))
    # Same convention as core/seam.jl::relative_permutation: sigma_X^(-1)sigma_Y.
    invperm(pX)[pY]
end

function m2_cycles(pX::Vector{Int}, pY::Vector{Int})
    q = m2_relative_permutation(pX, pY)
    seen = falses(length(q)); cycles = Vector{Vector{Int}}()
    for r in eachindex(q)
        seen[r] && continue
        cyc = Int[]; s = r
        while !seen[s]
            push!(cyc, s); seen[s] = true; s = q[s]
        end
        push!(cycles, sort(cyc))
    end
    cycles
end

function m2_regions(k, pA, pB, pC)
    all(length(p) == k for p in (pA, pB, pC)) ||
        throw(DimensionMismatch("sector permutations must have k entries"))
    (A=M2Region("A", collect(pA)), B=M2Region("B", collect(pB)),
     C=M2Region("C", collect(pC)))
end

function m2_seam_wiring(bt::Dict, X::M2Region, Y::M2Region,
                        copies::Vector{Int})
    objs = Dict{Tuple{String,Int},ITensor}()
    for c in copies
        objs[("X", c)] = copy(bt[(X.name, c)])
        objs[("Y", c)] = copy(bt[(Y.name, c)])
    end
    carrier(side::M2Region, key::String, layer::String) = begin
        hits = [c for c in copies
                if ket(c) == layer || bra(side.sigma[c]) == layer]
        length(hits) == 1 ||
            error("layer $layer has $(length(hits)) carriers in $(side.name)")
        (key, only(hits))
    end
    for c in copies, layer in (ket(c), bra(X.sigma[c]))
        kx = carrier(X, "X", layer); ky = carrier(Y, "Y", layer)
        ix = theind(objs[kx], "Virt", layer)
        iy = theind(objs[ky], "Virt", layer)
        objs[ky] = replaceind(objs[ky], iy, ix)
    end
    reduce(vcat, ([objs[("X", c)], objs[("Y", c)]] for c in copies))
end

function m2_leg_key(ix)
    tags0 = strip.(split(replace(string(tags(ix)), "\"" => ""), ","))
    join(filter(t -> t != "Chi" && t != "L" && t != "R", tags0), ",")
end

function m2_left_right(Ts::Vector{ITensor})
    lefts = [theind(T, "Chi", "L") for T in Ts]
    rights = [theind(T, "Chi", "R") for T in Ts]
    lmap = Dict(m2_leg_key(i) => i for i in lefts)
    rmap = Dict(m2_leg_key(i) => i for i in rights)
    length(lmap) == length(lefts) || error("duplicate left seam keys")
    length(rmap) == length(rights) || error("duplicate right seam keys")
    Set(keys(lmap)) == Set(keys(rmap)) || error("left/right seam keys differ")
    lefts, lmap, rmap
end

function m2_seam_apply(Ts::Vector{ITensor}, R::ITensor)
    peak = 0
    for T in Ts
        R *= T
        peak = max(peak, prod(ITensors.dim.(inds(R))))
    end
    R, peak
end

function m2_apply_array(Ts, lefts, lmap, rmap, x)
    R = ITensor(x, lefts...)
    R, peak = m2_seam_apply(Ts, R)
    for (key, ir) in rmap
        R = replaceind(R, ir, lmap[key])
    end
    Array(R, lefts...), peak
end

"Apply the Hermitian-adjoint seam map in exactly the forward map's basis."
function m2_apply_adjoint_array(Ts, lefts, lmap, rmap, x)
    # The public array basis is the ordered list `lefts`.  The forward map
    # returns its right legs to that basis by logical-leg key, so the adjoint
    # must first place x on the corresponding right legs in the same order.
    rights_in_left_order = [rmap[m2_leg_key(il)] for il in lefts]
    R = ITensor(x, rights_in_left_order...)
    peak = 0
    for T in reverse(Ts)
        R *= dag(T)
        peak = max(peak, prod(ITensors.dim.(inds(R))))
    end
    Set(inds(R)) == Set(lefts) ||
        error("adjoint seam map did not return exactly the input legs")
    Array(R, lefts...), peak
end

"Numerically verify <y,A*x> = <A^dag*y,x> for a named seam map."
function m2_adjoint_residual(Ts; seed::Int=Int(0x6a756e63))
    lefts, lmap, rmap = m2_left_right(Ts)
    shape = Tuple(ITensors.dim.(lefts))
    rng = Xoshiro(seed)
    T = foldl(promote_type, (eltype(tensor) for tensor in Ts); init=Float64)
    x = randn(rng, T, shape)
    y = randn(rng, T, shape)
    Ax, _ = m2_apply_array(Ts, lefts, lmap, rmap, x)
    Aty, _ = m2_apply_adjoint_array(Ts, lefts, lmap, rmap, y)
    lhs = dot(vec(y), vec(Ax))
    rhs = dot(vec(Aty), vec(x))
    abs(lhs-rhs) / max(abs(lhs), abs(rhs), eps(Float64))
end

"""
    m2_corner_boundary(lefts)

Close the two spatial sides of a seam at infinity with one retained-bond
identity metric per replica copy.  This is the LMPS counterpart of closing a
CTMRG defect edge with the same corner environment in the bent and straight
windows.  It is *not* an all-one vector: for a two-cycle its nonzero entries
are `delta(X1,Y1) * delta(X2,Y2)`.

Within each region the four orientations have already been transported to one
regional retained-bond frame. In the shared-rail route the delta is also the
natural common-frame far cap. In the regional route it is only a nonzero
coordinate seed for the mixed K_XY eigensolve; it must not be interpreted as
an identification of the X and Y gauges. Region and copy ownership is read
from ITensor tags; array-axis position is never used to decide a pairing.
"""
function m2_corner_boundary(lefts)
    bycopy = Dict{Int,Vector{Index}}()
    for ix in lefts
        _, copy = m2_edge_key(ix)
        push!(get!(bycopy, copy, Index[]), ix)
    end
    all(length(xs) == 2 for xs in values(bycopy)) ||
        error("corner boundary needs exactly two spatial legs per replica copy")
    F = ITensor(1.0)
    for copy in sort!(collect(keys(bycopy)))
        i, j = bycopy[copy]
        ITensors.dim(i) == ITensors.dim(j) ||
            error("corner boundary copy $copy has unequal retained dimensions")
        F *= delta(i, j)
    end
    Set(inds(F)) == Set(lefts) ||
        error("corner boundary did not expose exactly the seam input legs")
    Array(F, lefts...)
end

function m2_endpoint(Ts::Vector{ITensor}; nmax::Int=1200,
                     tol::Real=1e-10, krylovdim::Int=40,
                     far_boundary=nothing,
                     far_boundary_tolerance::Real=1e-7,
                     far_boundary_tail_tolerance::Real=1e-9)
    isfinite(far_boundary_tolerance) && far_boundary_tolerance > 0 ||
        throw(ArgumentError("far_boundary_tolerance must be finite and positive"))
    isfinite(far_boundary_tail_tolerance) && far_boundary_tail_tolerance > 0 ||
        throw(ArgumentError("far_boundary_tail_tolerance must be finite and positive"))
    lefts, lmap, rmap = m2_left_right(Ts)
    shape = Tuple(ITensors.dim.(lefts))
    boundary0 = far_boundary === nothing ? m2_corner_boundary(lefts) : far_boundary
    size(boundary0) == shape ||
        throw(DimensionMismatch("far boundary has size $(size(boundary0)), expected $shape"))
    T = foldl(promote_type,
              (eltype(tensor) for tensor in Ts);
              init=promote_type(Float64, eltype(boundary0)))
    boundary = T.(boundary0)
    norm(boundary) > 0 || throw(ArgumentError("far boundary must be nonzero"))
    x0 = boundary / norm(boundary)
    peak = Ref(0)
    transfer = x -> begin
        y, p = m2_apply_array(Ts, lefts, lmap, rmap, x)
        peak[] = max(peak[], p); y
    end
    adjoint_transfer = x -> begin
        y, p = m2_apply_adjoint_array(Ts, lefts, lmap, rmap, x)
        peak[] = max(peak[], p); y
    end
    adjoint_gate = m2_adjoint_residual(Ts)
    adjoint_gate <= 1e-11 ||
        error("named seam adjoint residual $adjoint_gate exceeds gate")
    alg = KrylovKit.Arnoldi(; tol=Float64(tol), maxiter=nmax,
                            krylovdim=min(prod(shape), krylovdim), verbosity=0)
    # A chi=1 product boundary has a one-dimensional seam space. Requesting a
    # second diagnostic eigenvalue is invalid there; the leading fixed point
    # and its normalization are still well defined.
    nev = min(2, prod(shape))
    vals, vecs, info = KrylovKit.eigsolve(transfer, x0, nev, :LM, alg)
    info.converged >= 1 || error("endpoint Arnoldi did not converge")
    lambda = vals[1]
    R = vecs[1] / norm(vecs[1])
    j = argmax(abs.(R)); abs(R[j]) > 0 && (R /= R[j] / abs(R[j]))
    y = transfer(R)
    residual = norm(y - lambda * R) /
               max(norm(y), abs(lambda) * norm(R), eps(Float64))
    residual <= max(10tol, sqrt(eps(Float64))) ||
        error("endpoint residual $residual exceeds gate")
    gap = length(vals) > 1 ? (abs(vals[1]) - abs(vals[2])) / abs(vals[1]) : NaN

    # A unit-Frobenius right eigenvector has no physical normalization relative
    # to a different replica sector. Recover the thermodynamic intercept from
    # the *same finite far boundary* used to define every sector:
    #
    #   A^n C = exp(s_n) x_n,    ||x_n|| = 1,
    #   s_n -> n log|lambda| + log|c|,    x_n -> exp(i phi_n) R.
    #
    # This is algebraically the same coefficient
    #
    #   c = <L,C>/<L,R>,
    #
    # but it avoids an independent adjoint Arnoldi solve.  For a strongly
    # non-normal seam map, separately converged right and left Ritz pairs can
    # have tiny residuals yet poorly matched eigenvalues (pseudospectral
    # sensitivity). Directly propagating and normalising C selects the required
    # sector. A tail fit of s_n gives the intercept without repeatedly dividing
    # by an approximate Ritz value. Only |c| enters all stored entropies and the
    # common-boundary construction; its arbitrary scalar phase is fixed to +1.
    initial_norm = norm(boundary)
    x = boundary / initial_norm
    lognorm = log(initial_norm)
    ns = Float64[0.0]
    lognorms = Float64[lognorm]
    far_overlap = zero(T)
    prefactor_alignment = Inf
    prefactor_drift = Inf
    prefactor_iters = 0
    prefactor_slope_match = Inf
    prefactor_gate = max(100Float64(tol), Float64(far_boundary_tail_tolerance))
    alignment_gate = max(1000Float64(tol), Float64(far_boundary_tolerance))
    fit_window = min(24, nmax)
    fit_window >= 4 || throw(ArgumentError("nmax must be at least 4"))
    for n in 1:nmax
        yprop = transfer(x)
        scale = norm(yprop)
        isfinite(scale) && scale > 0 ||
            error("far-boundary propagation became nonfinite or zero")
        x = yprop / scale
        lognorm += log(scale)
        push!(ns, n)
        push!(lognorms, lognorm)

        coefficient = dot(vec(R), vec(x))
        abs(coefficient) > 100eps(Float64) ||
            error("corner far boundary has zero overlap with the dominant seam sector")
        remainder = x - coefficient * R
        prefactor_alignment = norm(remainder) /
                              max(norm(x), abs(coefficient), eps(Float64))
        prefactor_iters = n

        if length(ns) >= fit_window
            xfit = @view ns[(end-fit_window+1):end]
            yfit = @view lognorms[(end-fit_window+1):end]
            xbar = sum(xfit) / fit_window
            ybar = sum(yfit) / fit_window
            denominator = sum((u-xbar)^2 for u in xfit)
            slope = sum((xfit[j]-xbar) * (yfit[j]-ybar)
                        for j in eachindex(xfit)) / denominator
            intercept = ybar - slope*xbar
            prefactor_drift = maximum(abs(yfit[j] -
                (intercept + slope*xfit[j])) for j in eachindex(xfit))
            prefactor_slope_match = abs(slope - log(abs(lambda)))
            far_overlap = T(exp(intercept))
            prefactor_alignment <= alignment_gate &&
                prefactor_drift <= prefactor_gate && break
        end
    end
    prefactor_alignment <= alignment_gate ||
        error("far-boundary alignment residual $prefactor_alignment exceeds gate $alignment_gate")
    prefactor_drift <= prefactor_gate ||
        error("far-boundary tail-fit residual $prefactor_drift exceeds gate $prefactor_gate")
    abs(far_overlap) > 100eps(Float64) ||
        error("far-boundary tail fit produced a zero prefactor")
    log_prefactor = log(abs(far_overlap))

    # These legacy fields described the removed independent adjoint eigensolve.
    # Keep them explicitly unavailable rather than reporting a misleading
    # left/right match as a production diagnostic.
    left_residual = NaN
    left_gap = NaN
    left_iters = 0
    left_match = 0
    eigenvalue_match = NaN
    (R=R, lefts, lambda, residual, gap, iters=Int(info.numiter), peak=peak[],
     far_prefactor=far_overlap, log_prefactor, left_residual, left_gap,
     left_iters, left_match, adjoint_gate, eigenvalue_match,
     prefactor_alignment, prefactor_drift, prefactor_iters,
     prefactor_slope_match)
end

# -----------------------------------------------------------------------------
# Six shared rank-four R, three rank-two metrics, and five replica sectors
# -----------------------------------------------------------------------------

# One descriptor table is shared by the octahedron, the three direct Z2
# closures, and the endpoint-audit CSV.  Keeping an alias here preserves the
# historical method-2 names without duplicating the replica topology.
const M2_DESCRIPTORS = SIXR_DESCRIPTORS

function m2_specs(rails, D, k, pA, pB, pC)
    regs = m2_regions(k, pA, pB, pC); specs = NamedTuple[]
    for (Xn, Yn) in (("A", "B"), ("A", "C"), ("B", "C"))
        seam = Xn * Yn
        X = getproperty(regs, Symbol(Xn)); Y = getproperty(regs, Symbol(Yn))
        for cyc in m2_cycles(X.sigma, Y.sigma)
            bt = Dict{Tuple{String,Int},ITensor}()
            for reg in (X, Y)
                merge!(bt, m2_boundary_map(reg.name,
                                            m2_rail(seam, reg.name, rails),
                                            reg.sigma, D))
            end
            Ts = m2_seam_wiring(bt, X, Y, cyc)
            lefts, _, _ = m2_left_right(Ts)
            push!(specs, (; seam, Xn, Yn, cyc, Ts, lefts))
        end
    end
    specs
end

function m2_solve_specs(specs; nmax=1200, tol=1e-10,
                        far_boundaries=nothing,
                        far_boundary_tolerance=1e-7,
                        far_boundary_tail_tolerance=1e-9)
    far_boundaries === nothing || length(far_boundaries) == length(specs) ||
        throw(DimensionMismatch("one far boundary is required per seam spec"))
    Rs = ITensor[]; vertices = Tuple{String,String,Vector{Int}}[]
    infos = NamedTuple[]
    for (j, s) in enumerate(specs)
        far = far_boundaries === nothing ? nothing : far_boundaries[j]
        e = m2_endpoint(s.Ts; nmax, tol, far_boundary=far,
                        far_boundary_tolerance, far_boundary_tail_tolerance)
        push!(Rs, ITensor(e.R, s.lefts...))
        push!(vertices, (s.Xn, s.Yn, s.cyc))
        push!(infos, (; residual=e.residual, gap=e.gap, iters=e.iters,
                       peak=e.peak, lambda=e.lambda,
                       far_prefactor=e.far_prefactor,
                       log_prefactor=e.log_prefactor,
                       left_residual=e.left_residual,
                       left_gap=e.left_gap, left_iters=e.left_iters,
                       adjoint_gate=e.adjoint_gate,
                       eigenvalue_match=e.eigenvalue_match,
                       prefactor_alignment=e.prefactor_alignment,
                       prefactor_drift=e.prefactor_drift,
                       prefactor_iters=e.prefactor_iters,
                       prefactor_slope_match=e.prefactor_slope_match))
    end
    Rs, vertices, infos
end

function m2_edge_key(ix)
    ts = strip.(split(replace(string(tags(ix)), "\"" => ""), ","))
    rr = findfirst(t -> t in ("A", "B", "C"), ts)
    pp = findfirst(t -> occursin(r"^\d+_\d+b$", t), ts)
    (rr === nothing || pp === nothing) && error("cannot parse edge tags $ts")
    (ts[rr], parse(Int, split(ts[pp], "_")[1]))
end

"""
    m2_metric_product_boundary(source, target)

Use the converged one-copy untwisted seam endpoint as the *common finite
boundary* of a nontrivial two-cycle.  If `target.cyc == (r,s)`, this returns
two relabelled copies of the same physical endpoint, including the two saved
finite-length prefactors.  This implements the synchronized normalization

    R_2^(0) = R_1^(r) tensor R_1^(s)

rather than comparing independently unit-normalized twisted and untwisted
eigenvectors.
"""
function m2_metric_product_boundary(source, target)
    length(source.spec.cyc) == 1 || error("source must be a one-copy metric")
    length(target.cyc) == 2 || error("target must be a two-cycle")
    source.spec.seam == target.seam || error("metric and target seams differ")
    srclegs = Dict(m2_edge_key(ix) => ix for ix in inds(source.R))
    dstlegs = Dict(m2_edge_key(ix) => ix for ix in target.lefts)
    regions = (target.Xn, target.Yn)
    F = ITensor(1.0)
    for replica in target.cyc
        Rc = copy(source.R)
        for region in regions
            old = srclegs[(region, only(source.spec.cyc))]
            new = dstlegs[(region, replica)]
            Rc = replaceind(Rc, old, new)
        end
        F *= Rc
    end
    # Preserve phase as well as magnitude for generic complex PEPS.
    F *= source.info.far_prefactor^length(target.cyc)
    Set(inds(F)) == Set(target.lefts) ||
        error("metric-product boundary did not produce the target seam legs")
    Array(F, target.lefts...)
end

function m2_shared_tensors(Rs, vertices;
                           hinges=Dict{String,Matrix{Float64}}())
    registry = Dict{Tuple{String,Int},Vector{Int}}()
    legs = [Dict{Tuple{String,Int},Index}() for _ in Rs]
    for (j, (X, Y, cyc)) in enumerate(vertices), reg in (X, Y), c in cyc
        push!(get!(registry, (reg,c), Int[]), j)
    end
    all(length(js) == 2 for js in values(registry)) ||
        error("closure registry is not pairwise")
    for (j, R) in enumerate(Rs), ix in inds(R)
        hastags(ix, "Chi") || continue
        legs[j][m2_edge_key(ix)] = ix
    end
    out = copy(Rs)
    for (edge, js) in registry
        j1, j2 = js
        region, _ = edge
        if haskey(hinges, region)
            # A/B are bent L-shaped objects.  The horizontal and vertical
            # half-infinite tails meet once at the junction; G^{-1} is a
            # single hinge there, not a metric repeated on every R bond.
            seam(j) = vertices[j][1] * vertices[j][2]
            vertical = filter(j -> seam(j) == "AB", js)
            length(vertical) == 1 ||
                error("region $region hinge has $(length(vertical)) AB endpoints")
            jv = only(vertical)
            jh = only(filter(!=(jv), js))
            ih, iv = legs[jh][edge], legs[jv][edge]
            Ginv = hinges[region]
            size(Ginv) == (dim(ih), dim(iv)) ||
                throw(DimensionMismatch("$region hinge shape mismatch"))
            out[jh] *= ITensor(Ginv, ih, iv)
        else
            # The straight C object has no separate junction projector.
            out[j2] = replaceind(out[j2], legs[j2][edge], legs[j1][edge])
        end
    end
    out
end

"""
    m2_topology_audit(Rs, vertices)

Audit the *actual named legs* of the six four-replica endpoints before any
scalar contraction.  This is stronger than checking that the final product is
a scalar: a wrong wiring can still close accidentally.

The required graph is K_{2,2,2}:

  * six rank-four vertices;
  * twelve logical edges `(region,copy)`, A1..A4, B1..B4, C1..C4;
  * each logical edge occurs on exactly two vertices;
  * vertices from the same seam are antipodal and share no edge;
  * vertices from different seams share exactly one edge.

`m2_shared_tensors` then replaces the second occurrence of every logical edge
by the first occurrence's ITensor `Index`.  The final occurrence check proves
that all twelve concrete Index objects occur exactly twice.
"""
function m2_topology_audit(Rs, vertices)
    length(Rs) == length(vertices) == 6 ||
        error("method2 topology expected six endpoint vertices")
    all(order(R) == 4 for R in Rs) ||
        error("method2 topology expected every endpoint to have four legs")

    expected = Set((region, copy) for region in ("A", "B", "C")
                                      for copy in 1:4)
    vertex_edges = Vector{Set{Tuple{String,Int}}}()
    logical_count = Dict{Tuple{String,Int},Int}()
    for (j, R) in enumerate(Rs)
        edges = Set(m2_edge_key(ix) for ix in inds(R) if hastags(ix, "Chi"))
        length(edges) == 4 ||
            error("method2 vertex $j does not carry four distinct named edges")
        push!(vertex_edges, edges)
        for edge in edges
            logical_count[edge] = get(logical_count, edge, 0) + 1
        end
    end
    Set(keys(logical_count)) == expected ||
        error("method2 topology edge labels differ from A1..C4")
    all(==(2), values(logical_count)) ||
        error("method2 topology logical edges are not pairwise")

    for i in 1:5, j in i+1:6
        shared = length(intersect(vertex_edges[i], vertex_edges[j]))
        same_seam = vertices[i][1:2] == vertices[j][1:2]
        want = same_seam ? 0 : 1
        shared == want ||
            error("method2 vertices $i,$j share $shared edges; expected $want")
    end

    concrete = m2_shared_tensors(Rs, vertices)
    occurrence = Dict{Index,Int}()
    for R in concrete, ix in inds(R)
        hastags(ix, "Chi") || continue
        occurrence[ix] = get(occurrence, ix, 0) + 1
    end
    length(occurrence) == 12 ||
        error("method2 concrete closure has $(length(occurrence)) edges, expected 12")
    all(==(2), values(occurrence)) ||
        error("method2 concrete ITensor indices are not pairwise")
    (vertices=6, edges=12, degree=4, concrete_pairing=true)
end

function m2_vertex_index(vertices, name::Symbol)
    X, Y, cyc0 = getproperty(M2_DESCRIPTORS, name)
    hits = findall(v -> v == (X, Y, collect(cyc0)), vertices)
    length(hits) == 1 || error("descriptor $name has $(length(hits)) matches")
    only(hits)
end

function m2_close_octahedron(Rs, vertices;
                             hinges=Dict{String,Matrix{Float64}}())
    out = m2_shared_tensors(Rs, vertices; hinges)
    names1 = (:AB12, :AC13, :BC14); names2 = (:AB34, :AC24, :BC23)
    left = out[m2_vertex_index(vertices, names1[1])] *
           out[m2_vertex_index(vertices, names1[2])]
    peak = prod(ITensors.dim.(inds(left)))
    left *= out[m2_vertex_index(vertices, names1[3])]
    peak = max(peak, prod(ITensors.dim.(inds(left))))
    right = out[m2_vertex_index(vertices, names2[1])] *
            out[m2_vertex_index(vertices, names2[2])]
    peak = max(peak, prod(ITensors.dim.(inds(right))))
    right *= out[m2_vertex_index(vertices, names2[3])]
    peak = max(peak, prod(ITensors.dim.(inds(right))))
    z = left * right
    order(z) == 0 || error("octahedron left $(order(z)) open legs")
    ITensors.scalar(z), peak
end

"Greedy named-index contraction for the one- and two-replica closures."
function m2_close_small(Rs, vertices;
                        hinges=Dict{String,Matrix{Float64}}())
    work = m2_shared_tensors(Rs, vertices; hinges); peak = 0
    while length(work) > 1
        candidates = Tuple{Int,Int,Int}[]
        for i in 1:length(work)-1, j in i+1:length(work)
            isempty(commoninds(work[i], work[j])) && continue
            shared = Set(commoninds(work[i], work[j]))
            # If the pair shares every remaining leg, its product is a scalar;
            # the empty product is therefore one, not an error.
            sz = prod((ITensors.dim(ix) for ix in
                       union(Set(inds(work[i])), Set(inds(work[j])))
                       if !(ix in shared)); init=1)
            push!(candidates, (sz, i, j))
        end
        isempty(candidates) && error("small closure became disconnected")
        _, i, j = minimum(candidates)
        T = work[i] * work[j]; peak = max(peak, prod(ITensors.dim.(inds(T))))
        deleteat!(work, j); deleteat!(work, i); push!(work, T)
    end
    order(only(work)) == 0 || error("small closure has open legs")
    ITensors.scalar(only(work)), peak
end

function m2_mapped_residual(Ts, lefts, data)
    _, lmap, rmap = m2_left_right(Ts)
    y, _ = m2_apply_array(Ts, lefts, lmap, rmap, data)
    x = vec(data); yv = vec(y); lambda = dot(x, yv) / dot(x, x)
    norm(yv - lambda * x) / max(norm(yv), abs(lambda) * norm(x), eps())
end

"Map one solved endpoint into a target replica sector by named logical legs."
function m2_relabel_endpoint(source::ITensor, source_spec, target_spec)
    source_spec.seam == target_spec.seam ||
        error("cannot reuse $(source_spec.seam) endpoint on $(target_spec.seam)")
    length(source_spec.cyc) == length(target_spec.cyc) ||
        error("source and target endpoint cycles have different lengths")
    source_legs = Dict(m2_edge_key(ix) => ix for ix in inds(source))
    target_legs = Dict(m2_edge_key(ix) => ix for ix in target_spec.lefts)
    length(source_legs) == order(source) || error("source endpoint has unnamed legs")
    length(target_legs) == length(target_spec.lefts) ||
        error("target endpoint has duplicate named legs")

    source_key(target_key) = begin
        region, target_copy = target_key
        j = findfirst(==(target_copy), target_spec.cyc)
        j === nothing && error("target copy $target_copy is outside its cycle")
        region in (target_spec.Xn, target_spec.Yn) ||
            error("bad $(target_spec.seam) endpoint region $region")
        (region, source_spec.cyc[j])
    end

    used = Set{Tuple{String,Int}}()
    out = source
    for (target_key, target_ix) in target_legs
        skey = source_key(target_key)
        skey in used && error("endpoint map reuses source leg $skey")
        haskey(source_legs, skey) || error("endpoint lacks source leg $skey")
        push!(used, skey)
        out = replaceind(out, source_legs[skey], target_ix)
    end
    length(used) == length(source_legs) ||
        error("endpoint map leaves source legs unused")
    Set(inds(out)) == Set(target_spec.lefts) ||
        error("named endpoint relabeling did not produce the target legs")
    out
end

"""
    m2_build_O2(name, specs, two_sources, metric_sources; ...)

Assemble one ordinary two-replica sector from the shared endpoint library.
Two seams contain a nontrivial 2-cycle and therefore receive one of the six
rank-four `R` arrays.  The remaining seam consists of identity 1-cycles and is
closed by two occurrences of its rank-two metric `S`.  No new rank-four fixed
point is solved here.

The source and target legs are matched by `(seam,region,cycle-position)`, not by
the storage order of an Array. `m2_mapped_residual` verifies the relabelled
tensor against the target transfer operator, so a mapping mistake cannot
silently produce a scalar.
"""
function m2_build_O2(name, specs, two_sources, metric_sources;
                     two_scale=Dict{Symbol,Float64}(),
                     metric_scale=Dict{String,Float64}(),
                     hinges=Dict{String,Matrix{Float64}}())
    Rs = ITensor[]
    vertices = Tuple{String,String,Vector{Int}}[]
    residuals = Float64[]
    for s in specs
        if length(s.cyc) == 2
            key = SIXR_O2_ASSIGNMENT[(name, s.seam)]
            source = two_sources[key]
            R = m2_relabel_endpoint(source.R, source.spec, s) *
                get(two_scale, key, 1.0)
        elseif length(s.cyc) == 1
            source = metric_sources[s.seam]
            R = m2_relabel_endpoint(source.R, source.spec, s) *
                get(metric_scale, s.seam, 1.0)
        else
            error("method2 supports only one- and two-cycle seam blocks")
        end
        data = Array(R, s.lefts...)
        push!(residuals, m2_mapped_residual(s.Ts, s.lefts, data))
        push!(Rs, R)
        push!(vertices, (s.Xn, s.Yn, s.cyc))
    end
    z, peak = m2_close_small(Rs, vertices; hinges)
    ComplexF64(z), maximum(residuals), peak
end

"Sum the far-boundary intercept and line eigenvalue of one reused O2 closure."
function m2_O2_asymptotic_data(name, specs, two_sources, metric_sources)
    intercept = 0.0
    line = 0.0
    for s in specs
        source = if length(s.cyc) == 2
            two_sources[SIXR_O2_ASSIGNMENT[(name, s.seam)]]
        elseif length(s.cyc) == 1
            metric_sources[s.seam]
        else
            error("method2 supports only one- and two-cycle seam blocks")
        end
        intercept += source.info.log_prefactor
        line += log(abs(source.info.lambda))
    end
    (; intercept, line)
end

"""
    m2_contract_shared_six_R(rails, D; nmax, tol, verbose)

Perform only the replica part of method 2 after the oriented one-copy rails
have already been constructed. `rails` may be either one shared four-rail
object or a regional `(A=...,B=...,C=...)` tuple. This separation is important
for the audit: different candidates for the spatial L-shaped environment can
be tested without changing the replica topology.

Exactly six rank-four fixed points are solved:

    AB12, AB34, AC13, AC24, BC14, BC23.

They form the six vertices of the Z4 octahedron.  Three additional rank-two
untwisted seam metrics are solved once.  Each Z2 sector reuses two of the six R
and the appropriate metric for its identity cycles; Z1 is the metric triangle.
Thus only six rank-four R exist, but a Z2 is not a bare two-R trace.
"""
function m2_contract_shared_six_R(rails, D::Int;
                                  nmax::Int=1200, tol::Real=1e-10,
                                  scale_tolerance::Real=1e-10,
                                  far_boundary_tolerance::Real=1e-7,
                                  far_boundary_tail_tolerance::Real=1e-9,
                                  verbose::Bool=true,
                                  hinges=Dict{String,Matrix{Float64}}())
    isfinite(scale_tolerance) && scale_tolerance > 0 ||
        throw(ArgumentError("scale_tolerance must be finite and positive"))
    isfinite(far_boundary_tolerance) && far_boundary_tolerance > 0 ||
        throw(ArgumentError("far_boundary_tolerance must be finite and positive"))
    isfinite(far_boundary_tail_tolerance) && far_boundary_tail_tolerance > 0 ||
        throw(ArgumentError("far_boundary_tail_tolerance must be finite and positive"))
    specs4 = m2_specs(rails, D, 4, M2_SIGMA_A, M2_SIGMA_B, M2_SIGMA_C)
    specs1 = m2_specs(rails, D, 1, [1], [1], [1])
    local Rs4, vertices4, infos4, Rs1, vertices1, infos1, metric_sources
    seamsec = @elapsed begin
        Rs1, vertices1, infos1 = m2_solve_specs(
            specs1; nmax, tol, far_boundary_tolerance,
            far_boundary_tail_tolerance)
        metric_sources = Dict(specs1[j].seam =>
                              (; R=Rs1[j], spec=specs1[j], info=infos1[j])
                              for j in eachindex(specs1))
        far4 = [m2_metric_product_boundary(metric_sources[s.seam], s)
                for s in specs4]
        Rs4, vertices4, infos4 =
            m2_solve_specs(specs4; nmax, tol, far_boundaries=far4,
                           far_boundary_tolerance,
                           far_boundary_tail_tolerance)
    end
    length(Rs4) == 6 || error("expected exactly six rank-four endpoints")
    length(Rs1) == 3 || error("expected exactly three rank-two seam metrics")
    topology = m2_topology_audit(Rs4, vertices4)
    sixr_library_audit(Rs4, vertices4)
    endpoint_residual = maximum(x.residual for x in infos4)
    metric_residual = maximum(x.residual for x in infos1)

    two_sources = Dict(name => begin
        j = m2_vertex_index(vertices4, name)
        (; R=Rs4[j], spec=specs4[j], info=infos4[j])
    end for name in propertynames(M2_DESCRIPTORS))

    local gamma4, gamma1, O2, O2_asymptotic, S2_raw, S3_raw, tildeS
    local kappaA_A, kappaA_B, kappaY, S2, S3
    local map_residual, peak, intercept_gate, local_identity_gate
    closesec = @elapsed begin
        gamma4, peak = m2_close_octahedron(Rs4, vertices4; hinges)
        gamma1, peak1 = m2_close_small(Rs1, vertices1; hinges)
        peak = max(peak, peak1)
        O2 = Dict{String,Float64}()
        O2_asymptotic = Dict{String,NamedTuple}()
        map_residual = 0.0
        for (name, k, pA, pB, pC) in SIXR_O2_SECTORS
            specs = m2_specs(rails, D, k, pA, pB, pC)
            z, residual, peak2 = m2_build_O2(name, specs,
                                             two_sources, metric_sources; hinges)
            O2[name] = log(abs(z))
            O2_asymptotic[name] =
                m2_O2_asymptotic_data(name, specs, two_sources, metric_sources)
            map_residual = max(map_residual, residual)
            peak = max(peak, peak2)
        end
        O1, O4 = log(abs(gamma1)), log(abs(gamma4))
        S2_raw = Dict("A" => -(O2["O2A"] - 2O1),
                      "B" => -(O2["O2B"] - 2O1),
                      "C" => -(O2["O2C"] - 2O1))
        S3_raw = -0.5 * (O4 - 4O1)
        tildeS = -(O4 + 2O1 - O2["O2A"] - O2["O2B"] - O2["O2C"])

        # CTMRG-compatible local-window convention.  `I` is the constant
        # (length-independent) term obtained after stripping each semi-infinite
        # seam's dominant eigenvalue while retaining a synchronized physical
        # far boundary.  The one-copy metrics are solved first; every twisted
        # two-cycle is then initialized by the corresponding R1 tensor product.
        # Region C is the straight half-plane reference; A and B are the two
        # bends.  For the present isotropic construction the three straight
        # k=4 references are rotations of this same C reference.
        I1 = O1 + sum(x.log_prefactor for x in infos1)
        I4 = O4 + sum(x.log_prefactor for x in infos4)
        I2 = Dict(name => O2[name] + O2_asymptotic[name].intercept
                  for name in keys(O2))
        kappaA_A = I2["O2A"] - I2["O2C"]
        kappaA_B = I2["O2B"] - I2["O2C"]
        kappaY = I4 + 2I1 - 3I2["O2C"]
        S2 = Dict("A" => -kappaA_A,
                  "B" => -kappaA_B,
                  "C" => 0.0)
        S3 = -0.5kappaY

        # The synchronized finite-boundary intercepts occur with exactly the
        # same multiplicities as the endpoint tensors themselves, so they must
        # cancel from S_tilde.
        tilde_intercept = -(I4 + 2I1 - I2["O2A"] - I2["O2B"] - I2["O2C"])
        intercept_gate = abs(tilde_intercept - tildeS)
        local_identity_gate = abs((kappaA_A + kappaA_B - kappaY) - tildeS)
    end

    # Rescale all six R and all three metrics independently.  In the final
    # five-sector combination every scale must cancel with its exact
    # multiplicity; this catches an omitted or duplicated endpoint.
    tf = Dict(zip(propertynames(M2_DESCRIPTORS),
                  (1.13, 0.91, 1.07, 0.89, 1.03, 0.97)))
    mf = Dict("AB" => 1.05, "AC" => 0.93, "BC" => 1.09)
    names_by_vertex = [only(name for name in propertynames(M2_DESCRIPTORS)
                            if m2_vertex_index(vertices4, name) == j)
                       for j in eachindex(vertices4)]
    scaled_Rs = [tf[names_by_vertex[j]] * Rs4[j]
                 for j in eachindex(Rs4)]
    gamma4s = m2_close_octahedron(scaled_Rs, vertices4)[1]
    scaled_metrics = [mf[specs1[j].seam] * Rs1[j]
                      for j in eachindex(Rs1)]
    gamma1s = m2_close_small(scaled_metrics, vertices1)[1]
    O2s = Dict{String,Float64}()
    for (name, k, pA, pB, pC) in SIXR_O2_SECTORS
        specs = m2_specs(rails, D, k, pA, pB, pC)
        z, _, _ = m2_build_O2(name, specs, two_sources, metric_sources;
                              two_scale=tf, metric_scale=mf)
        O2s[name] = log(abs(z))
    end
    scaled_tildeS = -(log(abs(gamma4s)) + 2log(abs(gamma1s)) -
                      O2s["O2A"] - O2s["O2B"] - O2s["O2C"])
    scale_gate = abs(scaled_tildeS - tildeS)

    map_residual <= max(100tol, 1e-8) ||
        error("method2 endpoint-axis map residual $map_residual exceeds gate")
    scale_gate <= scale_tolerance ||
        error("method2 endpoint scale-cancellation residual $scale_gate exceeds gate $scale_tolerance")
    intercept_gate <= 1e-10 ||
        error("method2 thermodynamic-intercept cancellation $intercept_gate exceeds gate")
    local_identity_gate <= 1e-10 ||
        error("method2 local-component identity $local_identity_gate exceeds gate")

    verbose && for (j, (v, info)) in enumerate(zip(vertices4, infos4))
        @printf("  R%d %s%s(%s) residual=%.3e gap=%.3e restarts=%d\n",
                j, v[1], v[2], join(v[3]), info.residual, info.gap, info.iters)
    end
    endpoint_audit = [begin
        descriptor = only(name for name in propertynames(M2_DESCRIPTORS)
                          if m2_vertex_index(vertices4, name) == j)
        edges = sort(["$(e[1])$(e[2])"
                      for e in (m2_edge_key(ix) for ix in inds(Rs4[j]))])
        # The octahedron edge only retains `(region,ket copy)`.  Keep the full
        # double-layer rail `(region,ket copy,bra copy)` in the audit as well,
        # so the six handwritten R tensors can be checked without rebuilding
        # their transfer operators from scratch.
        rails = sort(["$(q[1])$(q[2])b$(q[3])"
                      for q in (sixr_rail_key(ix) for ix in inds(Rs4[j]))])
        (; descriptor=String(descriptor), seam=vertices4[j][1]*vertices4[j][2],
           cycle=join(vertices4[j][3]), edges=join(edges, ";"),
           rails=join(rails, ";"),
           residual=infos4[j].residual, gap=infos4[j].gap,
           iters=infos4[j].iters, peak=infos4[j].peak,
           left_residual=infos4[j].left_residual,
           left_gap=infos4[j].left_gap,
           left_iters=infos4[j].left_iters,
           adjoint_gate=infos4[j].adjoint_gate,
           eigenvalue_match=infos4[j].eigenvalue_match,
           prefactor_alignment=infos4[j].prefactor_alignment,
           prefactor_drift=infos4[j].prefactor_drift,
           prefactor_iters=infos4[j].prefactor_iters,
           prefactor_slope_match=infos4[j].prefactor_slope_match,
           log_prefactor=infos4[j].log_prefactor,
           log_lambda=log(abs(infos4[j].lambda)))
    end for j in eachindex(Rs4)]
    mixed_corner_audit = [begin
        spec = specs1[j]
        info = infos1[j]
        (; seam=spec.seam,
           regions=(spec.Xn, spec.Yn),
           dimensions=Tuple(ITensors.dim.(spec.lefts)),
           residual=info.residual, gap=info.gap, iters=info.iters,
           left_residual=info.left_residual, left_gap=info.left_gap,
           left_iters=info.left_iters, adjoint_gate=info.adjoint_gate,
           eigenvalue_match=info.eigenvalue_match,
           prefactor_alignment=info.prefactor_alignment,
           prefactor_drift=info.prefactor_drift,
           prefactor_iters=info.prefactor_iters,
           prefactor_slope_match=info.prefactor_slope_match,
           log_prefactor=info.log_prefactor,
           log_lambda=log(abs(info.lambda)))
    end for j in eachindex(specs1)]
    (; tildeS, S2A=S2["A"], S2B=S2["B"], S2C=S2["C"], S3,
       S2A_raw=S2_raw["A"], S2B_raw=S2_raw["B"],
       S2C_raw=S2_raw["C"], S3_raw,
       kappaA_A, kappaA_B, kappaA=0.5*(kappaA_A+kappaA_B), kappaY,
       gamma4=ComplexF64(gamma4), gamma1=ComplexF64(gamma1),
       mirror=abs(S2["A"] - S2["B"]), map_residual, scale_gate,
       intercept_gate, local_identity_gate,
       endpoint_residual, metric_residual, endpoint_count=6, metric_count=3,
       peak_elements=peak, seamsec, closesec,
       topology, endpoint_audit, mixed_corner_audit)
end

"""
    m2_contract_regional_six_R((A=rails_A, B=rails_B, C=rails_C), D; ...)

Generic three-region six-R closure. The three regional rail objects may come
from independent boundary-MPS solves and therefore need not share a numerical
gauge. For each seam, the ordinary one-copy solve constructs the corresponding
mixed half-infinite map `K_XY`; products of that solved map initialize the two
twisted replica endpoints. The final octahedron connects only legs owned by
the same regional frame, so no bare identity is inserted between unrelated
A/B/C gauges.

The implementation deliberately reuses the audited replica topology in
`m2_contract_shared_six_R`; only rail selection differs.
"""
function m2_contract_regional_six_R(rails::NamedTuple, D::Int; kwargs...)
    all(hasproperty(rails, region) for region in (:A, :B, :C)) ||
        throw(ArgumentError("regional closure requires rails A, B, and C"))
    m2_contract_shared_six_R(rails, D; kwargs...)
end
