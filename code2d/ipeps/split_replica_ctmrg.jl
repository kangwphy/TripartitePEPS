# split_replica_ctmrg.jl -- product-environment defect CTMRG prototype.
# =============================================================================
# The ordinary k=4 calculation fuses the four replicas before CTMRG and keeps
# the same requested chi as k=2.  That destroys the exact replica-product
# environment at criticality.  Here CTMRG is run ONLY for one replica.  Its
# corner, edge and fixed-point projector are replicated without truncation:
#
#   C_k = C_1^(x k),  T_k = T_1^(x k),  Z_k = Z_1^(x k).
#
# By default only the canonical k=2 swap edge is co-evolved with each fresh
# lifted projector Z_2.  Every k=4 double-transposition edge is then built
# exactly as a product of two k=2 edges with the appropriate replica pairing.
# SRMODE=legacy retains the original explicit three-edge k=4 evolution as a
# regression oracle.  Replica labels stay explicit in the final product basis,
# and the existing n=1/2 window geometry is reused unchanged.
#
# Neither the iterative environment nor the final junction materializes a
# chi_1^4 edge.  The latter is contracted as a labeled network of canonical
# k=2 factors using a fixed, offline-optimized path.  Its largest intermediate
# has 64*chi_1^6 entries (for d=2), versus O(chi_1^8) storage in the fused
# window.  SRFULLGATE=1 retains the fused contraction as a small-chi oracle.
#
# Environment:
#   SRBETA  beta (default beta_c)
#   SRCHI1  colon-separated one-replica chi values (default 1:2:3)
#   SRNMAX  maximum seam-edge fixed-point iterations (default 5000)
#   SRTOL   seam-edge max-norm tolerance (default 1e-11)
#   SRMODE  pair (default) or legacy
#   SRFULLGATE 1 recomputes all explicit k=4 straight windows (default 0)
#   SRSTOP  obs (DEFAULT since the 10/10 bitwise regression below) gates the
#           seam fixed point on the drift of the reported gauge-invariant
#           observables s2 and kappa_A, evaluated every SRCHECK iterations and
#           required to pass twice in a row.  Validated against the stored
#           values at (beta,chi1) = (0.30,4), (0.30,16), (0.48,8), (0.70,8):
#           identical to 0.0e+00 while the iteration count at (0.30,16) drops
#           from the full 50000 (57 min) to 125 (15 s).
#           edge restores the historical gate on the
#           ELEMENTWISE |dTd2|, which is gauge dependent, in practice never
#           fires, and therefore burns the full SRNMAX on every point.  It is
#           kept only to reproduce pre-2026-08-17 rows.  |dTd2| is still
#           printed as a diagnostic in both modes, and both keep the
#           gauge-invariant corner-spectrum criterion.
#   SRCHECK observable re-evaluation interval for SRSTOP=obs (default 25)
#   SRFLIP  1 (default) projects the one-replica CTMRG onto global spin-flip
#           covariance with a parity-resolved cut; 0 disables ONLY that
#           projection and leaves everything else identical, giving a
#           controlled twirl-on/off comparison (the ordered phase then picks a
#           symmetry-broken branch, visible as z2split going from ~1e-15 to O(1)).
# =============================================================================
include(joinpath(@__DIR__, "yvx_core.jl"))

const SRBETA = parse(Float64, get(ENV, "SRBETA", string(BETA_C)))
const SRCHIS = parse.(Int, split(get(ENV, "SRCHI1", "1:2:3"), ":"))
const SRNMAX = parse(Int, get(ENV, "SRNMAX", "5000"))
const SRTOL  = parse(Float64, get(ENV, "SRTOL", "1e-11"))
const SRMODE = lowercase(get(ENV, "SRMODE", "pair"))
const SRFULLGATE = parse(Int, get(ENV, "SRFULLGATE", "0"))
const SRSTOP = lowercase(get(ENV, "SRSTOP", "obs"))
SRSTOP in ("edge", "obs") || error("SRSTOP must be edge or obs")
const SRCHECK = parse(Int, get(ENV, "SRCHECK", "25"))
const SRFLIP = parse(Int, get(ENV, "SRFLIP", "1"))
# Memory cap for the junction pre-flight, in GiB.  Default sits just under a
# 2048 GB node (1907 GiB), which is what makes chi1=33 the hard ceiling.
const SRMEMCAP = parse(Float64, get(ENV, "SRMEMCAP", "1850"))
# SRTRACE=1 prints the seam observables at every SRCHECK iteration, starting from
# it=1 where Td2 is still the UNDRESSED product edge T1^(x2).  That first line is
# exactly what "just use T and swap the bond at contraction time" would give: the
# seam gate applied inside the window with no seam ray accumulated behind it.
# The distance from that line to the fixed point is the contribution of the
# semi-infinite seam, i.e. the reason E* has to be evolved at all.
const SRTRACE = parse(Int, get(ENV, "SRTRACE", "0"))
SRFLIP in (0,1) || error("SRFLIP must be 0 or 1")
SRMODE in ("pair", "legacy") || error("SRMODE must be pair or legacy")
const NEED_EXPLICIT4 = SRMODE == "legacy" || SRFULLGATE == 1

# ---------------- tensor products and seam dressing ---------------------------
"Add one slow replica to a rank-4 site tensor; old replica bundle stays fast."
function replica_site(a)
    d = size(a, 1); D = d*d
    out = zeros(D, D, D, D)
    @inbounds for uo in 1:d, lo in 1:d, do_ in 1:d, ro in 1:d,
                  un in 1:d, ln in 1:d, dn in 1:d, rn in 1:d
        out[(un-1)*d+uo, (ln-1)*d+lo, (dn-1)*d+do_, (rn-1)*d+ro] =
            a[uo,lo,do_,ro] * a[un,ln,dn,rn]
    end
    out
end

"Product of two edge tensors, with A fast and B slow on all three indices."
function product_edge(A, B)
    ca, da = size(A, 1), size(A, 3)
    cb, db = size(B, 1), size(B, 3)
    out = zeros(ca*cb, ca*cb, da*db)
    @inbounds for ia in 1:ca, ja in 1:ca, pa in 1:da,
                  ib in 1:cb, jb in 1:cb, pb in 1:db
        out[(ib-1)*ca+ia, (jb-1)*ca+ja, (pb-1)*da+pa] =
            A[ia,ja,pa] * B[ib,jb,pb]
    end
    out
end

"k-fold product edge; replica 1 is the fastest digit."
function product_edge_power(T, k)
    out = copy(T)
    for _ in 2:k
        out = product_edge(out, T)
    end
    out
end

"Build a four-replica edge from two canonical two-replica pair edges.

`pairs=((r1,s1),(r2,s2))` assigns the two factors to the four explicit
replica digits.  No chi_1^4 seam fixed point is evolved or truncated."
function paired_edge4(T2, pairs)
    chi1 = isqrt(size(T2,1)); d1 = isqrt(size(T2,3))
    chi1^2 == size(T2,1) || error("paired_edge4: boundary dimension is not a square")
    d1^2 == size(T2,3) || error("paired_edge4: physical dimension is not a square")
    chi4 = chi1^4; d4 = d1^4
    left = zeros(Int,2,chi4); phys = zeros(Int,2,d4)
    for (factor,(r,s)) in enumerate(pairs), i0 in 0:chi4-1
        left[factor,i0+1] = digit0(i0,chi1,r) +
                            chi1*digit0(i0,chi1,s) + 1
    end
    for (factor,(r,s)) in enumerate(pairs), p0 in 0:d4-1
        phys[factor,p0+1] = digit0(p0,d1,r) + d1*digit0(p0,d1,s) + 1
    end
    out = zeros(chi4,chi4,d4)
    @inbounds for p in 1:d4, j in 1:chi4, i in 1:chi4
        out[i,j,p] = T2[left[1,i],left[1,j],phys[1,p]] *
                     T2[left[2,i],left[2,j],phys[2,p]]
    end
    out
end

# ---------------- factorized four-replica junction ---------------------------
"Dense tensor carrying explicit integer labels and its accumulated log scale."
struct LabeledTensor
    data::Array{Float64}
    labels::Vector{Int}
    logscale::Float64
end

"Split a canonical pair edge into (left_r,right_r,p_r,left_s,right_s,p_s)."
function split_pair_edge(T2)
    chi1 = isqrt(size(T2,1)); d1 = isqrt(size(T2,3))
    chi1^2 == size(T2,1) || error("split_pair_edge: non-square boundary bundle")
    d1^2 == size(T2,3) || error("split_pair_edge: non-square physical bundle")
    raw = reshape(T2,chi1,chi1,chi1,chi1,d1,d1)
    Array(permutedims(raw,(1,3,5,2,4,6)))
end

"Split a canonical pair site into the four legs of r followed by those of s."
function split_pair_site(a2)
    d1 = isqrt(size(a2,1))
    d1^2 == size(a2,1) || error("split_pair_site: non-square replica bundle")
    raw = reshape(a2,d1,d1,d1,d1,d1,d1,d1,d1)
    Array(permutedims(raw,(1,3,5,7,2,4,6,8)))
end

"Contract every common label of two dense factors, normalize, and track log scale."
function contract_labeled(A::LabeledTensor,B::LabeledTensor)
    bpositions = Dict(label=>position for (position,label) in enumerate(B.labels))
    apos_common = [position for (position,label) in enumerate(A.labels)
                   if haskey(bpositions,label)]
    apos_free = [position for position in eachindex(A.labels)
                 if !(position in apos_common)]
    common_labels = A.labels[apos_common]
    bpos_common = [bpositions[label] for label in common_labels]
    bpos_free = [position for position in eachindex(B.labels)
                 if !(position in bpos_common)]

    adims = size(A.data); bdims = size(B.data)
    m = prod(adims[position] for position in apos_free; init=1)
    k = prod(adims[position] for position in apos_common; init=1)
    n = prod(bdims[position] for position in bpos_free; init=1)
    k == prod(bdims[position] for position in bpos_common; init=1) ||
        error("contract_labeled: common-label dimensions disagree")

    AP = reshape(permutedims(A.data,vcat(apos_free,apos_common)),m,k)
    BP = reshape(permutedims(B.data,vcat(bpos_common,bpos_free)),k,n)
    matrix = AP*BP
    output_dims = vcat([adims[position] for position in apos_free],
                       [bdims[position] for position in bpos_free])
    # `matrix` is freshly allocated by AP*BP and is referenced nowhere else, and
    # reshape of a dense Array returns an Array SHARING that memory, so the
    # result is handed on directly.  Wrapping it in Array() made a second full
    # copy of the largest intermediate -- measured, Array(R) on an 8 MB Array
    # allocates another 8 MB, it is a constructor, not a convert.
    #
    # HONEST RECORD: removing that copy was expected to cut the resident set and
    # DID NOT.  At beta=0.40, chi1=16 the peak RSS went 24.80 -> 24.72 GiB, i.e.
    # 0.3%, which is noise.  The measured 3.0x factor between the peak
    # intermediate and the resident set is therefore NOT this copy.  Two further
    # explanations were tested and both failed as well: --heap-size-hint helps
    # erratically at chi1=16 (1.09x to 3.11x across identical runs) and does
    # nothing at all at chi1=24 (274.85 GiB against a 273 GiB baseline), and
    # simply requesting less memory makes the job OOM rather than collect harder.
    # The change is kept because it is bitwise identical (4/4 regression) and one
    # fewer copy of the peak array is strictly better, but it buys no memory.
    #
    # The in-place ./= below writes through `matrix`, which is intended and safe:
    # nothing reads it again.  If reshape ever returned a ReshapedArray the
    # LabeledTensor constructor would raise on the field type rather than
    # silently keep a view alive.
    output = reshape(matrix,Tuple(output_dims))
    scale = maximum(abs,output)
    isfinite(scale) && scale > 0 || error("contract_labeled: zero/nonfinite intermediate")
    output ./= scale
    LabeledTensor(output,vcat(A.labels[apos_free],B.labels[bpos_free]),
                  A.logscale+B.logscale+log(scale))
end

# Cotengra random-greedy path for the exact factor ordering in
# junction_window_pair_log.  optimize_split_junction_path.py regenerates and
# audits it.  At chi1=10: largest intermediate 64,000,000 doubles and an
# estimated 5.52e10 scalar FLOPs.
#
# Re-audited (optimize_split_junction_path2.py, which also SIMULATES a path and
# reproduces the peak above as chi^6 d^6).  3.2e5 alternative orderings were
# tried -- 2.4e5 randomized greedy, 8e4 randomized min-fill, four seeds each --
# and none reached a chi-width below 6.  Min-fill converged from all four seeds
# to chi^6 d^6, i.e. it TIES this path in storage and loses marginally in flops
# (10^13.77 vs 10^13.76 at chi1=24).  Width 6 = 2 legs x 3 seam slots is the
# structural floor (see note_ipeps_en Sec. "two contraction schemes"), and the
# minor-min-width bound gives tw >= 6 on the chi-minor of the label graph.
# CONCLUSION: keep this path.  Do not replace it without re-running that audit.
const JUNCTION_PAIR_PATH = (
    (0,20),(3,50),(40,49),(39,42),(46,47),(3,24),(19,45),(14,44),
    (42,43),(2,33),(7,41),(25,40),(23,34),(37,38),(36,37),(1,21),
    (8,35),(22,34),(24,30),(31,32),(0,13),(14,30),(8,29),(27,28),
    (26,27),(10,12),(10,25),(2,24),(4,23),(16,22),(16,21),(16,20),
    (7,19),(4,8),(16,17),(2,6),(4,15),(13,14),(12,13),(2,4),(0,11),
    (5,10),(7,9),(7,8),(1,5),(0,6),(1,5),(0,1),(0,3),(1,2),(0,1),
)

"Logarithm of the exact four-replica 2x2 Y window without fused k=4 tensors.

Each untwisted object is kept as four one-replica factors.  Each double
transposition is kept as two canonical swap-pair factors, with pairings
AB=(12)(34), CA=(13)(24), and CB=(14)(23).  The labels reproduce the index
order of window_sweep exactly: sites are stored bottom-to-top in each column."
function junction_window_pair_log(C1,T1,Td2,a1,a2Gu,a2Gr)
    factors = LabeledTensor[]
    labels = Dict{Tuple{Symbol,Int},Int}()
    nextlabel = Ref(0)
    label(name::Symbol,replica::Int) = get!(labels,(name,replica)) do
        nextlabel[] += 1
        nextlabel[]
    end
    add(data,names) = push!(factors,LabeledTensor(Array(data),
        [label(name,replica) for (name,replica) in names],0.0))

    # Four corners per replica.  West indices run bottom -> top, while east
    # indices run top -> bottom, matching the matrix products in window_sweep.
    for r in 1:4
        add(C1,[(:w2,r),(:n0,r)])
        add(C1,[(:n2,r),(:e0,r)])
        add(C1,[(:e2,r),(:s2,r)])
        add(C1,[(:s0,r),(:w0,r)])
    end

    sitelegs = Dict(
        :tl => (:utl,:ltl,:vleft,:htop),
        :tr => (:utr,:htop,:vright,:rtr),
        :bl => (:vleft,:lbl,:dbl,:hbot),
        :br => (:vright,:hbot,:dbr,:rbr),
    )
    # Only the top-right site is untwisted in this junction window.
    for r in 1:4
        add(a1,[(name,r) for name in sitelegs[:tr]])
    end

    pairings = Dict(
        :AB => ((1,2),(3,4)),
        :CA => ((1,3),(2,4)),
        :CB => ((1,4),(2,3)),
    )
    pair_edge_data = split_pair_edge(Td2)
    pair_edge_flip_data = split_pair_edge(flipT(Td2))
    function add_edge(left::Symbol,right::Symbol,physical::Symbol;
                      sector=nothing,flipped=false)
        if sector === nothing
            for r in 1:4
                add(T1,[(left,r),(right,r),(physical,r)])
            end
        else
            data = flipped ? pair_edge_flip_data : pair_edge_data
            for (r,s) in pairings[sector]
                add(data,[(left,r),(right,r),(physical,r),
                          (left,s),(right,s),(physical,s)])
            end
        end
    end
    add_edge(:n0,:n1,:utl;sector=:AB)
    add_edge(:n1,:n2,:utr)
    add_edge(:e0,:e1,:rtr)
    add_edge(:e1,:e2,:rbr;sector=:CB,flipped=true)
    add_edge(:s0,:s1,:dbl)
    add_edge(:s1,:s2,:dbr)
    add_edge(:w0,:w1,:lbl;sector=:CA)
    add_edge(:w1,:w2,:ltl)

    pair_u = split_pair_site(a2Gu)
    pair_r = split_pair_site(a2Gr)
    for (site,sector,data) in ((:tl,:AB,pair_r),
                               (:bl,:CA,pair_u),
                               (:br,:CB,pair_u))
        for (r,s) in pairings[sector]
            add(data,vcat([(name,r) for name in sitelegs[site]],
                          [(name,s) for name in sitelegs[site]]))
        end
    end
    length(factors) == 52 || error("junction_window_pair_log: factor ordering changed")

    # Structural guards.  The Python generator asserts that every label occurs in
    # exactly two factors; until now nothing asserted it HERE, where the
    # contraction actually runs, and a count of 52 is not a structure.  This
    # matters because contract_labeled builds `bpositions` as a Dict keyed by
    # label: a label repeated inside B keeps only its LAST position, the earlier
    # occurrence is silently promoted to a free leg of the result, and the
    # closure check at the end can still pass while the number is wrong.  That is
    # the one way this routine can be silently incorrect, so it is checked.
    let counts = Dict{Int,Int}()
        for f in factors
            length(unique(f.labels)) == length(f.labels) ||
                error("junction_window_pair_log: self-trace in a factor")
            for l in f.labels
                counts[l] = get(counts, l, 0) + 1
            end
        end
        all(==(2), values(counts)) ||
            error("junction_window_pair_log: network is not pairwise (labels ",
                  [l for (l,c) in counts if c != 2], " occur != 2 times)")
    end
    length(JUNCTION_PAIR_PATH) == length(factors) - 1 ||
        error("junction_window_pair_log: path has ", length(JUNCTION_PAIR_PATH),
              " steps, need ", length(factors) - 1)

    # Pre-flight.  Replay the path on LABEL SETS only (microseconds) to get the
    # peak intermediate before allocating anything.  Without this a stale or
    # oversized path is discovered by an OOM kill only after the CTMRG
    # environment has already converged -- and the junction is the expensive
    # part (549 s of 616 s at chi1=24).  Measured on this cluster the resident
    # set is 3.0x the peak intermediate (chi1 = 16, 24, 32: 24.8 / 273 / 1543
    # GiB against 24.0 / 273.4 / 1536.0 GiB predicted), so that factor is used
    # to turn the peak into a memory forecast.
    let dims = Dict{Int,Int}(), ops = [copy(f.labels) for f in factors], peak = 0.0
        for f in factors, (k,l) in enumerate(f.labels)
            dims[l] = size(f.data, k)
        end
        for (i0,j0) in JUNCTION_PAIR_PATH
            i = i0+1; j = j0+1
            (1 <= i < j <= length(ops)) ||
                error("junction_window_pair_log: path step (", i0, ",", j0,
                      ") out of range for ", length(ops), " operands")
            common = intersect(ops[i], ops[j])
            out = setdiff(union(ops[i], ops[j]), common)
            peak = max(peak, prod(Float64[dims[l] for l in out]; init=1.0))
            deleteat!(ops, j); deleteat!(ops, i); push!(ops, out)
        end
        forecast = 3.0 * 8 * peak / 2^30
        @printf("JUNCTION-PREFLIGHT peak=%.4g doubles (%.1f GiB) forecast_rss=%.1f GiB cap=%.1f GiB\n",
                peak, 8*peak/2^30, forecast, SRMEMCAP)
        flush(stdout)
        forecast < SRMEMCAP ||
            error("junction_window_pair_log: forecast ", round(forecast, digits=1),
                  " GiB exceeds SRMEMCAP=", SRMEMCAP, " GiB; raise the cap or lower chi1")
    end

    # Cotengra paths use the current zero-based operand positions, remove the
    # contracted pair, and append its result.
    for (i0,j0) in JUNCTION_PAIR_PATH
        i=i0+1; j=j0+1
        i < j || error("junction path must be ordered")
        result = contract_labeled(factors[i],factors[j])
        deleteat!(factors,j); deleteat!(factors,i); push!(factors,result)
    end
    length(factors) == 1 && isempty(factors[1].labels) ||
        error("junction_window_pair_log: path did not close the network")
    scalar = only(factors[1].data)
    log(abs(scalar))+factors[1].logscale
end

"k-fold product corner; replica 1 is the fastest digit."
function product_corner_power(C, k)
    out = copy(C)
    for _ in 2:k
        out = kron(C, out)  # new replica slow, existing bundle fast
    end
    out
end

"Zero-based digit of `index0` belonging to `replica` in a bundled basis."
@inline function digit0(index0::Int, base::Int, replica::Int)
    (index0 ÷ base^(replica-1)) % base
end

"Tensor-product isometry Z_k[(old_bundle,physical_bundle),new_bundle]."
function product_projector(Z1, chi1, d1, k)
    chio = chi1
    chin = size(Z1,2)
    chiok = chio^k; chink = chin^k; dk = d1^k
    Zk = zeros(chiok*dk, chink)
    @inbounds for j0 in 0:chink-1, p0 in 0:dk-1, i0 in 0:chiok-1
        value = 1.0
        for replica in 1:k
            ii = digit0(i0, chio, replica) + 1
            pp = digit0(p0, d1, replica) + 1
            jj = digit0(j0, chin, replica) + 1
            value *= Z1[(pp-1)*chio + ii, jj]
        end
        Zk[p0*chiok+i0+1, j0+1] = value
    end
    Zk
end

"Contract seam bond matrix `G` into one selected virtual leg of a site."
function dress(aT, G, slot::Symbol)
    D = size(aT, 1); out = zeros(D, D, D, D)
    @inbounds for u in 1:D, l in 1:D, dn in 1:D, r in 1:D, x in 1:D
        if slot === :u
            out[u,l,dn,r] += G[u,x] * aT[x,l,dn,r]
        elseif slot === :l
            out[u,l,dn,r] += G[l,x] * aT[u,x,dn,r]
        elseif slot === :r
            out[u,l,dn,r] += G[r,x] * aT[u,l,dn,x]
        else
            error("dress: slot must be :u, :l or :r")
        end
    end
    out
end

"One symmetric CTMRG corner step; return the new corner and fresh isometry."
function ctm_step_Z(C, T, a, chimax)
    chi = size(C,1); dd = size(a,1); cd = chi*dd
    CTl = reshape(C*reshape(T, chi, chi*dd), chi, chi, dd)
    res = reshape(reshape(permutedims(T,(2,1,3)), chi, chi*dd)' * reshape(CTl, chi, chi*dd),
                  chi, dd, chi, dd)
    Cb  = reshape(reshape(permutedims(res,(1,3,2,4)), chi*chi, dd*dd) * reshape(a, dd*dd, dd*dd),
                  chi, chi, dd, dd)
    Cbig = reshape(permutedims(Cb,(1,4,2,3)), cd, cd)
    Cbig = (Cbig + Cbig')/2
    F = eigen(Symmetric(Cbig))
    order = sortperm(abs.(F.values), rev=true)
    keep = order[1:min(chimax, cd)]
    Z = F.vectors[:, keep]
    Cn = Z' * Cbig * Z
    Cn = (Cn + Cn')/2
    Cn ./= maximum(abs, Cn)
    Cn, Z
end

"Spin-flip block CTMRG step with a deterministic parity-resolved cut.

`R` is the Z2 action on the current boundary basis.  Diagonalizing the
enlarged corner separately in the +/- eigenspaces of `kron(Pflip,R)` avoids
roundoff-dependent mixing at a critical degenerate cut."
function ctm_step_Z_flip(C, T, a, chimax, R)
    chi = size(C,1); dd = size(a,1); cd = chi*dd
    dd == 2 || error("ctm_step_Z_flip currently requires the one-replica d=2 tensor")
    CTl = reshape(C*reshape(T, chi, chi*dd), chi, chi, dd)
    res = reshape(reshape(permutedims(T,(2,1,3)), chi, chi*dd)' * reshape(CTl, chi, chi*dd),
                  chi, dd, chi, dd)
    Cb  = reshape(reshape(permutedims(res,(1,3,2,4)), chi*chi, dd*dd) * reshape(a, dd*dd, dd*dd),
                  chi, chi, dd, dd)
    Cbig = reshape(permutedims(Cb,(1,4,2,3)), cd, cd)
    Cbig = (Cbig+Cbig')/2
    Pflip = [0.0 1.0; 1.0 0.0]
    U = kron(Pflip,(R+R')/2)  # physical digit slow, boundary digit fast
    Cbig = (Cbig + U*Cbig*U')/2

    FU = eigen(Symmetric((U+U')/2))
    vectors = Vector{Vector{Float64}}(); values = Float64[]; parities = Int[]
    for parity in (-1,1)
        ids = findall(x -> parity*x > 0, FU.values)
        isempty(ids) && continue
        V = FU.vectors[:,ids]
        FB = eigen(Symmetric((V'*Cbig*V + (V'*Cbig*V)')/2))
        for j in eachindex(FB.values)
            v = V*FB.vectors[:,j]
            pivot = argmax(abs.(v)); v[pivot] < 0 && (v .*= -1)
            push!(vectors,v); push!(values,FB.values[j]); push!(parities,parity)
        end
    end
    order = sortperm(eachindex(values), by=i->(-abs(values[i]),-parities[i]))
    keep = order[1:min(chimax,cd)]
    Z = hcat(vectors[keep]...)
    Cn = Z'*Cbig*Z; Cn = (Cn+Cn')/2; Cn ./= maximum(abs,Cn)
    Rn = Matrix(Diagonal(Float64.(parities[keep])))
    Cn,Z,Rn
end

"Project a one-replica edge onto global spin-flip covariance."
function sym_flip_edge(T,R)
    chi = size(T,1); dd = size(T,3)
    dd == 2 || error("sym_flip_edge requires d=2")
    A = reshape(R*reshape(T,chi,chi*dd),chi,chi,dd)
    B = permutedims(A,(2,1,3))
    B = reshape(R*reshape(B,chi,chi*dd),chi,chi,dd)
    B = permutedims(B,(2,1,3))
    (T+B[:,:,2:-1:1])/2
end

"Grow one edge by `adr` and truncate both enlarged boundary legs with `Z`."
function grow_T_raw(T, adr, Z)
    chi = size(T,1); dd = size(adr,1); cd = chi*dd
    res5 = reshape(reshape(permutedims(T,(3,1,2)), dd, chi*chi)' * reshape(adr, dd, dd*dd*dd),
                   chi, chi, dd, dd, dd)
    Tbig = reshape(permutedims(res5,(1,3,2,5,4)), cd, cd, dd)
    chin = size(Z,2)
    Tmid = reshape(Z' * reshape(Tbig, cd, cd*dd), chin, cd, dd)
    permutedims(
        reshape(Z' * reshape(permutedims(Tmid,(2,1,3)), cd, chin*dd), chin, chin, dd),
        (2,1,3),
    )
end

"Max-normalize an edge and fix its overall sign relative to `reference`."
function normalize_edge(T; reference=nothing)
    T ./= maximum(abs, T)
    if reference !== nothing && size(reference) == size(T)
        overlap = sum(T .* reference)
        overlap < 0 && (T .*= -1)
    end
    T
end

"Gauge-invariant probe: the two reported k=2 observables of the current
 environment.  Cost is a few window sweeps, i.e. the same order as one edge
 growth, so it is evaluated only every SRCHECK iterations."
function probe_observables(C1, T1, Td2, a2, a2Gu)
    C2 = product_corner_power(C1, 2)
    T2 = product_edge_power(T1, 2)
    Z0_1 = window_sweep(C2,[T2],[T2],[T2],[T2],[[a2]])
    Z0_2 = window_sweep(C2,[T2,T2],[T2,T2],[T2,T2],[T2,T2],[[a2,a2],[a2,a2]])
    Zs1  = window_sweep(C2,[T2],[flipT(Td2)],[T2],[Td2],[[a2Gu]])
    Zs2  = window_sweep(C2,[T2,T2],[flipT(Td2),T2],[T2,T2],[Td2,T2],
                        [[a2Gu,a2],[a2Gu,a2]])
    Zb1  = window_sweep(C2,[Td2],[T2],[T2],[Td2],[[a2Gu]])
    s2 = -(log(abs(Zs2/Z0_2)) - log(abs(Zs1/Z0_1)))
    kA = log(abs(Zb1/Zs1))
    (s2, kA)
end

"Synchronously evolve one-replica CTMRG and one canonical k=2 swap edge.

All k=4 straight defects are exact pair products of the returned `Td2`, so
there is no k=4 projector or edge grow inside this fixed-point loop."
function coevolved_pair_environment(a1, a2, adr2, a2Gu, chi1;
                                    nmax=5000, ctol=1e-13, dtol=SRTOL)
    C1,T1 = init_CT(a1)
    R1 = [0.0 1.0; 1.0 0.0]
    Td2 = product_edge_power(T1,2)
    spectrum_old = Float64[]
    cdrift = Inf; d2res = Inf; plain2 = Inf; itdone = nmax
    obs_old = (NaN, NaN); obsdrift = Inf; obspass = 0
    for it in 1:nmax
        local Cn, Z1, R1n
        if SRFLIP == 1
            Cn,Z1,R1n = ctm_step_Z_flip(C1,T1,a1,chi1,R1)
        else
            Cn,Z1 = ctm_step_Z(C1,T1,a1,chi1)
            R1n = R1                      # unused when the projection is off
        end
        Z2 = product_projector(Z1,size(T1,1),2,2)
        T1n = normalize_edge(grow_T_raw(T1,a1,Z1);reference=T1)
        T1n = (T1n+permutedims(T1n,(2,1,3)))/2
        SRFLIP == 1 && (T1n = sym_flip_edge(T1n,R1n))
        T1n = normalize_edge(T1n;reference=T1)
        Td2n = normalize_edge(grow_T_raw(Td2,adr2,Z2);reference=Td2)
        d2res = size(Td2n)==size(Td2) ? maximum(abs.(Td2n-Td2)) : Inf

        spectrum = sort(abs.(eigvals(Symmetric(Cn))),rev=true)
        spectrum ./= spectrum[1]
        cdrift = length(spectrum)==length(spectrum_old) ?
                 maximum(abs.(spectrum-spectrum_old)) : Inf
        if SRTRACE == 1 && (it == 1 || it % SRCHECK == 0)
            tr_s2, tr_kA = probe_observables(Cn, T1n, Td2n, a2, a2Gu)
            @printf("  TRACE it=%5d s2=%.10f kA=%+.10f d2res=%.2e\n",
                    it, tr_s2, tr_kA, d2res); flush(stdout)
        end
        if SRSTOP == "obs"
            converged = false
            if cdrift < ctol && it % SRCHECK == 0
                obs_new = probe_observables(Cn, T1n, Td2n, a2, a2Gu)
                obsdrift = maximum(abs.(collect(obs_new) .- collect(obs_old)))
                obs_old = obs_new
                # Two consecutive passes: a slow mode can look converged inside
                # a single SRCHECK window.  This can only postpone the stop.
                obspass = obsdrift < dtol ? obspass + 1 : 0
                converged = obspass >= 2
            end
        else
            converged = cdrift < ctol && d2res < dtol
        end

        # Empirical k=2 product gate at startup and convergence.  The k=4
        # plain edge is an exact product by construction in pair mode.
        if it == 1 || converged
            T2old = product_edge_power(T1,2)
            T2n = product_edge_power(T1n,2)
            T2grown = normalize_edge(grow_T_raw(T2old,a2,Z2);reference=T2n)
            plain2 = maximum(abs.(T2grown-T2n))
        end
        C1,T1,Td2,R1 = Cn,T1n,Td2n,R1n
        spectrum_old = spectrum
        if converged
            itdone = it
            break
        end
        if it % 100 == 0
            @printf("    [pair it=%d cdrift=%.2e d2=%.2e p2=%.2e]\n",
                    it,cdrift,d2res,plain2)
            flush(stdout)
        end
    end
    itdone==nmax && @printf("WARN pair coevolve hit nmax=%d cdrift=%.2e d2=%.2e obsdrift=%.2e\n",
                            nmax,cdrift,d2res,obsdrift)
    C1,T1,Td2,itdone,cdrift,d2res,plain2,obsdrift
end

"Legacy oracle: synchronously evolve one-, two-, and four-replica edges.

The projector is never frozen: every fresh one-replica Z1 is lifted to
Z1^(x2) and Z1^(x4) in the same sweep.  This preserves the physical CTMRG
environment while avoiding any independent replica-sector truncation."
function coevolved_environments_legacy(a1, a2, a4, adr2, adrs4, chi1;
                                       nmax=50000, ctol=1e-13, dtol=SRTOL)
    C1,T1 = init_CT(a1)
    Td2 = product_edge_power(T1,2)
    Tds4 = [product_edge_power(T1,4) for _ in adrs4]
    spectrum_old = Float64[]
    cdrift = Inf; d2res = Inf; d4res = fill(Inf,length(adrs4))
    plain2 = Inf; plain4 = Inf; itdone = nmax
    for it in 1:nmax
        Cn,Z1 = ctm_step_Z(C1,T1,a1,chi1)
        Z2 = product_projector(Z1,size(T1,1),2,2)
        Z4 = product_projector(Z1,size(T1,1),2,4)

        T1n = normalize_edge(grow_T_raw(T1,a1,Z1);reference=T1)
        T1n = (T1n+permutedims(T1n,(2,1,3)))/2
        T1n = normalize_edge(T1n;reference=T1)
        Td2n = normalize_edge(grow_T_raw(Td2,adr2,Z2);reference=Td2)
        Tds4n = [normalize_edge(grow_T_raw(Td,adr,Z4);reference=Td)
                  for (Td,adr) in zip(Tds4,adrs4)]
        if size(Td2n)==size(Td2)
            d2res = maximum(abs.(Td2n-Td2))
            d4res = [maximum(abs.(new-old)) for (new,old) in zip(Tds4n,Tds4)]
        else
            d2res = Inf
            d4res .= Inf
        end

        spectrum = sort(abs.(eigvals(Symmetric(Cn))),rev=true)
        spectrum ./= spectrum[1]
        if length(spectrum)==length(spectrum_old)
            cdrift = maximum(abs.(spectrum-spectrum_old))
        else
            cdrift = Inf
        end
        converged = cdrift < ctol && d2res < dtol && maximum(d4res) < dtol

        # This is a validation gate, not part of the evolution.  In
        # particular, materializing and growing the plain four-replica edge
        # at every iteration is needlessly expensive for chi1 >= 4.  Check it
        # once at startup and once again for the converged environment.
        if it == 1 || converged
            T2old = product_edge_power(T1,2)
            T4old = product_edge_power(T1,4)
            T2n = product_edge_power(T1n,2)
            T4n = product_edge_power(T1n,4)
            T2grown = normalize_edge(grow_T_raw(T2old,a2,Z2);reference=T2n)
            T4grown = normalize_edge(grow_T_raw(T4old,a4,Z4);reference=T4n)
            plain2 = maximum(abs.(T2grown-T2n))
            plain4 = maximum(abs.(T4grown-T4n))
        end
        C1,T1,Td2,Tds4 = Cn,T1n,Td2n,Tds4n
        spectrum_old = spectrum
        if converged
            itdone = it
            break
        end
        if it % 100 == 0
            @printf("    [coevolve it=%d cdrift=%.2e d2=%.2e d4=%.2e p2=%.2e p4=%.2e]\n",
                    it,cdrift,d2res,maximum(d4res),plain2,plain4)
            flush(stdout)
        end
    end
    itdone==nmax && @printf("WARN coevolve hit nmax=%d cdrift=%.2e d2=%.2e d4=%.2e\n",
                            nmax,cdrift,d2res,maximum(d4res))
    C1,T1,Td2,Tds4,itdone,cdrift,d2res,d4res,plain2,plain4
end

# ---------------- observables --------------------------------------------------
"""Correlation length from the boundary transfer channel.

In the ordered phase the two leading eigenvalues are the degenerate
symmetric/antisymmetric pair of the two ferromagnetic sectors; their ratio
tends to one and the naive `-1/log(lambda_2/lambda_1)` diverges (the observed
1e15 / -Inf entries).  That pair encodes the spontaneous magnetization, not a
decay.  The connected spin correlation length is set by the next eigenvalue
that is NOT degenerate with the leading one.  `degtol` selects it; the
splitting itself is returned as an order-parameter diagnostic."""
function xi_from_T(T; degtol=1e-10)
    chi = size(T,1); d = size(T,3)
    chi == 1 && return (0.0, 0.0, 0.0)
    E = zeros(chi*chi, chi*chi)
    for p in 1:d
        A = T[:,:,p]
        E .+= kron(A,A)
    end
    values = sort(abs.(eigvals(Symmetric((E+E')/2))), rev=true)
    ratios = values ./ values[1]
    splitting = 1 - ratios[2]                     # ~0 <=> long-range order
    k = findfirst(r -> r < 1 - degtol, ratios[2:end])
    xi_naive = ratios[2] < 1 ? -1/log(ratios[2]) : Inf
    k === nothing && return (Inf, xi_naive, splitting)
    (-1/log(ratios[k+1]), xi_naive, splitting)
end

flipT(T) = permutedims(T,(2,1,3))

# Ring/window convention copied from ctmrg_stilde.jl.  It is already gated
# there against the direct n=1/2 contractions.
function window_sweep(C, TNs, TEs, TSs, TWs, cols)
    n = length(TNs); d = size(cols[1][1],1); chi = size(C,1)
    dn_ = d^n
    v = zeros(chi, ntuple(_->d,n)..., chi)
    for I in CartesianIndices(ntuple(_->d,n))
        B = C
        for y in 1:n; B = B*TWs[y][:,:,I[y]]; end
        B = B*C
        @views v[:,I,:] .= B
    end
    for x in 1:n
        TSf = flipT(TSs[x]); TNx = TNs[x]
        M = reshape(v, chi*dn_, chi) * reshape(TNx, chi, chi*d)
        t = reshape(M, chi, ntuple(_->d,n)..., chi, d)
        if n == 1
            s1 = cols[x][1]
            tp = reshape(permutedims(t,(1,3,2,4)), chi*chi, d*d)
            S1 = reshape(permutedims(s1,(2,1,3,4)), d*d, d*d)
            q = reshape(tp*S1, chi, chi, d, d)
            qp = reshape(permutedims(q,(2,4,1,3)), chi*d, chi*d)
            TSm = reshape(permutedims(TSf,(1,3,2)), chi*d, chi)
            r = reshape(qp*TSm, chi, d, chi)
            v = permutedims(r,(3,2,1))
        else
            s1, s2 = cols[x][1], cols[x][2]
            tp = reshape(permutedims(t,(1,2,4,3,5)), chi*d*chi, d*d)
            S2 = reshape(permutedims(s2,(2,1,3,4)), d*d, d*d)
            q = reshape(tp*S2, chi, d, chi, d, d)
            qp = reshape(permutedims(q,(1,3,5,2,4)), chi*chi*d, d*d)
            S1 = reshape(permutedims(s1,(2,1,3,4)), d*d, d*d)
            w = reshape(qp*S1, chi, chi, d, d, d)
            wp = reshape(permutedims(w,(2,3,5,1,4)), chi*d*d, chi*d)
            TSm = reshape(permutedims(TSf,(1,3,2)), chi*d, chi)
            r = reshape(wp*TSm, chi, d, d, chi)
            v = permutedims(r,(4,3,2,1))
        end
    end
    value = 0.0
    for I in CartesianIndices(ntuple(_->d,n))
        B = C
        for y in n:-1:1; B = B*TEs[y][:,:,I[y]]; end
        B = B*C
        value += sum(transpose(@view v[:,I,:]) .* B)
    end
    value
end

function relerr(x,y)
    abs(x-y)/max(abs(y),1e-300)
end

# ---------------- main ---------------------------------------------------------
beta = SRBETA
@printf("SPLIT-REPLICA-CTMRG beta=%.12f chis1=%s nmax=%d tol=%.1e mode=%s flip=%d\n",
        beta, string(SRCHIS), SRNMAX, SRTOL, SRMODE, SRFLIP)
# am1 is the sigma-inserted site tensor; it was previously discarded.  It is
# kept now so the one-replica environment can report a magnetization, because
# "is the state symmetry broken?" is a question the entropies alone do not
# answer.  Note what the answer MUST be in production: SRFLIP=1 twirls the
# environment over the global spin flip, so <sigma> is projected to zero by
# construction and the state computed is the Z2-symmetric cat state, not a
# broken one.  Reporting m therefore certifies the twirl rather than measuring
# order; the order itself shows up in z2split, which collapses to ~1e-16 in the
# ordered phase because the two leading transfer eigenvalues become degenerate.
# Running the same beta with SRFLIP=0 lets the environment break and gives the
# spontaneous value for comparison against Onsager.
a1, am1 = ising_tensors(beta)
a2 = replica_site(a1)
G2 = Matrix(seam_G(2,beta,'C','A',[1,2],[1,2],PSWAP2))
# Fused k=4 objects are constructed only for the legacy/full regression
# oracle.  Production pair mode never allocates them.
a4 = NEED_EXPLICIT4 ? replica_site(a2) : nothing
GCA = NEED_EXPLICIT4 ? Matrix(seam_G(4,beta,'C','A',SA4,SB4,SC4)) : nothing
GCB = NEED_EXPLICIT4 ? Matrix(seam_G(4,beta,'C','B',SA4,SB4,SC4)) : nothing
GAB = NEED_EXPLICIT4 ? Matrix(seam_G(4,beta,'A','B',SA4,SB4,SC4)) : nothing

for chi1 in SRCHIS
    chi_start = time()
    println("="^88)
    @printf("chi1=%d  implicit chi2=%d chi4=%d\n", chi1, chi1^2, chi1^4)
    adr2 = dress(a2,G2,:r)
    env_result = nothing
    env_seconds = @elapsed begin
        if SRMODE == "legacy"
            adrs4 = [dress(a4,GCA,:r),dress(a4,GCB,:r),dress(a4,GAB,:r)]
            env_result = coevolved_environments_legacy(
                a1,a2,a4,adr2,adrs4,chi1;
                nmax=SRNMAX,ctol=1e-13,dtol=SRTOL,
            )
        else
            env_result = coevolved_pair_environment(
                a1,a2,adr2,dress(a2,G2,:u),chi1;nmax=SRNMAX,ctol=1e-13,dtol=SRTOL,
            )
        end
    end
    if SRMODE == "legacy"
        C1,T1,Td2,Tds4,it1,dr1,resd2,resd4,plain2,plain4 = env_result
        obsdrift = NaN
        pair_edges = [paired_edge4(Td2,((1,3),(2,4))),
                      paired_edge4(Td2,((1,4),(2,3))),
                      paired_edge4(Td2,((1,2),(3,4)))]
        pairres = maximum(maximum(abs.(a-b)) for (a,b) in zip(pair_edges,Tds4))
        it4reported = it1
    else
        C1,T1,Td2,it1,dr1,resd2,plain2,obsdrift = env_result
        # Explicit paired k=4 edges are needed only by the fused regression.
        Tds4 = NEED_EXPLICIT4 ?
            [paired_edge4(Td2,((1,3),(2,4))),
             paired_edge4(Td2,((1,4),(2,3))),
             paired_edge4(Td2,((1,2),(3,4)))] : nothing
        resd4 = fill(resd2,3)
        plain4 = 0.0  # structural identity: T4 is never independently truncated
        pairres = 0.0
        it4reported = 0
    end
    xi1, xi1_naive, xi1_split = xi_from_T(T1)
    @printf("  synchronized env: mode=%s it=%d drift=%.2e xi1=%.8f xi1_naive=%.3e z2split=%.3e plain2=%.2e plain4=%.2e env_seconds=%.3f\n",
            SRMODE,it1,dr1,xi1,xi1_naive,xi1_split,plain2,plain4,env_seconds)
    @printf("  pair construction: legacy_difference=%.2e k4_fixed_point_iterations=%d\n",
            pairres,it4reported)

    # k=2 product environment and seam channel.
    C2 = product_corner_power(C1,2)
    T2 = product_edge_power(T1,2)
    a2Gu = dress(a2,G2,:u)
    Z0_1 = window_sweep(C2,[T2],[T2],[T2],[T2],[[a2]])
    Z0_2 = window_sweep(C2,[T2,T2],[T2,T2],[T2,T2],[T2,T2],[[a2,a2],[a2,a2]])
    Zs1 = window_sweep(C2,[T2],[flipT(Td2)],[T2],[Td2],[[a2Gu]])
    Zs2 = window_sweep(C2,[T2,T2],[flipT(Td2),T2],[T2,T2],[Td2,T2],
                       [[a2Gu,a2],[a2Gu,a2]])
    Zb1 = window_sweep(C2,[Td2],[T2],[T2],[Td2],[[a2Gu]])
    s2 = -(log(abs(Zs2/Z0_2))-log(abs(Zs1/Z0_1)))
    kappaA = log(abs(Zb1/Zs1))
    @printf("  k2 product: plain_gate=%.2e seam_res=%.2e s2=%.10f kA=%+.10f\n",
            plain2,resd2,s2,kappaA)

    # k=4 product environment.  Production evaluates it through the explicit
    # one-/two-replica factors below; fused objects exist only in fullgate.
    resCA,resCB,resAB = resd4
    # Every straight double-transposition window is exactly the square of
    # its canonical k=2 window.  Production pair mode therefore evaluates
    # only the genuinely four-replica Y junction.  SRFULLGATE=1 retains the
    # eight redundant explicit straight/plain windows as a regression gate.
    W0_1 = Z0_1^2; W0_2 = Z0_2^2
    Ws1_pair = Zs1^2; Ws2_pair = Zs2^2
    tensions = fill(2s2,3)
    asym = 0.0; gate = eps(Float64)  # exact structurally; finite for log plots
    straight_regression = 0.0
    if NEED_EXPLICIT4
        C4 = product_corner_power(C1,4)
        T4 = product_edge_power(T1,4)
        TdCA,TdCB,TdAB = Tds4
        aCAu = dress(a4,GCA,:u)
        aCBu = dress(a4,GCB,:u)
        aABr = dress(a4,GAB,:r)
        W0_1e = window_sweep(C4,[T4],[T4],[T4],[T4],[[a4]])
        W0_2e = window_sweep(C4,[T4,T4],[T4,T4],[T4,T4],[T4,T4],[[a4,a4],[a4,a4]])
        explicit = Float64[]
        for (Td,aGu) in ((TdCA,aCAu),(TdCB,aCBu))
            Ws1e = window_sweep(C4,[T4],[flipT(Td)],[T4],[Td],[[aGu]])
            Ws2e = window_sweep(C4,[T4,T4],[flipT(Td),T4],[T4,T4],[Td,T4],
                                [[aGu,a4],[aGu,a4]])
            push!(explicit,-(log(abs(Ws2e/W0_2e))-log(abs(Ws1e/W0_1e))))
            straight_regression = max(straight_regression,relerr(Ws1e,Ws1_pair),
                                      relerr(Ws2e,Ws2_pair))
        end
        WsAB1e = window_sweep(C4,[TdAB],[T4],[flipT(TdAB)],[T4],[[aABr]])
        WsAB2e = window_sweep(C4,[TdAB,T4],[T4,T4],[flipT(TdAB),T4],[T4,T4],
                              [[aABr,aABr],[a4,a4]])
        push!(explicit,-(log(abs(WsAB2e/W0_2e))-log(abs(WsAB1e/W0_1e))))
        straight_regression = max(straight_regression,relerr(W0_1e,W0_1),
                                  relerr(W0_2e,W0_2),relerr(WsAB1e,Ws1_pair),
                                  relerr(WsAB2e,Ws2_pair))
        tensions = explicit
        asym = (maximum(tensions)-minimum(tensions))/maximum(abs.(tensions))
        gate = maximum(abs.(tensions .- 2s2))/max(abs(2s2),1e-300)
    end
    @printf("  k4 product: plain_gate=%.2e seam_res=(%.1e,%.1e,%.1e)\n",
            plain4,resCA,resCB,resAB)
    @printf("  straight tensions: CA=%.10f CB=%.10f AB=%.10f 2s2=%.10f asym=%.2e gate=%.2e\n",
            tensions[1],tensions[2],tensions[3],2s2,asym,gate)
    @printf("  straight factorization: explicit_regression=%.2e fullgate=%d\n",
            straight_regression,SRFULLGATE)

    # Exact factorized junction.  Its three dressed pair sites use the same
    # canonical relative-swap G2; only their pair labels and leg slots differ.
    a2Gu = dress(a2,G2,:u)
    a2Gr = dress(a2,G2,:r)
    logWY_pair = 0.0
    junction_seconds = @elapsed logWY_pair =
        junction_window_pair_log(C1,T1,Td2,a1,a2Gu,a2Gr)
    junction_regression = -1.0  # sentinel: fused oracle was not evaluated
    logWY = logWY_pair
    if NEED_EXPLICIT4
        WY_explicit = window_sweep(C4,[TdAB,T4],[flipT(TdCB),T4],[T4,T4],[TdCA,T4],
                                   [[aCAu,aABr],[aCBu,a4]])
        logWY_explicit = log(abs(WY_explicit))
        junction_regression = abs(expm1(logWY_pair-logWY_explicit))
        SRMODE == "legacy" && (logWY = logWY_explicit)
    end
    kappaY = logWY-3log(abs(Zs2))+log(abs(Z0_2))
    tSJ = 2kappaA-kappaY
    total_seconds = time()-chi_start
    @printf("  junction: kY=%+.10f tSJ=%+.10f pair_vs_fused=%.2e seconds=%.3f\n",
            kappaY,tSJ,junction_regression,junction_seconds)
    # <sigma> in the converged one-replica environment, and the exact Onsager
    # spontaneous value for reference.  With SRFLIP=1 mag must come out at
    # roundoff; a nonzero mag would mean the twirl is not being applied.
    mag = magnetization(C1, T1, a1, am1)
    magexact = onsager_m(beta)

    @printf("  SUMMARY-SPLIT beta=%.12f chi1=%d chi2=%d chi4=%d xi1=%.8f z2split=%.3e plain2=%.2e plain4=%.2e s2=%.10f kA=%+.10f tCA=%.10f tCB=%.10f tAB=%.10f asym=%.2e factgate=%.2e kY=%+.10f tSJ=%+.10f it1=%d it2d=%d it4max=%d mode=%s stop=%s flip=%d mag=%.3e magexact=%.6f obsdrift=%.2e pairres=%.2e juncgate=%.2e envsec=%.3f juncsec=%.3f totalsec=%.3f\n",
            beta,chi1,chi1^2,chi1^4,xi1,xi1_split,plain2,plain4,s2,kappaA,
            tensions[1],tensions[2],tensions[3],asym,gate,kappaY,tSJ,
            it1,it1,it4reported,SRMODE,SRSTOP,SRFLIP,mag,magexact,obsdrift,pairres,junction_regression,
            env_seconds,junction_seconds,total_seconds)
    flush(stdout)
end
println("DONE SPLIT-REPLICA-CTMRG")
