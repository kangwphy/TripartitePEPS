# mc_tildeS.jl -- Monte Carlo for the tripartite multi-entropy, DIRECT estimator.
# =============================================================================
# Identity:  exp(-tildeS) = Z4 * Z1^2 / (Z2A * Z2B * Z2C).
# Both sides are SIX coupled Ising layers on the open LxL lattice:
#   ensemble D (denominator): layers (1,2)=2A pair, (3,4)=2B, (5,6)=2C
#   ensemble N (numerator)  : layers (1..4)=the 4-replica cube, (5),(6)=free
# Interior bonds are identical (beta * Id per layer) -> they cancel; the two
# ensembles differ ONLY on seam bonds, where the coupling matrix is
#   J = (beta/2) * (Id + P_rel),  P_rel[a,b]=1 iff b = pj(pi^{-1}(a)),
# i.e. every coupling is a FERROMAGNETIC pair coupling -> Swendsen-Wang valid,
# including along the whole thermodynamic-integration path
#   J_lambda = (1-lambda) J_D + lambda J_N   (still ferromagnetic pairwise).
# TI:  tildeS = -int_0^1 dlambda < dE/dlambda >_lambda,
#      dE/dlambda = sum_{seam bonds} sum_{a,b} (J_N - J_D)[a,b] s_a(i) s_b(j).
# The target is O(0.1) (extensive seam parts cancel exactly), so modest
# statistics resolve it -- unlike per-logZ estimation.
# Gates: L=8/10 vs certified bmps values.  SLURM smp/preempt only.
# Env: MCL (L), MCB (beta or "crit"), MCSW (measure sweeps), MCSEED,
#      MCUPD ("sw" or "local").  A local sweep is one checkerboard
#      Metropolis attempt per one of the 6*L^2 layer spins.
# =============================================================================
using Random, Printf, Statistics

const BETA_C = 0.5*log(1+sqrt(2))
const L    = parse(Int, ENV["MCL"])
const bstr = ENV["MCB"]
const beta = bstr == "crit" ? BETA_C : parse(Float64, bstr)
const NSW  = parse(Int, get(ENV, "MCSW", "4000"))
const NGL  = parse(Int, get(ENV, "MCGL", "12"))
const QUAD = lowercase(get(ENV, "MCQUAD", "gl"))  # gl | trap
const NEQO = parse(Int, get(ENV, "MCEQ", "0"))
const REV  = get(ENV, "MCREV", "0") == "1"      # sweep lambda 1->0 (hysteresis test)
const PROF = get(ENV, "MCPROF", "0") == "1"     # per-seam-bond resolved integrand profile
const FRESH= get(ENV, "MCFRESH", "0") == "1"    # random restart + long equil per node
const INIT = lowercase(get(ENV, "MCINIT", "plus")) # plus | random | sector
const SECTOR_MOVES = get(ENV, "MCSECTOR", "0") == "1"
const SECTOR_EVERY = parse(Int, get(ENV, "MCSECTOREVERY", "1"))
const SECTOR_MODE = lowercase(get(ENV, "MCSECTORMODE", "single")) # single | subset
const SECTOR_CHECK = get(ENV, "MCSECTORCHECK", "0") == "1" # expensive test only
const seed = parse(Int, get(ENV, "MCSEED", "1"))
const MCQ  = get(ENV, "MCQ", "tS")   # tS | S2A/S2B/S2C | S3 | K4C
const UPD  = lowercase(get(ENV, "MCUPD", "sw"))  # sw | local
const MCOUT = get(ENV, "MCOUT", "")             # optional per-chain output directory
const MCSECTOROUT = get(ENV, "MCSECTOROUT", "") # optional per-node sector diagnostics
@assert UPD in ("sw", "local") "MCUPD must be sw or local"
@assert QUAD in ("gl", "trap") "MCQUAD must be gl or trap"
@assert INIT in ("plus", "random", "sector") "MCINIT must be plus, random, or sector"
@assert SECTOR_MODE in ("single", "subset") "MCSECTORMODE must be single or subset"
@assert SECTOR_EVERY >= 1 "MCSECTOREVERY must be positive"
@assert !SECTOR_MOVES || UPD == "local" "global sector proposals are defined for local update runs"
const NL   = MCQ in ("S2A", "S2B", "S2C") ? 2 : MCQ == "S3" ? 4 : 6
const rng  = Xoshiro(seed)

"""
    region(x, y) -> Char

Assign the zero-based lattice coordinate `(x,y)` to region `A`, `B`, or `C`.
For even `L`, `C` is the upper half, and the lower half is split into `A`
(left) and `B` (right).  The global lattice size `L` is used.
"""
region(x, y) = (y >= L ÷ 2) ? 'C' : (x < L ÷ 2 ? 'A' : 'B')
# replica perms (symmetric gauge, as the bmps runs)
# cube sigma_X in the σ_A=1 gauge (user's multientropy_PEPS.pdf); Z4 gauge invariant.
# NOTE: in this gauge q_AB = (12)(34) = D's effective pairing on layers 1-4, so the
# A|B seam is INACTIVE in the TI (tN==tD numerically) — the N/D difference lives
# entirely on the ∂C line. Different TI path, same endpoints (path independence).
const permA = if MCQ == "K4C"
    # factorizable null: sigma_A=sigma_B=(12)(34), sigma_C=id
    Dict('A'=>[2,1,4,3], 'B'=>[2,1,4,3], 'C'=>[1,2,3,4])
elseif get(ENV, "MCGAUGE", "user") == "sym"
    # old symmetric gauge, retained for comparison runs
    Dict('A'=>[3,4,1,2], 'B'=>[2,1,4,3], 'C'=>[4,3,2,1])
else
    Dict('A'=>[1,2,3,4], 'B'=>[2,1,4,3], 'C'=>[3,4,1,2])
end
# layer blocks: N: cube on 1..4 via permA, layers 5,6 free
#               D: (1,2) 2A-pair, (3,4) 2B-pair, (5,6) 2C-pair
sw2 = [2,1]; idp = [1,2]

"""
    block_J(pi_, pj_, off, beta) -> Vector{Tuple{Int,Int,Float64}}

Construct the nonzero pair couplings for one spatial bond in a `k`-replica
block.  `pi_` and `pj_` are the replica permutations at the two endpoints,
and `off` embeds the local replica labels into the full layer numbering.

Each replica contributes a ket edge and a permutation-twisted bra edge, both
with strength `beta/2`.  Coincident edges are accumulated in the dictionary,
so an untwisted ket--bra pair is returned as a single edge of strength `beta`.
The result consists of triples `(layer_i, layer_j, coupling)`.
"""
function block_J(pi_, pj_, off, beta)
    k = length(pi_)
    T = Dict{Tuple{Int,Int},Float64}()
    for r in 1:k
        T[(off+r, off+r)] = get(T,(off+r,off+r),0.0) + beta/2       # ket factor
        T[(off+pi_[r], off+pj_[r])] = get(T,(off+pi_[r],off+pj_[r]),0.0) + beta/2  # bra
    end
    [(a,b,J) for ((a,b),J) in T]
end

"""
    bond_couplings(x1, y1, x2, y2) -> (tN, tD)

Return the coupling-triple lists on the nearest-neighbor bond from
`(x1,y1)` to `(x2,y2)` for the numerator (`N`) and denominator (`D`)
ensembles.  Coordinates are zero based.

Interior bonds are identical in the two ensembles.  On a region boundary,
the lists implement the replica seams appropriate to `MCQ=tS`, `S2C`, or
`K4C`.  For an interior bond the same vector object is returned twice; the
bond builder uses this object identity to mark only region-boundary bonds as
seams.
"""
function bond_couplings(x1,y1,x2,y2)
    r1 = region(x1,y1); r2 = region(x2,y2)
    if r1 == r2                                        # interior: same in N and D
        t = [(l,l,beta) for l in 1:NL]
        return (t, t)
    end
    if MCQ in ("S2A", "S2B", "S2C")
        # N = Z_2X (two replicas, swap on X boundary), D = Z_1^2.
        target = MCQ == "S2A" ? 'A' : MCQ == "S2B" ? 'B' : 'C'
        pC = Dict('A'=>idp, 'B'=>idp, 'C'=>idp); pC[target] = sw2
        tN = block_J(pC[r1], pC[r2], 0, beta)
        tD = [(1,1,beta),(2,2,beta)]
        return (tN, tD)
    end
    if MCQ == "S3"
        # N = Z4 (four-copy cube); D = Z1^4 (four independent layers).
        tN = block_J(permA[r1], permA[r2], 0, beta)
        tD = [(l,l,beta) for l in 1:4]
        return (tN, tD)
    end
    pN1 = permA[r1]; pN2 = permA[r2]
    tN = vcat(block_J(pN1, pN2, 0, beta), [(5,5,beta),(6,6,beta)])
    # D: pair (1,2) twisted iff seam belongs to A's boundary in sector 2A, etc.
    p2 = Dict('A'=>sw2, 'B'=>idp, 'C'=>idp)
    tD12 = block_J(p2[r1], p2[r2], 0, beta)
    p2 = Dict('A'=>idp, 'B'=>sw2, 'C'=>idp)
    tD34 = block_J(p2[r1], p2[r2], 2, beta)
    p2 = Dict('A'=>idp, 'B'=>idp, 'C'=>sw2)
    tD56 = block_J(p2[r1], p2[r2], 4, beta)
    (tN, vcat(tD12, tD34, tD56))
end

# ---- build bond lists ----
"A spatial bond together with its numerator and denominator layer couplings."
struct Bond
    i::Int
    j::Int
    tN::Vector{Tuple{Int,Int,Float64}}
    tD::Vector{Tuple{Int,Int,Float64}}
    seam::Bool
end

"""
    site(x, y) -> Int

Map a zero-based lattice coordinate to the one-based, row-major site index
used internally.  The global lattice width is `L`.
"""
site(x,y) = y*L + x + 1
const bonds = Bond[]
for y in 0:L-1, x in 0:L-1
    if x < L-1
        tN,tD = bond_couplings(x,y,x+1,y)
        push!(bonds, Bond(site(x,y), site(x+1,y), tN, tD, tN !== tD))
    end
    if y < L-1
        tN,tD = bond_couplings(x,y,x,y+1)
        push!(bonds, Bond(site(x,y), site(x,y+1), tN, tD, tN !== tD))
    end
end
nseam = count(b->b.seam, bonds)
# seam metadata for PROF: arm ('V'=A|B vertical, 'W'=A|C west, 'E'=B|C east) and
# r = graph distance from the junction site (L/2, L/2) along the seam
seam_idx = Int[]; seam_arm = Char[]; seam_r = Int[]
for (bi, b) in enumerate(bonds)
    b.seam || continue
    x1 = (b.i-1) % L; y1 = (b.i-1) ÷ L
    x2 = (b.j-1) % L; y2 = (b.j-1) ÷ L
    push!(seam_idx, bi)
    if x1 != x2          # vertical seam A|B: bond ((L/2-1,y),(L/2,y)), junction at y=L/2
        push!(seam_arm, 'V'); push!(seam_r, L÷2 - y1)
    elseif x1 < L÷2      # horizontal west arm A|C
        push!(seam_arm, 'W'); push!(seam_r, L÷2 - x1)
    else                  # horizontal east arm B|C
        push!(seam_arm, 'E'); push!(seam_r, x1 - L÷2 + 1)
    end
end
bond_pos = Dict(bi => k for (k, bi) in enumerate(seam_idx))
@printf("mc_tildeS: L=%d beta=%.10f sweeps=%d eq=%d seed=%d  bonds=%d seam=%d  gauge=%s update=%s quantity=%s reverse=%s fresh=%s init=%s sector_moves=%s sector_mode=%s\n",
        L, beta, NSW, NEQO > 0 ? NEQO : max(300, NSW ÷ 10), seed,
        length(bonds), nseam, get(ENV,"MCGAUGE","user"), UPD, MCQ,
        REV, FRESH, INIT, SECTOR_MOVES, SECTOR_MODE); flush(stdout)

# A union--find vertex represents one `(layer, spatial site)` spin.
"""
    nid(l, s) -> Int

Map one-based layer `l` and one-based spatial site `s` to the flattened spin
index.  `NL` is two in `S2C` mode and six otherwise.
"""
nid(l, s) = (s-1)*NL + l
const NN = NL*L*L
const spins = ones(Int8, NN)
const layer_mags = zeros(Int, NL)

"Recompute the total magnetization of every replica layer."
function recompute_layer_mags!()
    fill!(layer_mags, 0)
    @inbounds for s in 1:L*L, l in 1:NL
        layer_mags[l] += spins[nid(l,s)]
    end
end

"Initialize either one ordered sector, random spins, or independent ordered replica sectors."
function initialize_spins!(mode::String)
    if mode == "plus"
        fill!(spins, Int8(1))
    elseif mode == "random"
        @inbounds for i in eachindex(spins)
            spins[i] = rand(rng) < 0.5 ? Int8(-1) : Int8(1)
        end
    else
        # Draw every replica's global Z2 sign independently.  This starts the
        # chain directly in one of the 2^NL ordered sectors without inserting
        # an extensive domain wall inside a layer.
        @inbounds for l in 1:NL
            sg = rand(rng) < 0.5 ? Int8(-1) : Int8(1)
            for s in 1:L*L
                spins[nid(l,s)] = sg
            end
        end
    end
    recompute_layer_mags!()
end

initialize_spins!(INIT)

# union-find
const parent = collect(1:NN)

"""
    findp(i) -> Int

Return the root of union--find vertex `i`.  Path halving is applied during
the search so later connectivity queries become cheaper.
"""
function findp(i)
    while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    i
end

"""
    unite(i, j)

Join the two union--find components containing vertices `i` and `j`.  Cluster
sizes are not needed, so the implementation links one root directly to the
other without rank bookkeeping.
"""
unite(i,j) = (parent[findp(i)] = findp(j))

"""
    sw_sweep!(lam)

Perform one Swendsen--Wang sweep for the interpolating ensemble at
`lambda = lam`.

For an aligned spin pair with ferromagnetic strength `J`, an FK bond is
activated with probability `1-exp(-2J)`.  Interior edges use their full
coupling.  On a seam, the `D` and `N` lists are sampled as parallel edges
with strengths `(1-lam)J_D` and `lam*J_N`; this is exactly equivalent to the
combined interpolated coupling.  Union--find builds all FK clusters, after
which every cluster is independently multiplied by `+1` or `-1` with equal
probability.  The global `spins` array is updated in place.
"""
function sw_sweep!(lam)
    for i in 1:NN; parent[i] = i; end
    for b in bonds
        trips_N = b.tN; trips_D = b.tD
        if !b.seam
            for (a,c,J) in trips_N
                sa = spins[nid(a,b.i)]; sc = spins[nid(c,b.j)]
                if sa == sc && rand(rng) < 1 - exp(-2J)
                    unite(nid(a,b.i), nid(c,b.j))
                end
            end
        else
            # J_lambda = (1-lam) J_D + lam J_N : activate the two lists scaled
            for (a,c,J) in trips_D
                Jl = (1-lam)*J
                sa = spins[nid(a,b.i)]; sc = spins[nid(c,b.j)]
                if sa == sc && rand(rng) < 1 - exp(-2Jl)
                    unite(nid(a,b.i), nid(c,b.j))
                end
            end
            for (a,c,J) in trips_N
                Jl = lam*J
                sa = spins[nid(a,b.i)]; sc = spins[nid(c,b.j)]
                if sa == sc && rand(rng) < 1 - exp(-2Jl)
                    unite(nid(a,b.i), nid(c,b.j))
                end
            end
        end
    end
    # flip clusters
    flip = Dict{Int,Int8}()
    for i in 1:NN
        r = findp(i)
        f = get!(flip, r) do; rand(rng) < 0.5 ? Int8(-1) : Int8(1); end
        spins[i] *= f
    end
end

# Incident coupling terms used by the local Metropolis update.  A tuple
# `(v, J_D, J_N)` means that the current spin couples to neighbor spin `v`
# with `J_lambda=(1-lambda)J_D+lambda*J_N`.  Repeated graph edges are merged
# so the energy change is evaluated once per distinct neighbor.
const local_adj = let tmp=[Dict{Int,Tuple{Float64,Float64}}() for _ in 1:NN]
    for b in bonds
        merged=Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
        for (a,c,J) in b.tD
            key=(a,c); jd,jn=get(merged,key,(0.0,0.0)); merged[key]=(jd+J,jn)
        end
        for (a,c,J) in b.tN
            key=(a,c); jd,jn=get(merged,key,(0.0,0.0)); merged[key]=(jd,jn+J)
        end
        for ((a,c),(jd,jn)) in merged
            u=nid(a,b.i); v=nid(c,b.j)
            old=get(tmp[u],v,(0.0,0.0)); tmp[u][v]=(old[1]+jd,old[2]+jn)
            old=get(tmp[v],u,(0.0,0.0)); tmp[v][u]=(old[1]+jd,old[2]+jn)
        end
    end
    [[(v,jd,jn) for (v,(jd,jn)) in d] for d in tmp]
end

"""
    local_sweep!(lam)

Perform one checkerboard single-spin Metropolis sweep of `K_lambda`.
Every one of the `NL*L^2` layer spins is attempted exactly once.  For a
proposed flip, `Delta K` is computed from the prebuilt incident-coupling
list and the move is accepted with `min(1,exp(Delta K))`.  Checkerboard
ordering is valid because every interaction still joins nearest-neighbor
spatial sites, even when its two endpoints carry different replica labels.
"""
function local_sweep!(lam)
    for parity in 0:1, y in 0:L-1, x in 0:L-1
        ((x+y)&1)==parity || continue
        s=site(x,y)
        for l in 1:NL
            u=nid(l,s); field=0.0
            @inbounds for (v,jd,jn) in local_adj[u]
                field += ((1-lam)*jd+lam*jn)*spins[v]
            end
            delta=-2.0*spins[u]*field
            if delta>=0.0 || rand(rng)<exp(delta)
                old = spins[u]
                spins[u] = -spins[u]
                layer_mags[l] -= 2*old
            end
        end
    end
end

# Unique cross-replica graph edges.  Flipping a whole replica leaves every
# same-replica spatial bond invariant, so only these O(L) seam edges enter the
# Metropolis ratio of a global sector proposal.
const sector_edges = let per_layer=[Tuple{Int,Int,Float64,Float64}[] for _ in 1:NL]
    for b in bonds
        merged=Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
        for (a,c,J) in b.tD
            key=(a,c); jd,jn=get(merged,key,(0.0,0.0)); merged[key]=(jd+J,jn)
        end
        for (a,c,J) in b.tN
            key=(a,c); jd,jn=get(merged,key,(0.0,0.0)); merged[key]=(jd,jn+J)
        end
        for ((a,c),(jd,jn)) in merged
            a == c && continue
            edge=(nid(a,b.i),nid(c,b.j),jd,jn)
            push!(per_layer[a],edge); push!(per_layer[c],edge)
        end
    end
    per_layer
end
# The same graph edges stored once, for collective subset flips.  A term
# changes sign iff exactly one of its endpoint replica layers is selected.
const sector_unique_edges = let edges=Tuple{Int,Int,Int,Int,Float64,Float64}[]
    for b in bonds
        merged=Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
        for (a,c,J) in b.tD
            key=(a,c); jd,jn=get(merged,key,(0.0,0.0)); merged[key]=(jd+J,jn)
        end
        for (a,c,J) in b.tN
            key=(a,c); jd,jn=get(merged,key,(0.0,0.0)); merged[key]=(jd,jn+J)
        end
        for ((a,c),(jd,jn)) in merged
            a == c && continue
            push!(edges,(a,c,nid(a,b.i),nid(c,b.j),jd,jn))
        end
    end
    edges
end
const sector_attempts = Ref(0)
const sector_accepts = Ref(0)
const update_counter = Ref(0)

"Evaluate K_lambda directly; used only to audit sector-move energy differences."
function total_logweight(lam)
    value=0.0
    @inbounds for b in bonds
        for (a,c,J) in b.tD
            value += (1-lam)*J*spins[nid(a,b.i)]*spins[nid(c,b.j)]
        end
        for (a,c,J) in b.tN
            value += lam*J*spins[nid(a,b.i)]*spins[nid(c,b.j)]
        end
    end
    value
end

"Attempt an exact Metropolis flip of one layer or a collective replica subset."
function sector_flip_attempt!(lam)
    mask = if SECTOR_MODE == "single"
        1 << (rand(rng,1:NL)-1)
    else
        # Exclude the empty and all-layer masks.  The latter is the redundant
        # overall Z2 symmetry and does not change a relative sector.  Applying
        # any retained mask twice is the identity, so this uniform proposal is
        # symmetric and the usual Metropolis acceptance is exact.
        rand(rng, 1:((1 << NL)-2))
    end
    delta = 0.0
    if SECTOR_MODE == "single"
        l = trailing_zeros(mask)+1
        @inbounds for (u,v,jd,jn) in sector_edges[l]
            delta -= 2*((1-lam)*jd + lam*jn)*spins[u]*spins[v]
        end
    else
        @inbounds for (a,c,u,v,jd,jn) in sector_unique_edges
            selected_a = ((mask >> (a-1)) & 1) == 1
            selected_c = ((mask >> (c-1)) & 1) == 1
            selected_a == selected_c && continue
            delta -= 2*((1-lam)*jd + lam*jn)*spins[u]*spins[v]
        end
    end
    sector_attempts[] += 1
    if delta >= 0.0 || rand(rng) < exp(delta)
        before = SECTOR_CHECK ? total_logweight(lam) : 0.0
        @inbounds for l in 1:NL
            ((mask >> (l-1)) & 1) == 1 || continue
            for s in 1:L*L
                u=nid(l,s); spins[u] = -spins[u]
            end
            layer_mags[l] = -layer_mags[l]
        end
        SECTOR_CHECK && @assert isapprox(total_logweight(lam)-before,delta; atol=1e-9,rtol=1e-10)
        sector_accepts[] += 1
        return true
    end
    false
end

"Encode replica magnetization signs relative to layer 1 in `NL-1` bits."
function relative_sector_code()
    base = layer_mags[1] >= 0
    code = 0
    @inbounds for l in 2:NL
        (layer_mags[l] >= 0) != base && (code |= 1 << (l-2))
    end
    code
end

"Perform one sweep with the selected update and optional exact sector move."
function update_sweep!(lam)
    if UPD == "sw"
        sw_sweep!(lam)
    else
        local_sweep!(lam)
        update_counter[] += 1
        SECTOR_MOVES && update_counter[] % SECTOR_EVERY == 0 && sector_flip_attempt!(lam)
    end
end

"""
    dEdlam_prof!(acc) -> Float64

Measure `Delta K = K_N-K_D` on the current spin configuration, resolved by
seam bond.  The contribution of seam bond `k` is added to `acc[k]`, allowing
the caller to accumulate a spatial profile over many sweeps.  Return the sum
over all seam bonds for use as the thermodynamic-integration estimator.
"""
function dEdlam_prof!(acc)
    tot = 0.0
    for (k, bi) in enumerate(seam_idx)
        b = bonds[bi]; e = 0.0
        for (a,c,J) in b.tN
            e += J * spins[nid(a,b.i)] * spins[nid(c,b.j)]
        end
        for (a,c,J) in b.tD
            e -= J * spins[nid(a,b.i)] * spins[nid(c,b.j)]
        end
        acc[k] += e; tot += e
    end
    tot
end

"""
    dEdlam() -> Float64

Measure `Delta K = dK_lambda/dlambda = K_N-K_D` on the current configuration.
Only seam bonds are visited because all interior contributions cancel
identically between the two endpoint ensembles.
"""
function dEdlam()
    e = 0.0
    for b in bonds
        b.seam || continue
        for (a,c,J) in b.tN
            e += J * spins[nid(a,b.i)] * spins[nid(c,b.j)]
        end
        for (a,c,J) in b.tD
            e -= J * spins[nid(a,b.i)] * spins[nid(c,b.j)]
        end
    end
    e
end

# ---- Gauss-Legendre nodes on [0,1], any order (Newton on Legendre roots) ----
"""
    gauss_legendre01(n) -> (x, w)

Compute the `n`-point Gauss--Legendre nodes `x` and weights `w` on `[0,1]`.
Each root of the degree-`n` Legendre polynomial is refined by Newton iteration
from the standard cosine estimate.  The returned nodes are in ascending
order, and the weights include the Jacobian for mapping `[-1,1]` to `[0,1]`.
"""
function gauss_legendre01(n)
    x = zeros(n); w = zeros(n)
    for i in 1:n
        t = cos(pi*(i-0.25)/(n+0.5))
        local dp = 0.0
        for _ in 1:100
            p0, p1 = 1.0, t
            for j in 2:n
                p0, p1 = p1, ((2j-1)*t*p1 - (j-1)*p0)/j
            end
            dp = n*(t*p1 - p0)/(t^2 - 1)
            dt = p1/dp; t -= dt
            abs(dt) < 1e-15 && break
        end
        x[i] = 0.5*(1 - t)          # map [-1,1] -> [0,1], ascending in i
        w[i] = 1.0/((1 - t^2)*dp^2) # includes the 1/2 Jacobian
    end
    (x, w)
end
quad_x, quad_w = if QUAD == "trap"
    @assert NGL >= 2 "trapezoidal quadrature requires at least two lambda points"
    x = collect(range(0.0, 1.0, length=NGL))
    h = 1.0/(NGL-1)
    w = fill(h, NGL); w[1] = h/2; w[end] = h/2
    (x, w)
else
    gauss_legendre01(NGL)
end
@assert abs(sum(quad_w) - 1.0) < 1e-12
NEQ = NEQO > 0 ? NEQO : max(300, NSW ÷ 10)
NBIN = 20
@assert NSW >= NBIN && NSW % NBIN == 0 "NSW must be a positive multiple of NBIN=20"
integ = 0.0; integ_err2 = 0.0
NS = length(seam_idx)
prof_I    = zeros(NS)     # per-bond integrand, GL-weighted sum over nodes
prof_err2 = zeros(NS)
t0 = time()
node_order = REV ? reverse(collect(enumerate(quad_x))) : collect(enumerate(quad_x))
sector_diag = NamedTuple[]
for (qi, lam) in node_order
    if FRESH
        initialize_spins!("random")
    end
    neq_eff = FRESH ? max(NEQ, 4000) : NEQ
    for s in 1:neq_eff; update_sweep!(lam); end
    sector_attempts[] = 0; sector_accepts[] = 0; update_counter[] = 0
    binsz = NSW ÷ NBIN
    binmeans = Float64[]
    acc = 0.0; cnt = 0
    pacc = zeros(NS); pbins = PROF ? zeros(NS, NBIN) : zeros(0,0); bk = 0
    sector_counts = zeros(Int, 1 << (NL-1)); sector_transitions = 0; last_sector = -1
    for s in 1:NSW
        update_sweep!(lam)
        if SECTOR_MOVES
            sc = relative_sector_code()
            sector_counts[sc+1] += 1
            last_sector >= 0 && sc != last_sector && (sector_transitions += 1)
            last_sector = sc
        end
        if PROF
            acc += dEdlam_prof!(pacc)
        else
            acc += dEdlam()
        end
        cnt += 1
        if cnt == binsz
            push!(binmeans, acc/cnt)
            if PROF
                bk += 1; pbins[:, bk] .= pacc ./ cnt; pacc .= 0.0
            end
            acc = 0.0; cnt = 0
        end
    end
    m = mean(binmeans); e = std(binmeans)/sqrt(length(binmeans))
    if PROF
        for kk in 1:NS
            pm = mean(@view pbins[kk, 1:bk]); pe = std(@view pbins[kk, 1:bk])/sqrt(bk)
            prof_I[kk]    += quad_w[qi] * pm
            prof_err2[kk] += (quad_w[qi] * pe)^2
        end
    end
    global integ += quad_w[qi]*m
    global integ_err2 += (quad_w[qi]*e)^2
    if SECTOR_MOVES
        arate = sector_attempts[] > 0 ? sector_accepts[]/sector_attempts[] : 0.0
        nvisited = count(>(0), sector_counts)
        dominant = maximum(sector_counts)/NSW
        push!(sector_diag, (node=qi, lambda=lam, mean=m, error=e,
              attempts=sector_attempts[], accepts=sector_accepts[], acceptance=arate,
              sectors_visited=nvisited, dominant_fraction=dominant,
              transitions=sector_transitions, counts=join(sector_counts, ';')))
        @printf("  lam=%.4f  <dE/dl>=%.6f +- %.6f  secacc=%.4f visited=%d dom=%.3f transitions=%d  (%.0fs)\n",
                lam, m, e, arate, nvisited, dominant, sector_transitions, time()-t0)
    else
        @printf("  lam=%.4f  <dE/dl>=%.6f +- %.6f   (%.0fs)\n", lam, m, e, time()-t0)
    end
    flush(stdout)
end
# The raw four-copy ratio is 2*S_2^(3); report S_2^(3) itself for S3.
observable_scale = MCQ == "S3" ? 0.5 : 1.0
tS = -integ * observable_scale; err = sqrt(integ_err2) * observable_scale; elapsed = time()-t0
@printf("RESULT_MC L=%d beta=%s update=%s tildeS = %.6f +- %.6f   (%.0fs total)\n",
        L, bstr, UPD, tS, err, elapsed)
tag = string(QUAD == "trap" ? "TRAP" : "GL", NGL,
             REV ? "R" : "", FRESH ? "F" : "", MCQ == "tS" ? "" : MCQ,
             (MCQ == "tS" && get(ENV,"MCGAUGE","user") == "user") ? "U" : "",
             UPD=="local" ? "LOCAL" : "", INIT=="sector" ? "SECTORINIT" : "",
             SECTOR_MOVES ? (SECTOR_MODE=="subset" ? "SECTORSUBSET" : "SECTORMOVE") : "")
if UPD == "local"
    # Slurm launches many independent chains concurrently. Give every chain
    # its own file: parallel appends to a shared CSV are not reliably atomic.
    outdir = isempty(MCOUT) ? joinpath(@__DIR__, "..", "..", "data/rk_ising/mc", "audits", "local_update_pilot", "raw") : MCOUT
    mkpath(outdir)
    betakey = replace(bstr, "."=>"p")
    outfile = @sprintf("L%03d_b%s_n%d_eq%d_seed%d.csv", L, betakey, NSW, NEQ, seed)
    open(joinpath(outdir, outfile), "w") do io
        println(io, "L,beta,n_sweeps,n_eq,seed,tildeS,error,runtime_s,n_gl,update,quantity,gauge,tag")
        @printf(io, "%d,%s,%d,%d,%d,%.8f,%.8f,%.0f,%d,%s,%s,%s,%s\n",
                L, bstr, NSW, NEQ, seed, tS, err, elapsed, NGL, UPD, MCQ,
                get(ENV,"MCGAUGE","user"), tag)
    end
else
    outdir = isempty(MCOUT) ? joinpath(@__DIR__, "..", "..", "data/rk_ising/mc") : MCOUT
    mkpath(outdir)
    if isempty(MCOUT)
        open(joinpath(outdir, "mc_results.csv"), "a") do io
            @printf(io, "%d,%s,%d,%d,%.8f,%.8f,%.0f,%s\n",
                    L, bstr, NSW, seed, tS, err, elapsed, tag)
        end
    else
        # Long campaign jobs write independent files so concurrent chains never
        # depend on append atomicity of a shared filesystem object.
        betakey = replace(bstr, "."=>"p")
        outfile = @sprintf("L%03d_b%s_n%d_eq%d_seed%d.csv", L, betakey, NSW, NEQ, seed)
        open(joinpath(outdir, outfile), "w") do io
            println(io, "L,beta,n_sweeps,n_eq,seed,tildeS,error,runtime_s,n_gl,update,quantity,gauge,tag")
            @printf(io, "%d,%s,%d,%d,%d,%.8f,%.8f,%.0f,%d,%s,%s,%s,%s\n",
                    L, bstr, NSW, NEQ, seed, tS, err, elapsed, NGL, UPD, MCQ,
                    get(ENV,"MCGAUGE","user"), tag)
        end
    end
end
if SECTOR_MOVES
    diagdir = isempty(MCSECTOROUT) ?
              (isempty(MCOUT) ? joinpath(@__DIR__, "..", "..", "data/rk_ising/mc", "audits", "local_update_pilot", "diagnostics") : joinpath(dirname(MCOUT), "diagnostics")) :
              MCSECTOROUT
    mkpath(diagdir)
    betakey = replace(bstr, "."=>"p")
    diagfile = @sprintf("L%03d_b%s_n%d_eq%d_seed%d_sectors.csv", L, betakey, NSW, NEQ, seed)
    open(joinpath(diagdir, diagfile), "w") do io
        println(io, "L,beta,quantity,seed,node,lambda,mean_deltaK,error,attempts,accepts,acceptance,sectors_visited,dominant_fraction,transitions,relative_sector_counts")
        for d in sector_diag
            @printf(io, "%d,%s,%s,%d,%d,%.16g,%.10g,%.10g,%d,%d,%.10g,%d,%.10g,%d,\"%s\"\n",
                    L,bstr,MCQ,seed,d.node,d.lambda,d.mean,d.error,d.attempts,d.accepts,
                    d.acceptance,d.sectors_visited,d.dominant_fraction,d.transitions,d.counts)
        end
    end
end
if PROF
    println("PROFILE arm r I_b err   (I_b = per-bond contribution to tildeS; sum = tildeS)")
    for kk in 1:length(seam_idx)
        @printf("PROF %c %3d %+.6f %.6f\n", seam_arm[kk], seam_r[kk], -prof_I[kk], sqrt(prof_err2[kk]))
    end
end
println("DONE")
