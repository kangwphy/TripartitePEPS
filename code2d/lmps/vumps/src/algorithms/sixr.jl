# sixr.jl -- exact six-endpoint replica closure used by the LMPS method.
# =============================================================================
# This file contains only the replica topology.  It does not construct the
# spatial L-shaped boundary.  Its inputs are the six already-converged seam
# fixed points, each a rank-four ITensor with four named chi legs.
#
# Four-replica source permutations:
#
#   sigma_A = id,   sigma_B = (12)(34),   sigma_C = (13)(24).
#
# Hence the six two-cycles are
#
#   AB12, AB34, AC13, AC24, BC14, BC23.
#
# Z4 contracts all six on the octahedron.  An ordinary two-replica sector
# reuses two members of this library for its nontrivial two-cycles; its remaining
# identity cycle is closed by the corresponding rank-two seam metric S.  Thus
# no additional rank-four R is solved for Z2, but a Z2 is NOT a bare trace of
# two rank-four tensors.
# =============================================================================
using ITensors

const SIXR_DESCRIPTORS = (
    AB12=("A", "B", (1,2)), AB34=("A", "B", (3,4)),
    AC13=("A", "C", (1,3)), AC24=("A", "C", (2,4)),
    BC14=("B", "C", (1,4)), BC23=("B", "C", (2,3)),
)

# Full boundary-rail dictionary corresponding to the handwritten six-R
# construction.  A triple `(X,i,j)` denotes the double-layer boundary MPS
#
#                         X^X_{i\bar j}.
#
# This is more information than the octahedron needs for its final closure:
# the octahedron edge owner is only `(X,i)`, because `j=sigma_X(i)` is already
# fixed by the region permutation.  Keeping the full triples here is useful for
# auditing the ket/bra wiring before that information is suppressed.
const SIXR_RAILS = (
    AB12=(("A",1,1), ("A",2,2), ("B",1,2), ("B",2,1)),
    AB34=(("A",3,3), ("A",4,4), ("B",3,4), ("B",4,3)),
    AC13=(("A",1,1), ("A",3,3), ("C",1,3), ("C",3,1)),
    AC24=(("A",2,2), ("A",4,4), ("C",2,4), ("C",4,2)),
    BC14=(("B",1,2), ("B",4,3), ("C",1,3), ("C",4,2)),
    BC23=(("B",2,1), ("B",3,4), ("C",2,4), ("C",3,1)),
)

# Which two of the six four-copy endpoints supply the nontrivial two-cycles of
# each ordinary Renyi-2 sector.  Across the three sectors all six occur once.
const SIXR_O2_ASSIGNMENT = Dict(
    ("O2A", "AB") => :AB12, ("O2A", "AC") => :AC13,
    ("O2B", "AB") => :AB34, ("O2B", "BC") => :BC14,
    ("O2C", "AC") => :AC24, ("O2C", "BC") => :BC23,
)

const SIXR_O2_SECTORS = (
    ("O2A", 2, [2,1], [1,2], [1,2]),
    ("O2B", 2, [1,2], [2,1], [1,2]),
    ("O2C", 2, [1,2], [1,2], [2,1]),
)

"Locate one of the six named source tensors in a `(X,Y,cycle)` vertex list."
function sixr_vertex_index(vertices, name::Symbol)
    X, Y, cyc0 = getproperty(SIXR_DESCRIPTORS, name)
    target = (X, Y, collect(cyc0))
    hits = findall(==(target), vertices)
    length(hits) == 1 ||
        error("six-R descriptor $name has $(length(hits)) matches in vertex list")
    only(hits)
end

"Read the full `(region,ket-copy,bra-copy)` label of a named chi Index."
function sixr_rail_key(ix::Index)
    ts = strip.(split(replace(string(tags(ix)), "\"" => ""), ","))
    rr = findfirst(t -> t in ("A", "B", "C"), ts)
    pp = findfirst(t -> occursin(r"^\d+_\d+b$", t), ts)
    (rr === nothing || pp === nothing) &&
        error("six-R leg has no unambiguous region/copy tags: $ts")
    copies = split(chop(ts[pp]; tail=1), "_")
    length(copies) == 2 || error("cannot parse six-R pair tag $(ts[pp])")
    (ts[rr], parse(Int, copies[1]), parse(Int, copies[2]))
end

"Logical octahedron edge owner `(region,ket-copy)` of a named chi Index."
sixr_edge_key(ix::Index) = sixr_rail_key(ix)[1:2]

"Return the four named chi legs of one endpoint, keyed by `(region,copy)`."
function sixr_legmap(R::ITensor)
    order(R) == 4 || error("six-R endpoint must be rank four, got rank $(order(R))")
    out = Dict{Tuple{String,Int},Index}()
    for ix in inds(R)
        hastags(ix, "Chi") || error("six-R endpoint has a non-Chi open leg: $ix")
        key = sixr_edge_key(ix)
        haskey(out, key) && error("six-R endpoint carries edge $key twice")
        out[key] = ix
    end
    length(out) == 4 || error("six-R endpoint must have four distinct chi legs")
    out
end

"Check that the descriptors and all twelve `(region,copy)` edges are exact."
function sixr_library_audit(Rs, vertices)
    length(Rs) == length(vertices) == 6 ||
        error("six-R library must contain exactly six endpoints")
    for name in propertynames(SIXR_DESCRIPTORS)
        sixr_vertex_index(vertices, name)
    end
    counts = Dict{Tuple{String,Int},Int}()
    for R in Rs, edge in keys(sixr_legmap(R))
        counts[edge] = get(counts, edge, 0) + 1
    end
    expected = Set((region, copy) for region in ("A", "B", "C")
                                  for copy in 1:4)
    Set(keys(counts)) == expected ||
        error("six-R library does not contain exactly edges A1..C4")
    all(==(2), values(counts)) ||
        error("every six-R logical edge must occur on exactly two endpoints")

    # Stronger ket/bra check: every endpoint must carry exactly the four rails
    # written in `SIXR_RAILS`, and every one of the twelve physical rails must
    # occur on exactly two different seams.  This catches a wrong bra pairing
    # that the coarser `(region,ket-copy)` octahedron graph cannot see.
    rail_counts = Dict{Tuple{String,Int,Int},Int}()
    for name in propertynames(SIXR_DESCRIPTORS)
        j = sixr_vertex_index(vertices, name)
        got = Set(sixr_rail_key(ix) for ix in inds(Rs[j]))
        want = Set(getproperty(SIXR_RAILS, name))
        got == want || error("six-R rail mismatch at $name: got $got, want $want")
        for rail in got
            rail_counts[rail] = get(rail_counts, rail, 0) + 1
        end
    end
    length(rail_counts) == 12 ||
        error("six-R library must contain twelve distinct ket/bra rails")
    all(==(2), values(rail_counts)) ||
        error("every ket/bra rail must occur on exactly two seam endpoints")
    (; endpoints=6, edges=12, rails=12, all_pairwise=true,
       ket_bra_pairing=true)
end
