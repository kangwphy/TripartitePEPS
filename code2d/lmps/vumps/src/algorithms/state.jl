# state.jl -- one PEPS tensor, its ket and bra layers, and the
# double layer that a region permutation selects.
# =============================================================================
# A PEPS site tensor is A[s; u,l,d,r]: one physical leg of dimension d, four
# virtual legs of dimension D.  A genuine quantum state needs BOTH layers, so
# every copy appears twice, and which bra faces which ket is the only thing the
# region permutation decides.  Nothing here knows about regions or seams.
# =============================================================================
include(joinpath(@__DIR__, "indices.jl"))

"""
A single-site PEPS, stored as the raw array plus the dimensions.  Sites are
translation invariant, so one array is the whole state; the ITensor with its
tagged indices is built on demand for a given layer and position.
"""
struct PEPS
    A::Array{ComplexF64,5}   # [s, u, l, d, r]
    d::Int                   # physical dimension
    D::Int                   # virtual bond dimension
end

function PEPS(A::AbstractArray)
    ndims(A) == 5 || throw(ArgumentError("PEPS tensor must have five legs [s,u,l,d,r]"))
    size(A, 2) == size(A, 3) == size(A, 4) == size(A, 5) ||
        error("PEPS: the four virtual legs must share a dimension")
    PEPS(ComplexF64.(A), size(A, 1), size(A, 2))
end

"""
    layer(P, lay, x, y)

The site tensor of one sheet, with every index tagged by `lay` (a layer name
such as "k1" or "b2") and by the site.  The bra sheet is the elementwise
conjugate; taking `dag` as well would flip the arrows, which we do not want
because these indices are plain (non-QN) and we contract them by tag.
"""
function layer(P::PEPS, lay::AbstractString, x::Int, y::Int)
    valid_layer(lay) || throw(ArgumentError("invalid layer tag $lay"))
    s  = physind(P.d, lay, x, y)
    vs = [virtind(P.D, lay, dir, x, y) for dir in DIRS]
    data = startswith(lay, "b") ? conj(P.A) : P.A
    ITensor(data, s, vs...)
end

"""
    double_layer(P, c, d, x, y)

The double-layer site tensor E_{c, dbar}: ket copy `c` with bra copy `d`, their
PHYSICAL legs contracted with each other.  This is the object the note calls a
ket--bra pair; it is what a boundary MPS is built from, and it is the only place
the permutation enters (through the choice of `d = sigma(c)`).

The result keeps eight virtual legs -- four from the ket, four from the bra --
each still carrying its own layer tag, so the caller can fuse a direction's pair
into one leg of dimension D^2 or keep them apart, as it prefers.
"""
function double_layer(P::PEPS, c::Int, d::Int, x::Int, y::Int)
    K = layer(P, ket(c), x, y)
    B = layer(P, bra(d), x, y)
    # contract the two physical legs: same dimension, different tags, so the
    # pairing is explicit rather than positional.
    sk = theind(K, "Phys"); sb = theind(B, "Phys")
    K * delta(sk, sb) * B
end

"""
    fuse_virtual(E, dir, c, d)

Fuse the ket and bra virtual legs of `E` in direction `dir` into a single leg of
dimension D^2, tagged as the double-layer bond of the pairing (c, dbar).
Returns the new tensor and the combiner (needed to undo the fusion later).
"""
function fuse_virtual(E::ITensor, dir::Symbol, c::Int, d::Int)
    dir in DIRS || throw(ArgumentError("direction must be one of $DIRS, got $dir"))
    ik = theind(E, "Virt", ket(c), String(dir))
    ib = theind(E, "Virt", bra(d), String(dir))
    C  = combiner(ik, ib; tags = "Dbl,$(pairtag(c,d)),$dir")
    E * C, C
end
