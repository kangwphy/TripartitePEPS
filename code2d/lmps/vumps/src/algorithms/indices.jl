# lmps_index.jl -- index and tag conventions for the boundary-MPS multientropy.
# =============================================================================
# Everything downstream contracts by TAG, never by position, so a wrong pairing
# is a caught error rather than a silently wrong number.  ITensor allows four
# tags of at most sixteen characters each, which fixes the scheme below.
#
#   physical leg of a layer      Phys , <layer> , <site>
#   virtual leg of a layer       Virt , <layer> , <dir> , <site>
#   boundary-MPS bond            Chi  , <region> , <pair> , <cut>
#
# A LAYER is one sheet of the network: ket copy c is "k<c>", bra copy c is
# "b<c>" (read as c-bar).  A PAIR is the ket/bra pairing a boundary MPS carries,
# written "<c>_<d>b" for (ket c, bra d-bar), so the note's X^A_{1 1bar} appears
# in the tags as  Chi,A,1_1b .
# =============================================================================
using ITensors

const DIRS = (:U, :L, :D, :R)

function _copy_number(c::Int)
    c > 0 || throw(ArgumentError("replica number must be positive, got $c"))
    c
end
ket(c::Int) = "k$(_copy_number(c))"       # ket layer of copy c
bra(c::Int) = "b$(_copy_number(c))"       # bra layer of copy c (c-bar)
pairtag(c::Int, d::Int) = "$(_copy_number(c))_$(_copy_number(d))b"
sitetag(x::Int, y::Int) = "s$(x)_$(y)"

"Whether a layer tag has the supported form k1, k2, ... or b1, b2, ...."
valid_layer(layer::AbstractString) = occursin(r"^[kb][1-9][0-9]*$", layer)

"A string that remains one unambiguous ITensor tag."
valid_tag_atom(s::AbstractString) = !isempty(s) && !occursin(',', s) && ncodeunits(s) <= 16

"Physical index of one layer at one site."
physind(d::Int, layer::AbstractString, x::Int, y::Int) =
    valid_layer(layer) ? Index(d, "Phys,$layer,$(sitetag(x,y))") :
    throw(ArgumentError("invalid layer tag $layer"))

"Virtual index of one layer at one site, pointing in `dir`."
function virtind(D::Int, layer::AbstractString, dir::Symbol, x::Int, y::Int)
    valid_layer(layer) || throw(ArgumentError("invalid layer tag $layer"))
    dir in DIRS || throw(ArgumentError("direction must be one of $DIRS, got $dir"))
    Index(D, "Virt,$layer,$dir,$(sitetag(x,y))")
end

"Boundary-MPS bond of `region`, pairing (c, d-bar), at position `n` along the cut."
function chiind(chi::Int, region::AbstractString, c::Int, d::Int, n::Int)
    chi > 0 || throw(ArgumentError("chi must be positive, got $chi"))
    n > 0 || throw(ArgumentError("cut position must be positive, got $n"))
    valid_tag_atom(region) || throw(ArgumentError("invalid region tag $region"))
    Index(chi, "Chi,$region,$(pairtag(c,d)),n$n")
end

"Every index of `T` whose tagset contains all of `tags`."
function findinds_all(T::ITensor, tags::AbstractString...)
    filter(i -> all(t -> hastags(i, t), tags), inds(T))
end

"Exactly one index matching `tags`; error if zero or many.  Used everywhere a
 contraction must be unambiguous."
function theind(T::ITensor, tags::AbstractString...)
    m = findinds_all(T, tags...)
    length(m) == 1 ||
        error("expected exactly one index with tags $(tags), got $(length(m)): $m")
    m[1]
end

"The unique direction tag carried by an index."
function direction_tag(i::Index)
    found = [String(d) for d in DIRS if hastags(i, String(d))]
    length(found) == 1 ||
        error("expected one direction tag on index $i, got $found")
    only(found)
end

"""
    say(msg)

`println` that actually reaches the log file.  Julia block-buffers stdout when
it is redirected, so a long run prints nothing until the process exits and looks
hung.  Every progress line in this directory goes through here.
"""
say(msg) = (println(msg); flush(stdout))
