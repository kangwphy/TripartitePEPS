"""Exact square-lattice nearest-neighbour RVB PEPS (virtual bond D=3).

This file is deliberately independent of the TFIM model code.  The physical
index is `s=1,2`; a local tensor is one precisely when three virtual indices
are the singlet/vacuum label `3` and the remaining virtual index is `s`.
The four virtual legs are ordered `(north,east,south,west)`.
"""
module RVBD3Model

using TensorKit
using PEPSKit

export RVBD3, rvb_d3_peps, rvb_d3_dense_tensor

struct RVBD3{P}
    peps::P
end

"Dense local tensor in the PEPSKit order `(physical,north,east,south,west)`.

The paper writes the D=3 tensor with an oriented virtual singlet on every
nearest-neighbour bond.  PEPSKit's standard square-lattice contraction uses
the identity pairing on a virtual leg.  We therefore put the antisymmetric
singlet matrix on the north/east legs of every site.  Every vertical or
horizontal bond then contains exactly one such matrix, so the resulting
one-site tensor is gauge-equivalent to the oriented two-sublattice tensor in
the paper while remaining compatible with the one-site VUMPS solver.
" 
function rvb_d3_dense_tensor(; scalar_type::Type{<:Number}=ComplexF64)
    raw = zeros(scalar_type, 2, 3, 3, 3, 3)
    # Physical labels and virtual non-vacuum labels are both 1,2.  The
    # remaining three virtual legs are the vacuum/singlet label 3.
    for s in 1:2
        for active in 1:4
            inds = fill(3, 4)
            inds[active] = s
            raw[s, inds...] = one(scalar_type)
        end
    end
    # E implements |0,1>-|1,0> on the active subspace and leaves the vacuum
    # label 3 unchanged.  Its placement on north/east gives one E per bond.
    E = zeros(scalar_type, 3, 3)
    E[1, 2] = one(scalar_type)
    E[2, 1] = -one(scalar_type)
    E[3, 3] = one(scalar_type)
    A = zeros(scalar_type, 2, 3, 3, 3, 3)
    for s in 1:2, n in 1:3, e in 1:3, so in 1:3, w in 1:3
        value = zero(scalar_type)
        for n0 in 1:3, e0 in 1:3
            value += raw[s, n0, e0, so, w] * E[n0, n] * E[e0, e]
        end
        A[s, n, e, so, w] = value
    end
    A
end

"Construct the exact one-site RVB PEPS as a PEPSKit InfinitePEPS." 
function rvb_d3_peps(; scalar_type::Type{<:Number}=ComplexF64)
    A = rvb_d3_dense_tensor(; scalar_type)
    physical = ComplexSpace(2)
    ne = ComplexSpace(3)
    sw = dual(ne)
    tensor = TensorMap(A, physical ← ne ⊗ ne ⊗ sw ⊗ sw)
    PEPSKit.InfinitePEPS(tensor)
end

end # module RVBD3Model
