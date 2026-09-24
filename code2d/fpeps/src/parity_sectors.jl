"""Exact Walsh expansion of the occupation/graded replica correction.

Ordering is replica-major, with regional blocks A,B,C in each replica.
Every source copy is even, so pC = pA ⊻ pB. The coefficients are computed
from the actual block permutation, never fitted to a contraction result.
Each term inserts regional physical parity on the SOURCE ket, followed by
the graded replica permutation. For the four-copy cube there are 16 terms
of weight ±1/4. This algebraic identity does not assume seam factorization.
"""
function replica_parity_expansion(permutations)
    length(permutations)==3 || throw(ArgumentError("three regional permutations required"))
    k=length(first(permutations))
    k in (1,2,4) || throw(ArgumentError("supported copy counts are 1,2,4"))
    all(p->sort(collect(p))==collect(1:k),permutations) || throw(ArgumentError("invalid replica permutation"))
    order=[3*(permutations[x][r]-1)+x for r in 1:k for x in 1:3]
    crossings=[(order[i],order[j]) for i in eachindex(order) for j in i+1:length(order)
               if order[i]>order[j]]
    coefficients=Vector{Int}(undef,1<<(2k))
    for mask in 0:length(coefficients)-1
        parity=Bool[]
        for r in 1:k
            a=!iszero(mask & (1<<(2r-2))); b=!iszero(mask & (1<<(2r-1)))
            append!(parity,(a,b,xor(a,b)))
        end
        exponent=sum(Int(parity[i]&parity[j]) for (i,j) in crossings;init=0)
        coefficients[mask+1]=isodd(exponent) ? -1 : 1
    end
    # Unnormalized fast Walsh-Hadamard transform, entirely in integer arithmetic.
    stride=1
    while stride<length(coefficients)
        for start in 1:2stride:length(coefficients),offset in 0:stride-1
            i=start+offset; j=i+stride
            a,b=coefficients[i],coefficients[j]
            coefficients[i]=a+b; coefficients[j]=a-b
        end
        stride*=2
    end
    terms=NamedTuple[]
    for mask in 0:length(coefficients)-1
        coefficient=coefficients[mask+1]
        iszero(coefficient) && continue
        insertions=Tuple(x<3 && !iszero(mask & (1<<(2(r-1)+x-1)))
                         for r in 1:k for x in 1:3)
        push!(terms,(;weight=coefficient/length(coefficients),insertions))
    end
    (;terms,order=Tuple(order),crossings,copies=k,source_parity=:even)
end

"""Finite typed overlap from graded sewing and a sum of regional parity strings.

Independent of occupation_permute: only TensorKit twists and graded permute
are used after the integer parity compiler. This is a reference implementation
for validating the sector expansion before pushing strings onto PEPS boundaries.
"""
function tensor_replica_parity_overlap(psi,regions,permutations;maxsites=4)
    N=length(regions)
    N<=maxsites || throw(ArgumentError("finite replica oracle exceeds maxsites"))
    numout(psi)==N && numin(psi)==0 || throw(DimensionMismatch("even physical state tensor required"))
    sectortype(psi)==FermionParity || throw(ArgumentError("FermionParity required"))
    all(X->X in (:A,:B,:C),regions) || throw(ArgumentError("regions must be A/B/C"))
    expansion=replica_parity_expansion(permutations)
    ordering=sortperm(regions;by=X->findfirst(==(X),(:A,:B,:C)))
    v=permute(psi,(Tuple(ordering),()))
    regs=regions[ordering]
    product=foldl((a,_)->a⊗v,2:expansion.copies;init=v)
    order=_replica_order(regs,permutations)
    value=0.0im
    for term in expansion.terms
        axes=Int[]
        for r in 1:expansion.copies,i in 1:N
            x=findfirst(==(regs[i]),(:A,:B,:C))
            term.insertions[3(r-1)+x] && push!(axes,N*(r-1)+i)
        end
        inserted=twist!(copy(product),axes)
        value+=term.weight*dot(product,permute(inserted,(order,())))
    end
    value
end
