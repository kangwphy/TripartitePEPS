"""Ordinary coefficient permutation on an all-outgoing fermionic tensor.

TensorKit.permute performs the graded wire permutation. This function also
applies the diagonal parity correction required by the OCCUPATION replica
observable. It must not be used for physical Fock reordering, which is graded.
All elementary output legs must have FermionParity sectors.
"""
function occupation_permute(t, order::Tuple)
    numin(t)==0 || throw(ArgumentError("occupation_permute expects an all-outgoing tensor"))
    length(order)==numout(t) && sort(collect(order))==collect(1:numout(t)) ||
        throw(ArgumentError("invalid output permutation"))
    sectortype(t)==FermionParity || throw(ArgumentError("FermionParity spaces required"))
    result = permute(t,(order,()))
    for (fo,fi) in fusiontrees(result)
        # Output wire i carried source wire order[i]. Undo its Koszul sign.
        p = getproperty.(fo.uncoupled,:isodd)
        exponent = sum(Int(p[i]&p[j]) for i in 1:length(order) for j in i+1:length(order)
                       if order[i]>order[j];init=0)
        isodd(exponent) && (result[fo,fi] .*= -1)
    end
    result
end

"Return the permutation of elementary physical legs for a regional sewing."
function _replica_order(regions,permutations)
    N=length(regions)
    k=length(first(permutations))
    Tuple((permutations[findfirst(==(regions[i]),(:A,:B,:C))][r]-1)*N+i
          for r in 1:k for i in 1:N)
end

"""Exact typed replica overlap for a finite physical state tensor.

The source is grouped into regional Fock order using a graded permutation.
The subsequent sewing uses ordinary occupation permutations. Only small
systems are allowed because the full replica product is a reference oracle.
"""
function tensor_replica_sectors(psi,regions;maxsites=4)
    N=length(regions)
    N<=maxsites || throw(ArgumentError("typed replica oracle exceeds maxsites=$maxsites"))
    numout(psi)==N && numin(psi)==0 || throw(DimensionMismatch("physical state tensor"))
    all(X->X in (:A,:B,:C),regions) || throw(ArgumentError("regions must be A/B/C"))
    ordering=sortperm(regions;by=X->findfirst(==(X),(:A,:B,:C)))
    v=permute(psi,(Tuple(ordering),()))
    regs=regions[ordering]
    Z1=real(dot(v,v))
    v2=v⊗v
    id2=(1,2); swap2=(2,1)
    z2=map(1:3) do x
        permutations=ntuple(j->j==x ? swap2 : id2,3)
        order=_replica_order(regs,permutations)
        dot(v2,occupation_permute(v2,order))
    end
    v4=v2⊗v2
    order=_replica_order(regs,((1,2,3,4),(2,1,4,3),(3,4,1,2)))
    Z4=dot(v4,occupation_permute(v4,order))
    ratio=Z4*Z1^2/prod(z2)
    abs(imag(ratio))<=1e-11*max(abs(ratio),eps()) || error("typed cube has nonreal phase")
    real(ratio)>0 || error("typed cube has nonpositive ratio")
    (;Z1,Z2A=z2[1],Z2B=z2[2],Z2C=z2[3],Z4,stilde=-log(real(ratio)),order=ordering)
end
