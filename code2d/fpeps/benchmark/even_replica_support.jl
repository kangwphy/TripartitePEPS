# Algebraic diagnostic: use BOTH ket-copy and bra-copy even parity. This
# condition is automatic for a finite pure even state, but must be checked
# before applying it to an approximate double-layer boundary closure.
function even_replica_character(permutations)
    expansion=replica_parity_expansion(permutations)
    k=expansion.copies
    allowed=Tuple{Int,Bool}[]
    for mask in 0:(1<<(2k))-1
        p=Bool[]
        for r in 1:k
            a=isodd(mask>>(2r-2)); b=isodd(mask>>(2r-1))
            append!(p,(a,b,xor(a,b)))
        end
        all(r->!isodd(sum(p[expansion.order[3(r-1)+x]] for x in 1:3)),1:k) || continue
        sign=isodd(sum(p[i]&p[j] for (i,j) in expansion.crossings;init=0))
        push!(allowed,(mask,sign))
    end
    matches=[mask for mask in 0:(1<<(2k))-1 if
        all(isodd(count_ones(mask&m))==sign for (m,sign) in allowed)]
    isempty(matches) && error("parity correction is not linear on even ket/bra support")
    sort!(matches;by=m->(count_ones(m),m))
    chosen=first(matches)
    insertions=Tuple(x<3 && isodd(chosen>>(2(r-1)+x-1)) for r in 1:k for x in 1:3)
    (;insertions,mask=chosen,matching_masks=matches,allowed_count=length(allowed),copies=k)
end

function insertion_mask(insertions)
    k=length(insertions)÷3
    all(!insertions[3r] for r in 1:k) || error("expected the pC=pA xor pB character chart")
    sum(Int(insertions[3(r-1)+x])<<(2(r-1)+x-1) for r in 1:k for x in 1:2)
end
