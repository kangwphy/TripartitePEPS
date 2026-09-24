"""Four directed rails in the local LMPS retained frame.

The metric inverse uses the audited oriented cap G, never L*R. No retained
sector is silently discarded. hR/vL have dual chi spaces after bending; this
is what makes the two endpoints of every regional junction edge compatible.
"""
function fermionic_rails(objects;inverse_rtol=1e-12)
    inverse_rtol>0 || throw(ArgumentError("positive inverse threshold required"))
    singular=reduce(vcat,(svdvals(block) for (_,block) in blocks(objects.G)))
    isempty(singular) && error("empty LMPS metric")
    minimum(singular)>inverse_rtol*maximum(singular) ||
        error("LMPS metric is rank deficient at the requested tolerance; retained support must be resolved")
    Gi=inv(objects.G)
    hL=objects.AL
    hR=permute(hL,((4,2,3),(1,)))
    vL=permute(objects.L,((2,3,4),(1,)))
    vR=objects.R*Gi
    (;hL,hR,vL,vR,metric_condition=maximum(singular)/minimum(singular),
      inverse_residual=norm(objects.G*Gi-id(space(objects.G,1))),
      direction=objects.direction)
end

"Directed regional rails for the legacy A/B bends and straight C geometry."
function regional_seam_rails(regional)
    result=(AB=(regional.A.vL,regional.B.vR),
            AC=(regional.A.hL,regional.C.hL),
            BC=(regional.B.hR,regional.C.hR))
    for (name,(X,Y)) in pairs(result)
        all(i->space(X,i)==dual(space(Y,i)),(2,3)) ||
            throw(DimensionMismatch("physical ports do not match on seam $name"))
    end
    for (first,second) in ((result.AB[1],result.AC[1]),
                          (result.AB[2],result.BC[1]),
                          (result.AC[2],result.BC[2]))
        space(first,1)==dual(space(second,1)) ||
            throw(DimensionMismatch("regional junction chi ports must be dual"))
    end
    result
end
