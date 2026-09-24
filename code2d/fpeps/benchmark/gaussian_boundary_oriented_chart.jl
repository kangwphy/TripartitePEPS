# Include after FermionicPEPS / PEPSKit / TensorKit. No production mutation.
function gaussian_oriented_chart(peps,direction)
    original=PEPSKit.InfiniteTransferPEPS(peps,direction,1)
    A=PEPSKit.ket(original[1])
    vertical_flip=!isdual(space(A,2));horizontal_flip=!isdual(space(A,3))
    vertical_flip && (A=PEPSKit.flip_virtualspace(A,(1,3)))
    horizontal_flip && (A=PEPSKit.flip_virtualspace(A,(2,4)))
    canonical=PEPSKit.InfiniteTransferPEPS(PEPSKit.InfinitePEPS(A),1,1)
    fused=PEPSKit.mpotensor((A,twist(A,(2,4))))
    (;original,canonical,fused,vertical_flip,horizontal_flip)
end
function gaussian_flip_boundary(T,chart;inverse=false)
    chart.vertical_flip || return T
    # Native MPS bra metrics change when both physical arrows reverse.
    # The ket-leg parity is fixed by random-tensor AC/row-action identities.
    # Twisting the bra leg instead is the equivalent virtual-parity gauge.
    inverse ? flip(twist(T,2),(2,3);inv=true) : twist(flip(T,(2,3)),2)
end
