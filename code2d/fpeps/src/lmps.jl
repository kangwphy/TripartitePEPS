"""Oriented C-channel cap; GL/GR include PEPSKit's dual-leg twists.

This is not the ordinary product of flattened left/right endpoint matrices.
"""
function _lmps_metric(GL,GR,C)
    @tensor G[a;b] := GL[a k bra;l]*C[l;r]*GR[r k bra;b]
    G
end

"Grow a PEPS row into the boundary MPS, retaining all directed virtual legs."
function _grow_boundary(M,A,B)
    @tensor D[left wk wb;sk sb right ek eb] :=
        M[left nk nb;right]*A[p;nk ek sk wk]*conj(B[p;nb eb sb wb])
    D
end

"""Typed local LMPS objects of the single-copy fermionic environment.

G is the explicit oriented cap, NOT L*R, which can vanish through parity
cancellation. Check B ≈ kappa*AL*G, PEPSKit's independent C/AC actions and
center placement. Keep the stable AL rail without inverting small Schmidt
values. These objects do not yet certify a multicopy seam or six-R closure.
"""
function lmps_objects(boundary;tolerance=max(100boundary.galerkin,1e-7))
    length(boundary.state)==length(boundary.transfer)==1 ||
        throw(ArgumentError("one-site LMPS objects require a one-site boundary; use every site of a periodic cell"))
    boundary.converged || throw(ArgumentError("LMPS requires a converged boundary"))
    tolerance>0 || throw(ArgumentError("positive tolerance required"))
    state=boundary.state; env=boundary.environments; transfer=boundary.transfer
    AL,AC,C=state.AL[1],state.AC[1],state.C[1]
    GL=PEPSKit.twistdual(env.GLs[1],1)
    GR=PEPSKit.twistdual(env.GRs[1],numind(env.GRs[1]))
    A,B=PEPSKit.ket(transfer[1]),PEPSKit.bra(transfer[1])
    L=permute(GL,((1,),(4,2,3)))
    @tensor R[r k bra;a] := C[r;x]*GR[x k bra;a]
    D=_grow_boundary(AL,A,B)
    @tensor block[a sk sb;b] :=
        L[a;left wk wb]*D[left wk wb;sk sb right ek eb]*R[right ek eb;b]
    G=_lmps_metric(GL,GR,C)
    Dc=_grow_boundary(AC,A,B)
    @tensor center_block[a sk sb;b] :=
        L[a;left wk wb]*Dc[left wk wb;sk sb right ek eb]*GR[right ek eb;b]
    hc=MPSKit.C_hamiltonian(1,state,transfer,state,env)
    ha=MPSKit.AC_hamiltonian(1,state,transfer,state,env)
    gp,bp=hc(C),ha(AC)
    relative(x,y)=norm(x-y)/max(norm(x),norm(y),eps())
    package_metric_residual=relative(G,gp)
    package_block_residual=relative(block,bp)
    center_placement_residual=relative(block,center_block)
    target=AL*G
    real(dot(target,target))>eps() || error("LMPS metric has negligible support")
    kappa=dot(target,block)/dot(target,target)
    generalized_residual=norm(block-kappa*target)/max(norm(block),eps())
    residuals=(;package_metric_residual,package_block_residual,
                 center_placement_residual,generalized_residual)
    maximum(values(residuals))<=tolerance || error("fermionic LMPS local audit failed: $residuals")
    (;L,R,GL,GR,C,D,G,B=block,AL,AC,kappa,residuals,direction=boundary.direction,
      closure=:oriented_fermionic_cap)
end
