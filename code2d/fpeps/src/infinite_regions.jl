# Finite observed regions in an infinite CTMRG-contracted PEPS. The exterior
# is the infinite complement C, never a virtual-vacuum physical window.

"Convert a categorical, supertrace-normalized RDM to occupation coefficients."
function _occupation_rdm(rho)
    n=numout(rho)
    numin(rho)==n && sectortype(rho)==FermionParity ||
        throw(ArgumentError("a square FermionParity density tensor is required"))
    all(i->dim(space(rho,i))==2 && dim(space(rho,i),FermionParity(0))==1 &&
        dim(space(rho,i),FermionParity(1))==1,1:n) ||
        throw(ArgumentError("one spinless physical fermion per site required"))
    matrix=reshape(convert(Array,rho),2^n,2^n)
    # For even operators O, str(rho*O)=tr((rho*P_total)*O).
    # This is a pivotal conversion, not taking absolute eigenvalues.
    parity=Diagonal([isodd(count_ones(i)) ? -1 : 1 for i in 0:2^n-1])
    matrix*parity
end

function _rdm_diagnostics(rho;tolerance=1e-8)
    isfinite(tolerance) && tolerance>0 || throw(ArgumentError("positive finite tolerance required"))
    all(isfinite,rho) || error("nonfinite infinite-system density matrix")
    trace_error=abs(tr(rho)-1)
    hermiticity_error=norm(rho-rho')
    # The Hermitian part is used only for a diagnostic; rho is never replaced.
    minimum_eigenvalue=eigmin(Hermitian((rho+rho')/2))
    trace_error<=tolerance || error("infinite-system density matrix trace failed: $trace_error")
    hermiticity_error<=tolerance || error("infinite-system density matrix is not Hermitian: $hermiticity_error")
    minimum_eigenvalue>=-tolerance || error("infinite-system density matrix is not positive: $minimum_eigenvalue")
    (;trace_error,hermiticity_error,minimum_eigenvalue)
end

"""Physical RDM of observed sites in a converged infinite-plane CTMRG env.

`indices` defines physical Fock order. All other physical sites remain in the
infinite exterior. This is not a finite PEPS with vacuum exterior. Storage of
the explicit matrix is exponential in the number of observed sites; this API
is for independent local checks of an infinite environment, not yet a solver
for three half-infinite regions. Environment chi convergence remains separate.
"""
function infinite_region_rdm(boundary,indices;maxsites=8,tolerance=1e-8)
    boundary.converged || throw(ArgumentError("unconverged infinite CTMRG environment"))
    inds=Tuple(CartesianIndex(Tuple(i)) for i in indices)
    1<=length(inds)<=maxsites || throw(ArgumentError("observed region must have 1:$maxsites sites"))
    length(unique(inds))==length(inds) || throw(ArgumentError("repeated observed site"))
    categorical=PEPSKit.reduced_densitymatrix(inds,boundary.peps,boundary.environment)
    rho=_occupation_rdm(categorical)
    diagnostics=_rdm_diagnostics(rho;tolerance)
    (;rho,categorical,indices=inds,diagnostics,system=:infinite_plane,
      exterior=:converged_CTMRG,chi=boundary.chi,environment_info=boundary.info)
end

"Four replica sectors of rho_AB, with A preceding B in physical Fock order."
function _occupation_rdm_sectors(rho,na::Int;tolerance=1e-8)
    n=trailing_zeros(size(rho,1))
    size(rho)==(2^n,2^n) && 0<na<n || throw(ArgumentError("nonempty A and B and a 2^N square RDM required"))
    _rdm_diagnostics(rho;tolerance)
    da,db=2^na,2^(n-na)
    t=reshape(rho,da,db,da,db)
    ra=[sum(t[i,b,j,b] for b in 1:db) for i in 1:da,j in 1:da]
    rb=[sum(t[a,i,a,j] for a in 1:da) for i in 1:db,j in 1:db]
    realigned=reshape(permutedims(t,(1,3,2,4)),da^2,db^2)
    gram=realigned*realigned'
    sectors=(Z2A=tr(ra*ra),Z2B=tr(rb*rb),Z2C=tr(rho*rho),Z4=tr(gram*gram))
    all(z->abs(imag(z))<=tolerance && real(z)>0,values(sectors)) ||
        error("invalid infinite local occupation-replica sectors: $sectors")
    z=map(real,sectors)
    s2=-log.([z.Z2A,z.Z2B,z.Z2C])
    s3=-log(z.Z4)/2
    (;sectors=z,S2=s2,S3=s3,stilde=2s3-sum(s2))
end

"""S tilde for finite A and B, with C their infinite-plane complement.

The four-copy contraction is Tr[(R(rho_AB) R(rho_AB)^dagger)^2], where R
realigns the A/B occupation indices. This follows by tracing the four C copies
first in the pure-state cube. It is not the half-infinite-quadrant S tilde.
"""
function infinite_local_stilde(boundary,A_indices,B_indices;maxsites=8,tolerance=1e-8)
    isempty(A_indices) || isempty(B_indices) ? throw(ArgumentError("A and B must be nonempty")) : nothing
    rdm=infinite_region_rdm(boundary,(A_indices...,B_indices...);maxsites,tolerance)
    result=_occupation_rdm_sectors(rdm.rho,length(A_indices);tolerance)
    (;result...,rdm,geometry=:finite_AB_infinite_C,chi_converged=false)
end
