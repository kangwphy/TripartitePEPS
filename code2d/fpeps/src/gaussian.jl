"Pivoted antisymmetric elimination, retaining the complex Pfaffian phase."
function _pfaffian(matrix::AbstractMatrix)
    n,m = size(matrix)
    n == m && iseven(n) || throw(DimensionMismatch("Pfaffian requires an even square matrix"))
    A = ComplexF64.(matrix)
    norm(A+transpose(A)) <= 1e-11*max(norm(A),1) || throw(ArgumentError("matrix is not antisymmetric"))
    value = 1.0+0im
    for k in 1:2:n-1
        j = k+argmax(abs.(A[k,k+1:n]))
        iszero(A[k,j]) && return 0.0im
        if j != k+1
            A[:,[k+1,j]] = A[:,[j,k+1]]
            A[[k+1,j],:] = A[[j,k+1],:]
            value = -value
        end
        pivot = A[k,k+1]
        value *= pivot
        for i in k+2:n, j2 in i+1:n
            A[i,j2] += (A[k+1,i]*A[k,j2]-A[k,i]*A[k+1,j2])/pivot
            A[j2,i] = -A[i,j2]
        end
    end
    value
end

"""Independent BdG ground-state reference for the explicit parent Hamiltonian.

`:projector` is the native Q convention, `:paper` its uniform particle-hole
transform, and `:gaussian` the additional i^N gauge and energy scale used in
Gaussian/code/cirac2_model.py. Zero modes are reported and the covariance is
left `nothing`: selecting their occupation requires additional boundary data.
"""
function parent_gaussian_reference(nx::Int,ny::Int;periodic=true,twists=(1,1),
                                    convention=:projector,zero_tolerance=1e-10)
    nx>0 && ny>0 || throw(ArgumentError("positive lattice dimensions required"))
    periodic && min(nx,ny)<3 && throw(ArgumentError("periodic reference needs dimensions >= 3"))
    all(x->x in (-1,1),twists) || throw(ArgumentError("twists must be ±1"))
    hopping,pairing = convention==:projector ? (-1,1) :
        convention==:paper ? (1,1) : convention==:gaussian ? (0.5,-0.5) :
        throw(ArgumentError("convention must be :projector, :paper, or :gaussian"))
    N=nx*ny
    A=zeros(ComplexF64,N,N); B=similar(A); fill!(B,0)
    site(x,y)=mod1(x,nx)+nx*(mod1(y,ny)-1)
    for y in 1:ny,x in 1:nx
        i=site(x,y)
        for (dx,dy) in ((1,1),(1,-1))
            xx,yy=x+dx,y+dy
            !periodic && !(1<=xx<=nx && 1<=yy<=ny) && continue
            phase=(1<=xx<=nx ? 1 : twists[1])*(1<=yy<=ny ? 1 : twists[2])
            j=site(xx,yy)
            A[i,j]+=-hopping*phase; A[j,i]+=-hopping*phase
        end
        for (dx,dy,coefficient) in ((0,1,2im),(1,0,-2im))
            xx,yy=x+dx,y+dy
            !periodic && !(1<=xx<=nx && 1<=yy<=ny) && continue
            phase=(1<=xx<=nx ? 1 : twists[1])*(1<=yy<=ny ? 1 : twists[2])
            j=site(xx,yy)
            B[i,j]+=pairing*phase*coefficient
            B[j,i]-=pairing*phase*coefficient
        end
    end
    nambu=[A B; -conj(B) -conj(A)]
    omega=zeros(ComplexF64,2N,2N)
    for j in 1:N
        omega[2j-1,j]=omega[2j-1,N+j]=1
        omega[2j,j]=im; omega[2j,N+j]=-im
    end
    K=omega*nambu*omega'/4
    h=real.(-im*(K-transpose(K)))
    spectrum=eigen(Hermitian(im*h))
    zero_modes=count(x->abs(x)<=zero_tolerance,spectrum.values)÷2
    gamma=zero_modes==0 ? real.(im*spectrum.vectors*Diagonal(sign.(spectrum.values))*spectrum.vectors') : nothing
    energy=real(tr(A))/2-sum(abs,spectrum.values)/4
    (;A,B,h,gamma,energy,zero_modes,gap=minimum(abs,spectrum.values),convention)
end

"""Infinite-state density pair by midpoint Brillouin-zone quadrature and Wick.

For the native Q parent, ε(k)=4 cos(kx)cos(ky), Δ(k)=4(sin(kx)-sin(ky)).
The midpoint grid avoids the critical node for grid divisible by four.
This is a numerical integral; compare successive grids to assess its error.
"""
function ksvc_density_reference(dx::Int,dy::Int;grid::Int=256)
    grid>=4 && grid%4==0 || throw(ArgumentError("grid must be positive and divisible by four"))
    normal=0.0im; anomalous=0.0im; filling=0.0
    for j in 0:grid-1,i in 0:grid-1
        kx,ky=2pi*(i+0.5)/grid,2pi*(j+0.5)/grid
        epsilon=4cos(kx)*cos(ky)
        delta=4(sin(kx)-sin(ky))
        energy=hypot(epsilon,delta)
        nk=(1-epsilon/energy)/2
        phase=cis(kx*dx+ky*dy)
        normal+=nk*phase/grid^2
        anomalous+=delta/(2energy)*phase/grid^2
        filling+=nk/grid^2
    end
    nn=(dx,dy)==(0,0) ? filling : filling^2-abs2(normal)+abs2(anomalous)
    (;filling,nn,normal,anomalous,grid)
end

"""Exact Fourier-mode norm of the periodic Gaussian Q projection.

W=I+K B is the virtual Gaussian overlap matrix; the physical BCS amplitude
is f B W^-1 f^T. A bordered determinant computes its numerator without
inverting W, including modes where the physical vacuum amplitude vanishes.
The per-mode norm factor is hypot(abs(det(W)),abs(numerator)).
Use even sizes and twists=(-1,-1) as a midpoint quadrature of the infinite
norm. A vanishing factor is a zero projected torus sector, not a normalized
state or an arbitrary selection of a zero-mode occupation.
"""
function ksvc_norm_reference(nx::Int,ny::Int;twists=(1,1))
    min(nx,ny)>=3 || throw(ArgumentError("torus dimensions must be >= 3"))
    all(t->t in (-1,1),twists) || throw(ArgumentError("twists must be ±1"))
    # A k=pi virtual dark mode kills the entire Q projection exactly. For
    # APBC dimensions 2 mod 4 in both directions, the physical node also
    # occurs on the grid. Do not turn these exact zeros into roundoff states.
    virtual_zero=any(iseven(n)==(t==1) for (n,t) in zip((nx,ny),twists))
    physical_zero=all(==(-1),twists) && nx%4==ny%4==2
    if virtual_zero || physical_zero
        return (;lognorm=-Inf,lognorm_per_site=-Inf,lambda=0.0,minimum_factor=0.0,twists)
    end
    lognorm=0.0; minimum_factor=Inf
    for y in 0:ny-1,x in 0:nx-1
        kx=2pi*(x+(twists[1]==-1 ? 0.5 : 0))/nx
        ky=2pi*(y+(twists[2]==-1 ? 0.5 : 0))/ny
        factor=_ksvc_mode_norm(kx,ky)
        minimum_factor=min(minimum_factor,factor)
        lognorm+=log(factor)
    end
    (;lognorm,lognorm_per_site=lognorm/(nx*ny),lambda=exp(lognorm/(nx*ny)),minimum_factor,twists)
end

"Analytic simplification of the bordered-determinant Gaussian mode norm."
function _ksvc_mode_norm(kx,ky)
    q=hypot(cos(kx)*cos(ky),sin(kx)-sin(ky)) # = 1-sin(kx)sin(ky)
    16abs(cos(kx/2)*cos(ky/2))*sqrt(q)
end

"""Infinite Q norm per site, removing the known virtual-chain finite-size factor.

The integral is log(lambda)=log(4)+1/2 ∫ log(1-sin(kx)sin(ky)) d²k/(2pi)².
The two log|cos(k/2)| factors are integrated analytically, avoiding their
O(1/grid) torus normalization correction. Remaining quadrature is numerical.
"""
function ksvc_infinite_norm_reference(;grid=256)
    grid>=4 && grid%4==0 || throw(ArgumentError("grid must be positive and divisible by four"))
    integral=0.0
    for j in 0:grid-1,i in 0:grid-1
        kx,ky=2pi*(i+0.5)/grid,2pi*(j+0.5)/grid
        q=hypot(cos(kx)*cos(ky),sin(kx)-sin(ky))
        integral+=log(q)/grid^2
    end
    loglambda=log(4)+integral/2
    (;lambda=exp(loglambda),loglambda,grid)
end

function _cube_matrix(G,regions)
    n = length(regions)
    glue = Dict(X=>zeros(ComplexF64,4,4) for X in (:A,:B,:C))
    for (X,pairs) in ((:A,((1,3,1),(2,4,1))),
                      (:B,((1,2,1),(3,4,-1))),
                      (:C,((1,4,1),(2,3,-1))))
        for (i,j,s) in pairs
            glue[X][i,j]=s
            glue[X][j,i]=-s
        end
    end
    Lambda = kron(Matrix{ComplexF64}(I,4,4),1im*G)
    for mu in 1:2n,r in 1:4,s in 1:4
        Lambda[(r-1)*2n+mu,(s-1)*2n+mu] += glue[regions[(mu+1)÷2]][r,s]
    end
    Lambda
end

"""Independent pure Gaussian five-sector reference, with a phase-fixed Pfaffian.

Intended for small validation systems. Physical modes are grouped A/B/C;
Majorana pair order matches majorana_covariance. The vacuum fixes the
Pfaffian orientation rather than discarding a determinant square-root sign.
"""
function gaussian_sectors(G::AbstractMatrix,regions::AbstractVector{Symbol})
    n = length(regions)
    size(G)==(2n,2n) || throw(DimensionMismatch("covariance dimensions"))
    all(x->x in (:A,:B,:C),regions) || throw(ArgumentError("regions must be A/B/C"))
    norm(G+transpose(G))<1e-10 && norm(G*G+I)<1e-9 || throw(ArgumentError("pure covariance required"))
    order = sortperm(regions;by=x->findfirst(==(x),(:A,:B,:C)))
    mo = [mu for i in order for mu in (2i-1,2i)]
    g = G[mo,mo]
    regs = regions[order]
    purity = map((:A,:B,:C)) do X
        indices = [mu for i in 1:n if regs[i]==X for mu in (2i-1,2i)]
        gx = g[indices,indices]
        isempty(indices) && return 1.0
        logabs,phase = logabsdet((I-gx*gx)/2)
        abs(phase-1)<1e-10 || error("Gaussian purity determinant has invalid phase")
        exp(logabs/2)
    end
    vacuum = kron(Matrix{Float64}(I,n,n),[0.0 1.0; -1.0 0.0])
    Z4 = _pfaffian(_cube_matrix(g,regs))/_pfaffian(_cube_matrix(vacuum,regs))
    abs(imag(Z4))<=1e-10*max(abs(Z4),eps()) || error("Gaussian cube has nonreal phase")
    real(Z4)>0 || error("Gaussian cube is nonpositive")
    (;Z1=1.0,Z2A=purity[1],Z2B=purity[2],Z2C=purity[3],Z4,
      S2=-log.(collect(purity)),S3=-log(real(Z4))/2,
      stilde=-log(real(Z4)/prod(purity)),order)
end
