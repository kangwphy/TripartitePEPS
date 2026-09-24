# Independent bivariational VUMPS pilot. Does not load any legacy joint solver.
# L denotes the KET representation of the bra <L| in the SAME oriented chart as R.
include("../stable_boundary_pilot.jl")
using KrylovKit, SHA
const BIVUMPS_SCHEMA = "independent_bivumps_v1"

function bv_write(path,report)
    mkpath(dirname(path))
    open(path*".tmp","w") do io; TOML.print(io,report); end
    mv(path*".tmp",path;force=true)
end

function bv_matrix(f,x)
    n=length(x.data); n<=4096 || error("dense pilot dimension exceeds 4096")
    y=f(x); M=zeros(ComplexF64,length(y.data),n)
    for j in 1:n
        v=zero(x); v.data[j]=1
        M[:,j]=f(v).data
    end
    M
end

function bv_cap(f,x)
    if length(x.data)<=128
        E=eigen(bv_matrix(f,x)); p=sortperm(abs.(E.values);rev=true)
        vals=E.values[p]; v=TensorMap(copy(E.vectors[:,p[1]]),space(x))
    else
        vals,vs,info=eigsolve(f,x,2,:LM;tol=1e-13,maxiter=500,krylovdim=80)
        info.converged>=2 || error("mixed cap eigensolver failed")
        p=sortperm(abs.(vals);rev=true); vals=vals[p]; v=vs[p[1]]
    end
    v/=norm(v); z=vals[1]
    res=norm(f(v)-z*v)/norm(f(v)); gap=1-abs(vals[2]/z)
    res<1e-10 && gap>1e-9 || error("cap failed: residual=$res gap=$gap")
    (;v,z,res,gap)
end

function bv_grow(A,O)
    raw=FermionicPEPS._grow_boundary(A,PEPSKit.ket(O),PEPSKit.bra(O))
    Q=permute(PEPSKit.twistdual(raw,(2,3)),((1,2,3,4,5),(6,7,8)))
    Vl=space(Q,1)⊗space(Q,2)⊗space(Q,3); Vr=domain(Q)
    UL=isomorphism(ComplexF64,fuse(Vl)←Vl)
    UR=isomorphism(ComplexF64,fuse(Vr)←Vr)
    physical=id(ComplexF64,space(Q,4))⊗id(ComplexF64,space(Q,5))
    (;D=(UL⊗physical)*Q*UR',UL,UR,physical)
end

function bv_environment(R,L,O)
    gl=bv_grow(R.AL[1],O[1]); gr=bv_grow(R.AR[1],O[1])
    BL,BR=L.AL[1],L.AR[1]; rng=MersenneTwister(19429)
    lc(A,B)=bv_cap(x->MPSKit.transfer_left(x,A,B),randn(rng,ComplexF64,space(B,1)←space(A,1)))
    rc(A,B)=bv_cap(x->MPSKit.transfer_right(x,A,B),randn(rng,ComplexF64,dual(space(A,4))←dual(space(B,4))))
    lt,rt=lc(gl.D,BL),rc(gr.D,BR)
    l0,r0=lc(R.AL[1],BL),rc(R.AR[1],BR)
    spectral_error=max(abs(lt.z/rt.z-1),abs(l0.z/r0.z-1))
    spectral_error<1e-8 || error("canonical mixed channel eigenvalues disagree")
    l=lt.v*gl.UL; r=gr.UR'*rt.v; d=domain(l)
    Hac=x->(lt.v⊗gl.physical)*bv_grow(x,O[1]).D*rt.v
    Nac=x->(l0.v⊗gl.physical)*x*r0.v
    Hc=x->l*(x⊗id(ComplexF64,d[2])⊗id(ComplexF64,d[3]))*r
    Nc=x->l0.v*x*r0.v
    space(Hac(R.AC[1]))==space(L.AC[1]) || error("AC output is not the independent left space")
    space(Hc(R.C[1]))==space(L.C[1]) || error("C output is not the independent left space")
    ac=(H=bv_matrix(Hac,R.AC[1]),N=bv_matrix(Nac,R.AC[1]))
    c=(H=bv_matrix(Hc,R.C[1]),N=bv_matrix(Nc,R.C[1]))
    # This cross-check relates local center contractions to the independently
    # computed leading row/overlap channel eigenvalues, including graded cups.
    zac=dot(L.AC[1].data,ac.H*R.AC[1].data)/dot(L.AC[1].data,ac.N*R.AC[1].data)
    zc=dot(L.C[1].data,c.H*R.C[1].data)/dot(L.C[1].data,c.N*R.C[1].data)
    q=lt.z/l0.z; quotient_error=abs(zac/(zc*q)-1)
    quotient_error<1e-8 || error("center/channel quotient identity failed: $quotient_error")
    (;ac,c,q,lt,rt,l0,r0,spectral_error,quotient_error)
end

function bv_white(H,N;cutoff=1e-12)
    F=svd(N); ratio=minimum(F.S)/maximum(F.S)
    ratio>cutoff || error("mixed metric singular ratio $ratio; no silent rank truncation")
    # Separate dual bases; no explicit inverse of N. The center coordinates
    # are restored before ordinary canonicalization of each independent MPS.
    X=F.V*Diagonal(1 ./ sqrt.(F.S)); Y=F.U*Diagonal(1 ./ sqrt.(F.S))
    err=norm(Y'*N*X-I)/sqrt(size(N,1))
    err<1e-8 || error("biorthogonalization residual $err")
    (;K=Y'*H*X,X,Y,ratio,err)
end

function bv_pencil(P,r0,l0;target=nothing)
    W=bv_white(P.H,P.N); E=eigen(W.K); F=eigen(W.K')
    radius=maximum(abs,E.values)
    candidates=if target===nothing
        findall(abs.(E.values).>=radius*(1-1e-8))
    else
        distances=abs.(E.values.-target)
        findall(distances.<=minimum(distances)+1e-9*max(radius,1))
    end
    overlaps=[abs(dot(r0.data,W.X*E.vectors[:,j]))/norm(W.X*E.vectors[:,j]) for j in candidates]
    j=candidates[argmax(overlaps)]; z=E.values[j]
    k=argmin(abs.(F.values.-conj(z)))
    abs(F.values[k]-conj(z))<1e-8*max(abs(z),1) || error("left/right eigenvalues not matched")
    r=W.X*E.vectors[:,j]; l=W.Y*F.vectors[:,k]
    r/=norm(r); l/=norm(l)
    function align(v,x)
        ov=dot(x.data,v); abs(ov)>1e-14 && (v*=conj(ov)/abs(ov)); v
    end
    r=align(r,r0);l=align(l,l0)
    resr=norm(P.H*r-z*P.N*r)/max(norm(P.H*r),eps())
    resl=norm(P.H'*l-conj(z)*P.N'*l)/max(norm(P.H'*l),eps())
    max(resr,resl)<1e-8 || error("paired local pencil residual too large")
    (;r=TensorMap(copy(r),space(r0)),l=TensorMap(copy(l),space(l0)),z,
      report=Dict("right_inner_residual"=>resr,"left_inner_residual"=>resl,
        "metric_singular_ratio"=>W.ratio,"biorthogonal_error"=>W.err,
        "eigenvalue_real"=>real(z),"eigenvalue_imag"=>imag(z),
        "selected_modulus"=>abs(z),"largest_modulus"=>radius,
        "left_right_metric_overlap"=>abs(dot(l,P.N*r))))
end

function bv_residual(P,r,l)
    rv,lv=r.data,l.data; z=dot(lv,P.H*rv)/dot(lv,P.N*rv)
    rr=norm(P.H*rv-z*P.N*rv)/max(norm(P.H*rv),abs(z)*norm(P.N*rv),eps())
    rl=norm(P.H'*lv-conj(z)*P.N'*lv)/max(norm(P.H'*lv),abs(z)*norm(P.N'*lv),eps())
    W=bv_white(P.H,P.N)
    # Petrov residual in dual whitened coordinates, in addition to raw residual.
    wr=norm(W.Y'*(P.H*rv-z*P.N*rv))/max(norm(W.Y'*P.H*rv),eps())
    wl=norm(W.X'*(P.H'*lv-conj(z)*P.N'*lv))/max(norm(W.X'*P.H'*lv),eps())
    (;z,res=max(rr,rl,wr,wl),report=Dict("right"=>rr,"left"=>rl,
        "right_white"=>wr,"left_white"=>wl,"metric_singular_ratio"=>W.ratio,
        "biorthogonal_error"=>W.err))
end

function bv_audit(R,L,O)
    env=bv_environment(R,L,O)
    a=bv_residual(env.ac,R.AC[1],L.AC[1]); c=bv_residual(env.c,R.C[1],L.C[1])
    canonical(s)=max(norm(s.AC[1]-s.AL[1]*s.C[1]),
        norm(s.AC[1]-MPSKit._mul_front(s.C[0],s.AR[1])))/norm(s.AC[1])
    cr,cl=canonical(R),canonical(L)
    relation=abs(a.z/(c.z*env.q)-1)
    residual=max(a.res,c.res,cr,cl,relation)
    report=Dict("outer_residual"=>residual,"AC"=>a.report,"C"=>c.report,
        "canonical_right"=>cr,"canonical_left"=>cl,"center_relation_error"=>relation,
        "row_real"=>real(env.q),"row_imag"=>imag(env.q),
        "cap_residual"=>maximum(v.res for v in (env.lt,env.rt,env.l0,env.r0)),
        "cap_spectral_error"=>env.spectral_error,"quotient_identity_error"=>env.quotient_error)
    (;env,residual,report)
end

function bv_retract(old,ac,c,alpha)
    A=(1-alpha)*old.AC[1]/norm(old.AC[1])+alpha*ac
    C=(1-alpha)*old.C[1]/norm(old.C[1])+alpha*c
    raw=MPSKit.regauge!(A/norm(A),C/norm(C))
    MPSKit.InfiniteMPS([raw];tol=1e-13,maxiter=1000)
end

function bv_xi(R,L)
    spectra=Vector{ComplexF64}[]; A=R.AL[1];B=L.AL[1];rng=MersenneTwister(19)
    for q in (0,1)
        x=similar(A,space(B,1),Vect[FermionParity](FermionParity(q)=>1)'*space(A,1))
        randn!(rng,x.data); f=y->MPSKit.transfer_left(y,A,B)
        if length(x.data)<=128
            vals=eigvals(bv_matrix(f,x))
        else
            vals,_,info=eigsolve(f,x,3,:LM;tol=1e-12,maxiter=500,krylovdim=80)
            info.converged>=3 || error("xi spectrum unconverged")
        end
        push!(spectra,sort(vals;by=abs,rev=true))
    end
    ratio=max(abs(spectra[1][2]),maximum(abs,spectra[2]))/abs(spectra[1][1])
    Dict("ratio"=>ratio,"xi"=>(ratio<1 ? -1/log(ratio) : Inf),
        "decaying"=>ratio<1,"even_abs"=>abs.(spectra[1][1:min(4,end)]),
        "odd_abs"=>abs.(spectra[2][1:min(4,end)]))
end
