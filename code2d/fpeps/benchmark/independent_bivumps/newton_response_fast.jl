# Newton correction of the SAME independent mixed-boundary stationary problem.
# This is a separate globalization experiment, not the legacy direction-tied solver.
include("core.jl")

function bn_invsqrt(B)
    Q=zero(B)
    for (q,b) in blocks(B)
        F=eigen(Hermitian((b+b')/2))
        minimum(F.values)>0 || error("nonpositive polar metric")
        block(Q,q).=F.vectors*Diagonal(1 ./ sqrt.(F.values))*F.vectors'
    end
    Q
end

function bn_frechet(B,E)
    Y=zero(E)
    for (q,b) in blocks(B)
        F=eigen(Hermitian((b+b')/2));d=sqrt.(F.values)
        minimum(d)>0 || error("nonpositive polar metric")
        kernel=[-1/(a*c*(a+c)) for a in d,c in d]
        block(Y,q).=F.vectors*(kernel.*(F.vectors'*block(E,q)*F.vectors))*F.vectors'
    end
    Y
end

# Dense pilot: factor the bordered eigenvector-response equation once per cap.
# All derivatives vary R and L independently; transfer_left/right supplies the
# bra conjugation and the graded tensor conventions.
function bn_cap(f,x)
    M=bv_matrix(f,x);E=eigen(M);p=sortperm(abs.(E.values);rev=true)
    z=E.values[p[1]];v=TensorMap(copy(E.vectors[:,p[1]]),space(x));v/=norm(v)
    gap=1-abs(E.values[p[2]]/z);res=norm(f(v)-z*v)/norm(f(v))
    gap>1e-9 && res<1e-10 || error("unresolved response cap: $gap $res")
    m=length(v.data);border=zeros(ComplexF64,m+1,m+1)
    border[1:m,1:m].=M-z*I;border[1:m,end].=-v.data
    border[end,1:m].=conj.(v.data);factor=lu(border)
    function response(df)
        rhs=vcat(-df(v).data,0im);sol=factor\rhs
        err=norm(border*sol-rhs)/max(norm(rhs),eps())
        err<1e-9 || error("unresolved cap derivative $err")
        (;dv=TensorMap(copy(sol[1:m]),space(v)),dz=sol[end],err)
    end
    (;v,z,gap,res,response)
end

# R,L here are unrestricted complex, parity-even site tensors in an ordinary
# left-isometric chart. Using them also for the right channel is intentional:
# these are raw uniform tensors, with C=I, not a mixed-canonical representation.
function bn_cache(R,L,O;responses=true)
    grown=bv_grow(R,O[1]);D=grown.D;P=grown.physical;rng=MersenneTwister(2411)
    cap=responses ? bn_cap : bv_cap
    lc(A,B)=cap(x->MPSKit.transfer_left(x,A,B),randn(rng,ComplexF64,space(B,1)←space(A,1)))
    rc(A,B)=cap(x->MPSKit.transfer_right(x,A,B),randn(rng,ComplexF64,dual(space(A,4))←dual(space(B,4))))
    lt,rt=lc(D,L),rc(D,L);l0,r0=lc(R,L),rc(R,L)
    max(abs(lt.z/rt.z-1),abs(l0.z/r0.z-1))<1e-8 || error("raw channel spectra disagree")
    # Exact coefficient map for the already sign-checked graded row growth.
    # Cache its adjoint once, avoiding a dense effective-matrix construction
    # for every Hessian-vector product. No truncation or approximate inverse.
    G=bv_matrix(x->bv_grow(x,O[1]).D,R)
    growadj=Z->TensorMap(G'*Z.data,space(R))
    testR=randn(rng,ComplexF64,space(R));testD=randn(rng,ComplexF64,space(D))
    adjerr=abs(dot(testD,bv_grow(testR,O[1]).D)-dot(growadj(testD),testR))/max(norm(testR)*norm(testD),eps())
    adjerr<1e-12 || error("graded growth adjoint failed: $adjerr")
    h=(lt.v⊗P)*D*rt.v;n=(l0.v⊗P)*R*r0.v
    ht=dot(L,h);nt=dot(L,n)
    hb=growadj((lt.v⊗P)'*L*rt.v');nb=(l0.v⊗P)'*L*r0.v'
    GR=hb/conj(ht)-nb/conj(nt);GL=h/ht-n/nt;q=lt.z/l0.z
    function derivative(dR,dL)
        responses || error("response cache disabled")
        dD=bv_grow(dR,O[1]).D
        dl=lt.response(x->MPSKit.transfer_left(x,dD,L)+MPSKit.transfer_left(x,D,dL))
        dr=rt.response(x->MPSKit.transfer_right(x,dD,L)+MPSKit.transfer_right(x,D,dL))
        dl0=l0.response(x->MPSKit.transfer_left(x,dR,L)+MPSKit.transfer_left(x,R,dL))
        dr0=r0.response(x->MPSKit.transfer_right(x,dR,L)+MPSKit.transfer_right(x,R,dL))
        dh=(dl.dv⊗P)*D*rt.v+(lt.v⊗P)*dD*rt.v+(lt.v⊗P)*D*dr.dv
        dn=(dl0.dv⊗P)*R*r0.v+(l0.v⊗P)*dR*r0.v+(l0.v⊗P)*R*dr0.dv
        dhb=growadj((dl.dv⊗P)'*L*rt.v'+(lt.v⊗P)'*dL*rt.v'+(lt.v⊗P)'*L*dr.dv')
        dnb=(dl0.dv⊗P)'*L*r0.v'+(l0.v⊗P)'*dL*r0.v'+(l0.v⊗P)'*L*dr0.dv'
        dht=dot(dL,h)+dot(L,dh);dnt=dot(dL,n)+dot(L,dn)
        dGR=dhb/conj(ht)-hb*conj(dht/ht^2)-dnb/conj(nt)+nb*conj(dnt/nt^2)
        dGL=dh/ht-h*(dht/ht^2)-dn/nt+n*(dnt/nt^2)
        dlog=dl.dz/lt.z-dl0.dz/l0.z;predicted=dot(GR,dR)+dot(dL,GL)
        (;dGR,dGL,dlog,response_error=maximum(t.err for t in (dl,dr,dl0,dr0)),
          scalar_derivative_error=abs(dlog-predicted))
    end
    (;GR,GL,q,derivative,grow_adjoint_error=adjerr,cap_residual=maximum(t.res for t in (lt,rt,l0,r0)))
end

bn_real(x)=vcat(real.(x),imag.(x))
function bn_chart(R,L,O)
    NR=TensorKit.left_null(R);NL=TensorKit.left_null(L)
    tr=zero(NR'*R);tl=zero(NL'*L);nr=length(tr.data);nl=length(tl.data);count=nr+nl
    function unpack(v)
        length(v)==2count || error("wrong real tangent dimension")
        z=complex.(v[1:count],v[count+1:2count])
        VR=TensorMap(copy(z[1:nr]),space(tr));VL=TensorMap(copy(z[nr+1:end]),space(tl))
        VR,VL
    end
    function tensors(v)
        VR,VL=unpack(v)
        ZR=R+NR*VR;ZRmetric=id(ComplexF64,domain(R))+VR'*VR;QR=bn_invsqrt(ZRmetric)
        ZL=L+NL*VL;ZLmetric=id(ComplexF64,domain(L))+VL'*VL;QL=bn_invsqrt(ZLmetric)
        (;R=ZR*QR,L=ZL*QL,VR,VL,ZR,ZL,QR,QL,ZRmetric,ZLmetric)
    end
    base=bn_cache(R,L,O)
    gradient=bn_real(vcat((NR'*base.GR).data,(NL'*base.GL).data))
    CR=(R'*base.GR+base.GR'*R)/2;CL=(L'*base.GL+base.GL'*L)/2
    checks=Dict{String,Any}[]
    function action(v)
        VR,VL=unpack(v);d=base.derivative(NR*VR,NL*VL)
        push!(checks,Dict("cap_response_error"=>d.response_error,
            "scalar_derivative_error"=>d.scalar_derivative_error))
        bn_real(vcat((NR'*d.dGR-VR*CR).data,(NL'*d.dGL-VL*CL).data))
    end
    function evaluate(v)
        t=tensors(v);c=bn_cache(t.R,t.L,O;responses=false)
        ER=(t.ZR'*c.GR+c.GR'*t.ZR)/2;EL=(t.ZL'*c.GL+c.GL'*t.ZL)/2
        pr=NR'*c.GR*t.QR+2t.VR*bn_frechet(t.ZRmetric,ER)
        pl=NL'*c.GL*t.QL+2t.VL*bn_frechet(t.ZLmetric,EL)
        (;R=t.R,L=t.L,q=c.q,gradient=bn_real(vcat(pr.data,pl.data)))
    end
    (;R,L,O,count=2count,nr,nl,gradient,action,evaluate,tensors,base,checks)
end

function bn_audit(source,out)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");mkpath(out);BLAS.set_num_threads(2)
    n,s=deserialize.(joinpath.(source,("boundary_1.jls","boundary_3.jls")))
    R=n.state.AL[1];L=MPSKit.InfiniteMPS([FermionicPEPS._spatial_boundary_bra(s.state.AR[1])]).AL[1]
    rng=MersenneTwister(2412)
    # Nonstationary independent complex perturbations prevent vacuous zero tests.
    R=R+0.002randn(rng,ComplexF64,space(R));R=R*bn_invsqrt(R'*R)
    L=L+0.003randn(rng,ComplexF64,space(L));L=L*bn_invsqrt(L'*L)
    c=bn_chart(R,L,n.transfer);rows=Dict{String,Any}[];zero_v=zeros(c.count)
    g0=c.evaluate(zero_v).gradient
    norm(g0-c.gradient)<1e-9 || error("raw chart gradient differs at origin")
    directions=[]
    for side in ("right","left"),quadrature in ("real","imag")
        v=zeros(c.count);offset=quadrature=="real" ? 0 : div(c.count,2)
        inds=side=="right" ? (1:c.nr) : (c.nr+1:c.nr+c.nl)
        v[offset .+ inds].=randn(rng,length(inds));v/=norm(v);push!(directions,v)
        hv=c.action(v);predicted=dot(c.gradient,v);stencils=[]
        for h in (1e-4,3e-5,1e-5)
            p=c.evaluate(h*v);m=c.evaluate(-h*v)
            scalar=real((p.q-m.q)/(2h*c.base.q))
            fd=(p.gradient-m.gradient)/(2h)
            push!(stencils,Dict("step"=>h,"scalar_error"=>abs(scalar-predicted),
                "hessian_relative_error"=>norm(fd-hv)/max(norm(hv),eps())))
        end
        abs(predicted)>1e-7 || error("accidentally stationary scalar control")
        minimum(r["scalar_error"] for r in stencils)<1e-7 || error("scalar gradient check failed")
        minimum(r["hessian_relative_error"] for r in stencils)<1e-4 || error("Hessian finite-difference check failed")
        push!(rows,Dict("side"=>side,"quadrature"=>quadrature,"predicted"=>predicted,"stencils"=>stencils))
        bv_write(joinpath(out,"checks.toml"),Dict("complete"=>false,"checks"=>rows))
    end
    u,v=directions[1],directions[4];hu,hv=c.action(u),c.action(v)
    symmetry=abs(dot(u,hv)-dot(v,hu))/max(norm(hu),norm(hv),eps())
    linearity=norm(c.action(u+v)-hu-hv)/max(norm(hu),norm(hv),eps())
    max(symmetry,linearity)<1e-8 || error("analytic Hessian symmetry/linearity failed")
    bv_write(joinpath(out,"checks.toml"),Dict("complete"=>true,"passed"=>true,
        "independent_left_right"=>true,"direction_constraint"=>false,
        "checks"=>rows,"hessian_symmetry_error"=>symmetry,"hessian_linearity_error"=>linearity,
        "response_checks"=>c.checks,"parameter_dimension"=>c.count,
        "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))),
        "response_sha256"=>bytes2hex(sha256(read(@__FILE__)))))
    println("independent complex Newton response audit passed");flush(stdout)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    bn_audit(ARGS...)
end
