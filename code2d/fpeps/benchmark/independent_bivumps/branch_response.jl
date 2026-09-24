# Isolated analytic-branch continuation of signed transfer caps.
# Intermediate caps may be subdominant. Final acceptance still uses core.jl,
# which independently selects the unique maximal-modulus caps.
include("newton_metric_response.jl")

function bt_cap(f,x;target=nothing,responses=true)
    M=bv_matrix(f,x);E=eigen(M);order=sortperm(abs.(E.values);rev=true)
    index=target===nothing ? order[1] : argmin(abs.(E.values.-target))
    z=E.values[index];others=E.values[setdiff(eachindex(E.values),[index])]
    sep=minimum(abs.(others.-z))/max(abs(z),eps())
    sep>1e-9 || error("analytic cap eigenvalue not isolated: $sep")
    distance=target===nothing ? 0.0 : abs(z-target)/max(abs(z),eps())
    distance<0.4sep || error("ambiguous analytic cap continuation: $distance >= 0.4 * $sep")
    v=TensorMap(copy(E.vectors[:,index]),space(x));v/=norm(v)
    res=norm(f(v)-z*v)/norm(f(v));res<1e-10 || error("analytic cap residual failed")
    rank=findfirst(==(index),order)
    report=Dict("modulus_rank"=>rank,"relative_complex_separation"=>sep,
        "continuation_distance"=>distance,"selected_real"=>real(z),"selected_imag"=>imag(z),
        "selected_abs"=>abs(z),"dominant_abs"=>abs(E.values[order[1]]),
        "selected_to_dominant_modulus"=>abs(z/E.values[order[1]]),"residual"=>res)
    function unavailable(df);error("response cache disabled");end
    response=unavailable
    if responses
        m=length(v.data);border=zeros(ComplexF64,m+1,m+1)
        border[1:m,1:m].=M-z*I;border[1:m,end].=-v.data
        border[end,1:m].=conj.(v.data);factor=lu(border)
        response=df->begin
            rhs=vcat(-df(v).data,0im);sol=factor\rhs
            err=norm(border*sol-rhs)/max(norm(rhs),eps())
            err<1e-9 || error("analytic cap response failed: $err")
            (;dv=TensorMap(copy(sol[1:m]),space(v)),dz=sol[end],err)
        end
    end
    (;v,z,res,response,report)
end

function bt_cache(R,L,O;responses=true,refs=nothing)
    grown=bv_grow(R,O[1]);D=grown.D;P=grown.physical;rng=MersenneTwister(2411)
    target(k)=refs===nothing ? nothing : refs[k]
    lc(A,B,k)=bt_cap(x->MPSKit.transfer_left(x,A,B),randn(rng,ComplexF64,space(B,1)←space(A,1));target=target(k),responses)
    rc(A,B,k)=bt_cap(x->MPSKit.transfer_right(x,A,B),randn(rng,ComplexF64,dual(space(A,4))←dual(space(B,4)));target=target(k),responses)
    lt,rt=lc(D,L,:lt),rc(D,L,:rt);l0,r0=lc(R,L,:l0),rc(R,L,:r0)
    nextrefs=Dict(:lt=>lt.z,:rt=>rt.z,:l0=>l0.z,:r0=>r0.z)
    reports=Dict(string(k)=>getproperty((;lt,rt,l0,r0),k).report for k in (:lt,:rt,:l0,:r0))
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
    (;GR,GL,q,derivative,refs=nextrefs,reports,grow_adjoint_error=adjerr,cap_residual=maximum(t.res for t in (lt,rt,l0,r0)))
end

function bt_chart(R,L,O;refs=nothing)
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
    base=bt_cache(R,L,O;refs)
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
        t=tensors(v);c=bt_cache(t.R,t.L,O;responses=false,refs=base.refs)
        ER=(t.ZR'*c.GR+c.GR'*t.ZR)/2;EL=(t.ZL'*c.GL+c.GL'*t.ZL)/2
        pr=NR'*c.GR*t.QR+2t.VR*bn_frechet(t.ZRmetric,ER)
        pl=NL'*c.GL*t.QL+2t.VL*bn_frechet(t.ZLmetric,EL)
        (;R=t.R,L=t.L,q=c.q,refs=c.refs,reports=c.reports,gradient=bn_real(vcat(pr.data,pl.data)))
    end
    (;R,L,O,count=2count,nr,nl,gradient,action,evaluate,tensors,base,checks)
end

function btm_chart(R,L,O;refs=nothing)
    base=bt_chart(R.AL[1],L.AL[1],O;refs)
    wr=nm_weight(R.C[1]);wl=nm_weight(L.C[1])
    tr=zero(TensorKit.left_null(R.AL[1])'*R.AL[1])
    tl=zero(TensorKit.left_null(L.AL[1])'*L.AL[1])
    function pushforward(v)
        count=div(base.count,2);length(v)==base.count || error("wrong tangent dimension")
        z=complex.(v[1:count],v[count+1:end])
        vr=TensorMap(copy(z[1:base.nr]),space(tr));vl=TensorMap(copy(z[base.nr+1:end]),space(tl))
        bn_real(vcat((vr*wr.W).data,(vl*wl.W).data))
    end
    gradient=pushforward(base.gradient)
    action=v->pushforward(base.action(pushforward(v)))
    function evaluate(v)
        p=base.evaluate(pushforward(v))
        (;R=p.R,L=p.L,q=p.q,refs=p.refs,reports=p.reports,gradient=pushforward(p.gradient))
    end
    weights=Dict("right_density_min"=>wr.minimum,"left_density_min"=>wl.minimum,
        "inverse_density_power"=>0.5,"rank_truncated"=>false)
    (;count=base.count,nr=base.nr,nl=base.nl,gradient,action,evaluate,
      checks=base.checks,base=base.base,pushforward,weights)
end

# Preserve the raw AL gauge AND phase across outer iterations. Reuse the
# existing polar-overlap alignment mathematics, independently on each state.
function bt_canonical(A)
    state=MPSKit.InfiniteMPS([A];tol=1e-13,maxiter=1000)
    B=state.AL[1];rng=MersenneTwister(2491)
    cap=bv_cap(x->MPSKit.transfer_left(x,B,A),randn(rng,ComplexF64,space(A,1)←space(B,1)))
    U,_,Vh=svd_compact(cap.v);G=U*Vh;phase=cap.z/abs(cap.z)
    P=id(ComplexF64,space(A,2))⊗id(ComplexF64,space(A,3))
    move=X->(G⊗P)*X*G'/phase
    C=G*state.C[1]*G'
    result=MPSKit.InfiniteMPS([move(B)],[move(state.AR[1])],[C],[move(state.AC[1])])
    error=norm(result.AL[1]-A)/norm(A)
    error<1e-9 || Base.error("raw AL/phase not preserved: $error")
    canonical=max(norm(result.AC[1]-result.AL[1]*result.C[1]),
        norm(result.AC[1]-MPSKit._mul_front(result.C[0],result.AR[1])))/norm(result.AC[1])
    canonical<1e-10 || Base.error("canonical identities failed: $canonical")
    result,Dict("AL_relative_error"=>error,"canonical_error"=>canonical,
        "self_fidelity_error"=>abs(abs(cap.z)-1))
end
