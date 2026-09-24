# Full mixed-center equation field in left-canonical coordinates.
# The graded cache below is the unchanged branch_response.jl algebra with
# normalized H/N terms additionally exposed for residual normalization.
include("paired_tangent_metric.jl")

function cg_cache(R,L,O;responses=true,refs=nothing)
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
        dHR=dhb/conj(ht)-hb*conj(dht/ht^2)
        dNR=dnb/conj(nt)-nb*conj(dnt/nt^2)
        dHL=dh/ht-h*(dht/ht^2);dNL=dn/nt-n*(dnt/nt^2)
        (;dGR,dGL,dHR,dNR,dHL,dNL,dlog,response_error=maximum(t.err for t in (dl,dr,dl0,dr0)),
          scalar_derivative_error=abs(dlog-predicted))
    end
    (;GR,GL,HR=hb/conj(ht),NR=nb/conj(nt),HL=h/ht,NL=n/nt,q,derivative,refs=nextrefs,reports,grow_adjoint_error=adjerr,cap_residual=maximum(t.res for t in (lt,rt,l0,r0)))
end

cg_trace(x)=dot(id(ComplexF64,domain(x)),x)

function cg_density(A;responses=true)
    rng=MersenneTwister(31901)
    cap=bt_cap(x->MPSKit.transfer_right(x,A,A),
        randn(rng,ComplexF64,dual(space(A,4))←dual(space(A,4)));responses)
    trace=cg_trace(cap.v);abs(trace)>1e-12 || error("self density trace vanishes")
    raw=cap.v/trace
    herm=norm(raw-raw')/norm(raw);herm<1e-10 || error("self density is not Hermitian")
    rho=(raw+raw')/2;W=bn_invsqrt(rho)
    values=vcat([eigvals(Hermitian(b)) for (_,b) in blocks(rho)]...)
    minimum(values)>0 || error("self density is not positive; no rank cut")
    function derivative(dA)
        responses || error("self density response disabled")
        d=cap.response(x->MPSKit.transfer_right(x,dA,A)+MPSKit.transfer_right(x,A,dA))
        dr=(d.dv-raw*cg_trace(d.dv))/trace
        norm(dr-dr')<=1e-11+1e-8norm(dr) || error("density response lost Hermiticity")
        dr=(dr+dr')/2
        (;rho=dr,W=bn_frechet(rho,dr),cap_response_error=d.err)
    end
    (;rho,W,derivative,report=Dict("hermiticity_error"=>herm,
        "minimum_eigenvalue"=>minimum(values),"maximum_eigenvalue"=>maximum(values),
        "rank_truncated"=>false,"cap"=>cap.report))
end

# RMS normalization is smooth at H=N and differs from the original maximum
# normalization by at most sqrt(2). Final acceptance still uses core.jl.
function cg_fields(A,B,cache,wr,wl)
    hR,nR=cache.HR*wr,cache.NR*wr;hL,nL=cache.HL*wl,cache.NL*wl
    gR,gL=cache.GR*wr,cache.GL*wl
    hCR,nCR=A'*hR,A'*nR;hCL,nCL=B'*hL,B'*nL
    gCR,gCL=A'*gR,B'*gL
    scale(h,n)=sqrt((norm(h)^2+norm(n)^2)/2)
    sR,sL,sCR,sCL=scale(hR,nR),scale(hL,nL),scale(hCR,nCR),scale(hCL,nCL)
    minimum((sR,sL,sCR,sCL))>1e-12 || error("center normalization vanishes")
    FR,FL,FCR,FCL=gR/sR,gL/sL,gCR/sCR,gCL/sCL
    field=bn_real(vcat(FR.data,FL.data,FCR.data,FCL.data))
    (;field,FR,FL,FCR,FCL,hR,nR,hL,nL,hCR,nCR,hCL,nCL,gR,gL,gCR,gCL,sR,sL,sCR,sCL)
end

function cg_field_derivative(A,B,cache,fields,wr,wl,dA,dB,d,dwr,dwl)
    dhR,dnR=d.dHR*wr+cache.HR*dwr,d.dNR*wr+cache.NR*dwr
    dhL,dnL=d.dHL*wl+cache.HL*dwl,d.dNL*wl+cache.NL*dwl
    dgR,dgL=d.dGR*wr+cache.GR*dwr,d.dGL*wl+cache.GL*dwl
    dhCR,dnCR=dA'*fields.hR+A'*dhR,dA'*fields.nR+A'*dnR
    dhCL,dnCL=dB'*fields.hL+B'*dhL,dB'*fields.nL+B'*dnL
    dgCR,dgCL=dA'*fields.gR+A'*dgR,dB'*fields.gL+B'*dgL
    function normalize(dg,dh,dn,h,n,F,s)
        ds=real(dot(h,dh)+dot(n,dn))/(2s)
        dg/s-F*(ds/s)
    end
    dFR=normalize(dgR,dhR,dnR,fields.hR,fields.nR,fields.FR,fields.sR)
    dFL=normalize(dgL,dhL,dnL,fields.hL,fields.nL,fields.FL,fields.sL)
    dFCR=normalize(dgCR,dhCR,dnCR,fields.hCR,fields.nCR,fields.FCR,fields.sCR)
    dFCL=normalize(dgCL,dhCL,dnCL,fields.hCL,fields.nCL,fields.FCL,fields.sCL)
    (;field=bn_real(vcat(dFR.data,dFL.data,dFCR.data,dFCL.data)),dFR,dFL,dFCR,dFCL)
end

function cg_chart(R,L,O;refs=nothing)
    A,B=R.AL[1],L.AL[1];NR,NL=TensorKit.left_null(A),TensorKit.left_null(B)
    tr,tl=zero(NR'*A),zero(NL'*B);nr,nl=length(tr.data),length(tl.data);count=nr+nl
    function unpack(v)
        length(v)==2count || error("wrong center-gradient coordinate dimension")
        z=complex.(v[1:count],v[count+1:end])
        TensorMap(copy(z[1:nr]),space(tr)),TensorMap(copy(z[nr+1:end]),space(tl))
    end
    base=cg_cache(A,B,O;refs);dr,dl=cg_density(A),cg_density(B)
    fields=cg_fields(A,B,base,dr.W,dl.W)
    density_error=max(norm(dr.rho-R.C[1]*R.C[1]'/norm(R.C[1])^2),
        norm(dl.rho-L.C[1]*L.C[1]'/norm(L.C[1])^2))
    density_error<1e-10 || error("self density disagrees with canonical C")
    weights=pt_weights(R,L;refs=base.refs);P=weights.pushforward
    search=bn_real(vcat((NR'*fields.FR).data,(NL'*fields.FL).data))
    function raw_action(v)
        VR,VL=unpack(v);dA,dB=NR*VR,NL*VL
        d=base.derivative(dA,dB);ddr,ddl=dr.derivative(dA),dl.derivative(dB)
        j=cg_field_derivative(A,B,base,fields,dr.W,dl.W,dA,dB,d,ddr.W,ddl.W)
        search_j=bn_real(vcat((NR'*j.dFR-VR*(A'*fields.FR)).data,
            (NL'*j.dFL-VL*(B'*fields.FL)).data))
        (;field=j.field,search=search_j)
    end
    function raw_evaluate(v)
        VR,VL=unpack(v)
        ar=(A+NR*VR)*bn_invsqrt(id(ComplexF64,domain(A))+VR'*VR)
        al=(B+NL*VL)*bn_invsqrt(id(ComplexF64,domain(B))+VL'*VL)
        cache=cg_cache(ar,al,O;refs=base.refs,responses=false)
        rr,ll=cg_density(ar;responses=false),cg_density(al;responses=false)
        f=cg_fields(ar,al,cache,rr.W,ll.W)
        pr=(NR-A*VR')*bn_invsqrt(id(ComplexF64,codomain(VR))+VR*VR')
        pl=(NL-B*VL')*bn_invsqrt(id(ComplexF64,codomain(VL))+VL*VL')
        search_at_point=bn_real(vcat((pr'*f.FR).data,(pl'*f.FL).data))
        (;R=ar,L=al,field=f.field,search=search_at_point,refs=cache.refs,q=cache.q,
          reports=cache.reports,density_right=rr.report,density_left=ll.report)
    end
    evaluate=v->begin
        r=raw_evaluate(P(v))
        (;R=r.R,L=r.L,field=r.field,gradient=P(r.search),refs=r.refs,q=r.q,
          reports=r.reports,density_right=r.density_right,density_left=r.density_left)
    end
    action=v->begin
        j=raw_action(P(v));(;field=j.field,gradient=P(j.search))
    end
    (;base,fields,field=fields.field,gradient=P(search),raw_gradient=search,
      raw_action,raw_evaluate,action,evaluate,pushforward=P,weights,NR,NL,nr,nl,
      count=2count,unpack,density_error,density_right=dr.report,density_left=dl.report)
end

function cg_audit(source,snapshot,out;perturb=false)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    files=[joinpath(source,"boundary_1.jls"),joinpath(source,"boundary_3.jls"),snapshot]
    append!(files,[joinpath(@__DIR__,f) for f in ("center_gradient_response.jl",
        "paired_tangent_metric.jl","antiunitary_newton_chart.jl","branch_response.jl",
        "newton_metric_response.jl","newton_response_fast.jl","core.jl")])
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in files)
    meta=Dict{String,Any}("complete"=>false,"passed"=>false,"diagnostic_only"=>true,
        "nonstationary"=>true,"independent_left_right"=>true,"direction_constraint"=>false,
        "source_hashes"=>hashes,"job_id"=>ENV["SLURM_JOB_ID"],"perturbed"=>perturb,
        "accepted_entropy"=>false)
    bv_write(out,meta)
    try
        pair=deserialize(snapshot);R,L=deepcopy(pair.R),deepcopy(pair.L)
        refs=hasproperty(pair,:refs) ? pair.refs : nothing
        if perturb
            n,s=deserialize.(files[1:2]);maps=ba_pair_maps(n,s);states=[]
            for (i,state,K) in ((1,R,maps.right),(2,L,maps.left))
                symmetry=at_symmetry(state,K)
                A=state.AL[1]+0.002randn(MersenneTwister(31910+i),ComplexF64,space(state.AL[1]))
                A=(A+symmetry.apply(A))/2;A=A*bn_invsqrt(A'*A)
                candidate,_=bt_canonical(A);push!(states,candidate)
            end
            R,L=states;refs=nothing
        end
        c=cg_chart(R,L,pair.transfer;refs);audit=bv_audit(R,L,pair.transfer)
        norm(c.field)>1e-7 || error("stationary zero is not a derivative control")
        meta["field_norm"]=norm(c.field);meta["full_equations"]=audit.report
        meta["density_right"]=c.density_right;meta["density_left"]=c.density_left
        meta["density_C_error"]=c.density_error
        reference=Float64[]
        for (p,r,l) in ((audit.env.ac,R.AC[1],L.AC[1]),(audit.env.c,R.C[1],L.C[1]))
            h=p.H*r.data;n=p.N*r.data;ht=dot(l.data,h);nt=dot(l.data,n)
            for (h0,n0) in ((p.H'*l.data/conj(ht),p.N'*l.data/conj(nt)),(h/ht,n/nt))
                push!(reference,norm(h0-n0)/sqrt((norm(h0)^2+norm(n0)^2)/2))
            end
        end
        got=norm.((c.fields.FR,c.fields.FL,c.fields.FCR,c.fields.FCL))
        identity=norm(collect(got)-reference)/norm(reference)
        meta["direct_center_norms"]=reference;meta["converted_center_norms"]=collect(got)
        meta["center_field_identity_error"]=identity
        identity<1e-6 || error("center field differs from original equations")
        origin=c.evaluate(zeros(c.count));origin_error=norm(origin.field-c.field)/norm(c.field)
        meta["origin_error"]=origin_error;origin_error<1e-6 || error("center field origin differs")
        rows=[];meta["derivative_checks"]=rows;rng=MersenneTwister(31921);half=div(c.count,2)
        for side in ("right","left"),quadrature in ("real","imag")
            v=zeros(c.count);inds=side=="right" ? (1:c.nr) : (c.nr+1:half)
            offset=quadrature=="real" ? 0 : half
            v[offset .+ inds]=randn(rng,length(inds));v/=norm(c.pushforward(v))
            j=c.action(v);stencils=[]
            for h in (1e-4,3e-5,1e-5,3e-6,1e-6)
                p,m=c.evaluate(h*v),c.evaluate(-h*v)
                fd=(p.field-m.field)/(2h);sd=(p.gradient-m.gradient)/(2h)
                push!(stencils,Dict("step"=>h,"field_relative_error"=>norm(fd-j.field)/max(norm(j.field),eps()),
                    "search_relative_error"=>norm(sd-j.gradient)/max(norm(j.gradient),eps())))
            end
            push!(rows,Dict("side"=>side,"quadrature"=>quadrature,"stencils"=>stencils));bv_write(out,meta)
            minimum(s["field_relative_error"] for s in stencils)<1e-4 || error("center field derivative mismatch")
            minimum(s["search_relative_error"] for s in stencils)<1e-4 || error("center search derivative mismatch")
        end
        meta["passed"]=true
    catch err
        meta["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        meta["complete"]=true;bv_write(out,meta)
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    cg_audit(ARGS[1],ARGS[2],ARGS[3];perturb=parse(Bool,ARGS[4]))
end
