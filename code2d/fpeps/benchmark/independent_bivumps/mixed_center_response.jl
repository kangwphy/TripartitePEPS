# Full mixed-center residual including the CHANGING dual-whitening metric.
# Isolated from existing solvers and controls. Dense pilot for small chi.
include("center_gradient_response.jl")

# Positive whitening avoids differentiating arbitrary SVD vector phases.
# WL=(N*N')^(-1/4), WR=(N'*N)^(-1/4), evaluated via SVD without
# squaring the condition number in the numerical eigensolver.
function mw_metric(N)
    f=svd(N);s=f.S;ratio=minimum(s)/maximum(s)
    ratio>1e-12 || error("mixed metric lost rank; no rank cut: $ratio")
    U,V=f.U,f.V;a=sqrt.(s)
    WL=U*Diagonal(1 ./ a)*U';WR=V*Diagonal(1 ./ a)*V'
    # Divided difference for t^(-1/4), including coincident singular values.
    K=[-1/(x*y*(x+y)*(x*x+y*y)) for x in a,y in a]
    function derivative(dN)
        dleft=dN*N'+N*dN';dright=dN'*N+N'*dN
        (;WL=U*(K.*(U'*dleft*U))*U',WR=V*(K.*(V'*dright*V))*V')
    end
    (;WL,WR,derivative,ratio)
end

# The four fields are raw right/left (smooth RMS denominator), then
# whitened right/left (exact denominator used by bv_residual).
function mw_pencil(H,N,r,l)
    Hr,Nr=H*r,N*r;Hl,Nl=H'*l,N'*l
    ht,nt=dot(l,Hr),dot(l,Nr)
    min(abs(ht),abs(nt))>1e-12 || error("center contraction vanishes")
    hR,nR=Hr/ht,Nr/nt;hL,nL=Hl/conj(ht),Nl/conj(nt)
    metric=mw_metric(N)
    hs=(hR,hL,metric.WL*hR,metric.WR*hL)
    ns=(nR,nL,metric.WL*nR,metric.WR*nL)
    scales=[i<=2 ? sqrt((norm(hs[i])^2+norm(ns[i])^2)/2) : norm(hs[i]) for i in 1:4]
    minimum(scales)>1e-12 || error("center residual normalization vanishes")
    fields=[(hs[i]-ns[i])/scales[i] for i in 1:4]
    function derivative(dH,dN,dr,dl)
        dHr=dH*r+H*dr;dNr=dN*r+N*dr
        dHl=dH'*l+H'*dl;dNl=dN'*l+N'*dl
        dht=dot(dl,Hr)+dot(l,dHr);dnt=dot(dl,Nr)+dot(l,dNr)
        dhR=dHr/ht-hR*(dht/ht);dnR=dNr/nt-nR*(dnt/nt)
        dhL=dHl/conj(ht)-hL*conj(dht/ht)
        dnL=dNl/conj(nt)-nL*conj(dnt/nt)
        dm=metric.derivative(dN)
        dhs=(dhR,dhL,dm.WL*hR+metric.WL*dhR,dm.WR*hL+metric.WR*dhL)
        dns=(dnR,dnL,dm.WL*nR+metric.WL*dnR,dm.WR*nL+metric.WR*dnL)
        result=Vector{ComplexF64}[]
        for i in 1:4
            ds=i<=2 ? real(dot(hs[i],dhs[i])+dot(ns[i],dns[i]))/(2scales[i]) :
                real(dot(hs[i],dhs[i]))/scales[i]
            push!(result,(dhs[i]-dns[i])/scales[i]-fields[i]*(ds/scales[i]))
        end
        result
    end
    (;fields,derivative,metric,z=ht/nt)
end

# Fix the bond gauge smoothly: C=rho^(1/2)>0. Ordinary canonicalization
# can choose varying right-bond bases, which would pollute a vector FD test.
function mw_state(A;responses=true)
    density=cg_density(A;responses)
    W=density.W;C=inv(W);AC=A*C;AR=MPSKit._mul_front(W,AC)
    state=MPSKit.InfiniteMPS([A],[AR],[C],[AC])
    function derivative(dA)
        d=density.derivative(dA);dC=-C*d.W*C
        dAC=dA*C+A*dC
        dAR=MPSKit._mul_front(d.W,AC)+MPSKit._mul_front(W,dAC)
        (;AL=dA,AR=dAR,AC=dAC,C=dC)
    end
    canonical=max(norm(AC-A*C),norm(AC-MPSKit._mul_front(C,AR)))/norm(AC)
    canonical<1e-10 || error("positive-C chart is not canonical: $canonical")
    (;state,derivative,report=merge(density.report,Dict("canonical_error"=>canonical)))
end

function mw_cache(A,B,O;responses=true,refs=nothing)
    sr,sl=mw_state(A;responses),mw_state(B;responses)
    R,L=sr.state,sl.state
    gl,gr=bv_grow(A,O[1]),bv_grow(R.AR[1],O[1]);P=gl.physical
    rng=MersenneTwister(32201);target(k)=refs===nothing ? nothing : refs[k]
    lc(a,b,k)=bt_cap(x->MPSKit.transfer_left(x,a,b),
        randn(rng,ComplexF64,space(b,1)←space(a,1));target=target(k),responses)
    rc(a,b,k)=bt_cap(x->MPSKit.transfer_right(x,a,b),
        randn(rng,ComplexF64,dual(space(a,4))←dual(space(b,4)));target=target(k),responses)
    lt,rt=lc(gl.D,B,:lt),rc(gr.D,L.AR[1],:rt)
    l0,r0=lc(A,B,:l0),rc(R.AR[1],L.AR[1],:r0)
    nextrefs=Dict(:lt=>lt.z,:rt=>rt.z,:l0=>l0.z,:r0=>r0.z)
    max(abs(lt.z/rt.z-1),abs(l0.z/r0.z-1))<1e-8 || error("mixed cap eigenvalues disagree")
    l=lt.v*gl.UL;r=gr.UR'*rt.v;d=domain(l)
    extension=id(ComplexF64,d[2])⊗id(ComplexF64,d[3])
    Hac=x->(lt.v⊗P)*bv_grow(x,O[1]).D*rt.v
    Nac=x->(l0.v⊗P)*x*r0.v
    Hc=x->l*(x⊗extension)*r;Nc=x->l0.v*x*r0.v
    ac=(H=bv_matrix(Hac,R.AC[1]),N=bv_matrix(Nac,R.AC[1]))
    c=(H=bv_matrix(Hc,R.C[1]),N=bv_matrix(Nc,R.C[1]))
    fac=mw_pencil(ac.H,ac.N,R.AC[1].data,L.AC[1].data)
    fc=mw_pencil(c.H,c.N,R.C[1].data,L.C[1].data)
    fields=vcat(fac.fields,fc.fields);field=bn_real(vcat(fields...))
    relation=abs(fac.z/(fc.z*lt.z/l0.z)-1)
    relation<1e-8 || error("positive-C center/row quotient identity failed")
    function action(dA,dB)
        responses || error("mixed-center responses disabled")
        dr,dl=sr.derivative(dA),sl.derivative(dB)
        dgl,dgr=bv_grow(dA,O[1]).D,bv_grow(dr.AR,O[1]).D
        dlt=lt.response(x->MPSKit.transfer_left(x,dgl,B)+MPSKit.transfer_left(x,gl.D,dB))
        drt=rt.response(x->MPSKit.transfer_right(x,dgr,L.AR[1])+MPSKit.transfer_right(x,gr.D,dl.AR))
        dl0=l0.response(x->MPSKit.transfer_left(x,dA,B)+MPSKit.transfer_left(x,A,dB))
        dr0=r0.response(x->MPSKit.transfer_right(x,dr.AR,L.AR[1])+MPSKit.transfer_right(x,R.AR[1],dl.AR))
        dHac=x->(dlt.dv⊗P)*bv_grow(x,O[1]).D*rt.v+(lt.v⊗P)*bv_grow(x,O[1]).D*drt.dv
        dNac=x->(dl0.dv⊗P)*x*r0.v+(l0.v⊗P)*x*dr0.dv
        dleft=dlt.dv*gl.UL;dright=gr.UR'*drt.dv
        dHc=x->dleft*(x⊗extension)*r+l*(x⊗extension)*dright
        dNc=x->dl0.dv*x*r0.v+l0.v*x*dr0.dv
        da=fac.derivative(bv_matrix(dHac,R.AC[1]),bv_matrix(dNac,R.AC[1]),dr.AC.data,dl.AC.data)
        dc=fc.derivative(bv_matrix(dHc,R.C[1]),bv_matrix(dNc,R.C[1]),dr.C.data,dl.C.data)
        (;field=bn_real(vcat(da...,dc...)),cap_response_error=maximum(x.err for x in (dlt,drt,dl0,dr0)))
    end
    reports=Dict(string(k)=>v.report for (k,v) in pairs((;lt,rt,l0,r0)))
    (;R,L,ac,c,fields,field,action,refs=nextrefs,reports,relation,
      density_right=sr.report,density_left=sl.report,q=lt.z/l0.z)
end

function mw_chart(R,L,O;refs=nothing)
    A,B=R.AL[1],L.AL[1];NR,NL=TensorKit.left_null(A),TensorKit.left_null(B)
    tr,tl=zero(NR'*A),zero(NL'*B);nr,nl=length(tr.data),length(tl.data);half=nr+nl
    function unpack(v)
        length(v)==2half || error("mixed-center chart dimension mismatch")
        z=complex.(v[1:half],v[half+1:end])
        TensorMap(copy(z[1:nr]),space(tr)),TensorMap(copy(z[nr+1:end]),space(tl))
    end
    base=mw_cache(A,B,O;refs)
    action=v->begin
        vr,vl=unpack(v);base.action(NR*vr,NL*vl)
    end
    evaluate=v->begin
        vr,vl=unpack(v)
        a=(A+NR*vr)*bn_invsqrt(id(ComplexF64,domain(A))+vr'*vr)
        b=(B+NL*vl)*bn_invsqrt(id(ComplexF64,domain(B))+vl'*vl)
        mw_cache(a,b,O;responses=false,refs=base.refs)
    end
    (;base,field=base.field,action,evaluate,nr,nl,count=2half)
end

function mw_audit(source,out;perturb=false)
    haskey(ENV,"SLURM_JOB_ID") || error("submit via Slurm")
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    files=joinpath.(source,["boundary_1.jls","boundary_3.jls","independent_pair.jls"])
    append!(files,joinpath.(@__DIR__,["mixed_center_response.jl","center_gradient_response.jl",
        "paired_tangent_metric.jl","antiunitary_newton_chart.jl","branch_response.jl",
        "newton_metric_response.jl","newton_response_fast.jl","core.jl"]))
    hashes=Dict(f=>bytes2hex(sha256(read(f))) for f in files)
    report=Dict{String,Any}("complete"=>false,"passed"=>false,"accepted_entropy"=>false,
        "independent_left_right"=>true,"direction_constraint"=>false,"nonstationary"=>true,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>ENV["SLURM_JOB_PARTITION"],
        "source_hashes"=>hashes,"perturbed"=>perturb)
    bv_write(out,report)
    try
        n,s=deserialize.(files[1:2]);pair=deserialize(files[3]);R,L=deepcopy(pair.R),deepcopy(pair.L)
        if perturb
            maps=ba_pair_maps(n,s);states=[]
            for (i,state,K) in ((1,R,maps.right),(2,L,maps.left))
                symmetry=at_symmetry(state,K)
                a=state.AL[1]+0.002randn(MersenneTwister(32210+i),ComplexF64,space(state.AL[1]))
                a=(a+symmetry.apply(a))/2;a=a*bn_invsqrt(a'*a)
                candidate,_=bt_canonical(a);push!(states,candidate)
            end
            R,L=states
        end
        chart=mw_chart(R,L,pair.transfer);b=chart.base;original=bv_audit(R,L,pair.transfer)
        norm(chart.field)>1e-7 || error("stationary zero is not a derivative control")
        report["field_norm"]=norm(chart.field);report["original"]=original.report
        # Compare exact WHITE norms with the independently constructed original
        # canonical pencils, including all original bra/graded conventions.
        expected=[original.report[x][y] for x in ("AC","C") for y in ("right_white","left_white")]
        got=norm.(b.fields[[3,4,7,8]])
        identity=norm(got-expected)/norm(expected)
        report["white_identity_error"]=identity;report["white_norms"]=got
        report["original_white_norms"]=expected
        identity<1e-6 || error("mixed whitened field differs from original pencils: $identity")
        origin=chart.evaluate(zeros(chart.count))
        oe=norm(origin.field-chart.field)/norm(chart.field);report["origin_error"]=oe
        oe<1e-8 || error("mixed-center chart origin inconsistent")
        rows=[];report["derivative_checks"]=rows;rng=MersenneTwister(32221);half=div(chart.count,2)
        for side in ("right","left"),quadrature in ("real","imag")
            v=zeros(chart.count);indices=side=="right" ? (1:chart.nr) : (chart.nr+1:half)
            offset=quadrature=="real" ? 0 : half
            v[offset .+ indices]=randn(rng,length(indices));v/=norm(v)
            j=chart.action(v);stencils=[]
            for h in (1e-4,3e-5,1e-5,3e-6)
                p,m=chart.evaluate(h*v),chart.evaluate(-h*v)
                fd=(p.field-m.field)/(2h)
                push!(stencils,Dict("step"=>h,"relative_error"=>norm(fd-j.field)/norm(j.field)))
            end
            push!(rows,Dict("side"=>side,"quadrature"=>quadrature,"stencils"=>stencils,
                "cap_response_error"=>j.cap_response_error));bv_write(out,report)
            minimum(t["relative_error"] for t in stencils)<1e-4 || error("mixed-center derivative mismatch")
        end
        u=randn(rng,chart.count);u/=norm(u);v=randn(rng,chart.count);v/=norm(v)
        ju,jv=chart.action(u).field,chart.action(v).field
        linear=norm(chart.action(u+v).field-ju-jv)/max(norm(ju),norm(jv))
        report["real_linearity_error"]=linear;linear<1e-8 || error("response not real linear")
        report["passed"]=true
    catch err
        report["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(f)))==h for (f,h) in hashes) || error("source changed")
        report["complete"]=true;bv_write(out,report)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    mw_audit(ARGS[1],ARGS[2];perturb=length(ARGS)>2 && parse(Bool,ARGS[3]))
end
