# Experimental exact vector actions for dH; no existing mw_/ma_ methods
# are replaced. G is built from the original graded grow map, so its
# adjoint includes the same basis/fermionic conventions as the dense H.
include("audit_mixed_metric_actions.jl")

function mg_pencil(H,N,r,l)
    Hr,Nr=H*r,N*r;Hl,Nl=H'*l,N'*l
    ht,nt=dot(l,Hr),dot(l,Nr)
    min(abs(ht),abs(nt))>1e-12 || error("center contraction vanishes")
    hR,nR=Hr/ht,Nr/nt;hL,nL=Hl/conj(ht),Nl/conj(nt)
    metric=ma_metric(N)
    hs=(hR,hL,metric.WL*hR,metric.WR*hL)
    ns=(nR,nL,metric.WL*nR,metric.WR*nL)
    scales=[i<=2 ? sqrt((norm(hs[i])^2+norm(ns[i])^2)/2) : norm(hs[i]) for i in 1:4]
    minimum(scales)>1e-12 || error("center residual normalization vanishes")
    fields=[(hs[i]-ns[i])/scales[i] for i in 1:4]
    function derivative(dHright,dHleft,dN,dr,dl)
        dHr=dHright+H*dr;dNr=dN*r+N*dr
        dHl=dHleft+H'*dl;dNl=dN'*l+N'*dl
        dht=dot(dl,Hr)+dot(l,dHr);dnt=dot(dl,Nr)+dot(l,dNr)
        dhR=dHr/ht-hR*(dht/ht);dnR=dNr/nt-nR*(dnt/nt)
        dhL=dHl/conj(ht)-hL*conj(dht/ht)
        dnL=dNl/conj(nt)-nL*conj(dnt/nt)
        left,right=metric.actions(dN,hcat(hR,nR),hcat(hL,nL))
        dhs=(dhR,dhL,left[:,1]+metric.WL*dhR,right[:,1]+metric.WR*dhL)
        dns=(dnR,dnL,left[:,2]+metric.WL*dnR,right[:,2]+metric.WR*dnL)
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

function mg_cache(A,B,O;responses=true,refs=nothing)
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
    fac=mg_pencil(ac.H,ac.N,R.AC[1].data,L.AC[1].data)
    fc=mg_pencil(c.H,c.N,R.C[1].data,L.C[1].data)
    fields=vcat(fac.fields,fc.fields);field=bn_real(vcat(fields...))
    relation=abs(fac.z/(fc.z*lt.z/l0.z)-1)
    relation<1e-8 || error("positive-C center/row quotient identity failed")
    grown_ac=responses ? bv_grow(R.AC[1],O[1]).D : nothing
    lifted_c=responses ? R.C[1]⊗extension : nothing
    Gac=responses ? bv_matrix(x->bv_grow(x,O[1]).D,R.AC[1]) : nothing
    Gc=responses ? bv_matrix(x->x⊗extension,R.C[1]) : nothing
    checks=Dict{String,Any}()
    if responses
        checks["AC_forward_relative_error"]=norm(Gac*R.AC[1].data-grown_ac.data)/norm(grown_ac)
        checks["C_forward_relative_error"]=norm(Gc*R.C[1].data-lifted_c.data)/norm(lifted_c)
        hadj_ac=Gac'*((lt.v⊗P)'*L.AC[1]*rt.v').data
        hadj_c=Gc'*(l'*L.C[1]*r').data
        checks["AC_adjoint_relative_error"]=norm(hadj_ac-ac.H'*L.AC[1].data)/norm(ac.H'*L.AC[1].data)
        checks["C_adjoint_relative_error"]=norm(hadj_c-c.H'*L.C[1].data)/norm(c.H'*L.C[1].data)
        maximum(values(checks))<1e-10 || error("graded lifting/adjoint identity failed: $checks")
        checks["lift_bytes"]=sizeof(Gac)+sizeof(Gc)
    end
    function action(dA,dB)
        responses || error("mixed-center responses disabled")
        dr,dl=sr.derivative(dA),sl.derivative(dB)
        dgl,dgr=bv_grow(dA,O[1]).D,bv_grow(dr.AR,O[1]).D
        dlt=lt.response(x->MPSKit.transfer_left(x,dgl,B)+MPSKit.transfer_left(x,gl.D,dB))
        drt=rt.response(x->MPSKit.transfer_right(x,dgr,L.AR[1])+MPSKit.transfer_right(x,gr.D,dl.AR))
        dl0=l0.response(x->MPSKit.transfer_left(x,dA,B)+MPSKit.transfer_left(x,A,dB))
        dr0=r0.response(x->MPSKit.transfer_right(x,dr.AR,L.AR[1])+MPSKit.transfer_right(x,R.AR[1],dl.AR))
        # Only dH*r and dH'*l are needed. The original grow map is fixed
        # during this derivative, while both transfer caps vary.
        dHacR=((dlt.dv⊗P)*grown_ac*rt.v+(lt.v⊗P)*grown_ac*drt.dv).data
        dHacL=Gac'*((dlt.dv⊗P)'*L.AC[1]*rt.v'+(lt.v⊗P)'*L.AC[1]*drt.dv').data
        dNac=x->(dl0.dv⊗P)*x*r0.v+(l0.v⊗P)*x*dr0.dv
        dleft=dlt.dv*gl.UL;dright=gr.UR'*drt.dv
        dHcR=(dleft*lifted_c*r+l*lifted_c*dright).data
        dHcL=Gc'*(dleft'*L.C[1]*r'+l'*L.C[1]*dright').data
        dNc=x->dl0.dv*x*r0.v+l0.v*x*dr0.dv
        da=fac.derivative(dHacR,dHacL,bv_matrix(dNac,R.AC[1]),dr.AC.data,dl.AC.data)
        dc=fc.derivative(dHcR,dHcL,bv_matrix(dNc,R.C[1]),dr.C.data,dl.C.data)
        (;field=bn_real(vcat(da...,dc...)),cap_response_error=maximum(x.err for x in (dlt,drt,dl0,dr0)))
    end
    reports=Dict(string(k)=>v.report for (k,v) in pairs((;lt,rt,l0,r0)))
    (;R,L,ac,c,fields,field,action,refs=nextrefs,reports,relation,lift_checks=checks,
      density_right=sr.report,density_left=sl.report,q=lt.z/l0.z)
end

function mg_chart(R,L,O;refs=nothing)
    A,B=R.AL[1],L.AL[1];NR,NL=TensorKit.left_null(A),TensorKit.left_null(B)
    tr,tl=zero(NR'*A),zero(NL'*B);nr,nl=length(tr.data),length(tl.data);half=nr+nl
    function unpack(v)
        length(v)==2half || error("mixed-center chart dimension mismatch")
        z=complex.(v[1:half],v[half+1:end])
        TensorMap(copy(z[1:nr]),space(tr)),TensorMap(copy(z[nr+1:end]),space(tl))
    end
    base=mg_cache(A,B,O;refs)
    action=v->begin
        vr,vl=unpack(v);base.action(NR*vr,NL*vl)
    end
    evaluate=v->begin
        vr,vl=unpack(v)
        a=(A+NR*vr)*bn_invsqrt(id(ComplexF64,domain(A))+vr'*vr)
        b=(B+NL*vl)*bn_invsqrt(id(ComplexF64,domain(B))+vl'*vl)
        mg_cache(a,b,O;responses=false,refs=base.refs)
    end
    (;base,field=base.field,action,evaluate,nr,nl,count=2half)
end

function mg_audit(source,out,controlpath)
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    check=TOML.parsefile(controlpath)
    check["complete"] && check["passed"] && check["chi"]==16 || error("full-action control not passed")
    hashes=copy(check["source_hashes"])
    for (p,h) in hashes
        bytes2hex(sha256(read(p)))==h || error("stale source: $p")
    end
    for p in joinpath.(source,["boundary_1.jls","boundary_3.jls","independent_pair.jls"])
        get(hashes,p,"")==bytes2hex(sha256(read(p))) || error("control does not cover source")
    end
    hashes[controlpath]=bytes2hex(sha256(read(controlpath)))
    hashes[@__FILE__]=bytes2hex(sha256(read(@__FILE__)))
    n=deserialize(joinpath(source,"boundary_1.jls"));sum(n.chi)==16 || error("chi16 required")
    pair=deserialize(joinpath(source,"independent_pair.jls"));rows=[]
    report=Dict{String,Any}("complete"=>false,"passed"=>false,"chi"=>16,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>ENV["SLURM_JOB_PARTITION"],
        "source_hashes"=>hashes,"zero_optimization_steps"=>true,
        "accepted_entropy"=>false,"diagnostic_only"=>true,
        "independent_left_right"=>true,"direction_constraint"=>false,"rows"=>rows)
    bv_write(out,report)
    try
        slow=mw_chart(pair.R,pair.L,pair.transfer)
        fast=mg_chart(pair.R,pair.L,pair.transfer)
        baseerr=norm(slow.field-fast.field)/norm(slow.field)
        report["field_relative_error"]=baseerr;report["lift_checks"]=fast.base.lift_checks
        baseerr<1e-8 || error("full base field changed")
        bv_write(out,report)
        rng=MersenneTwister(32418);half=div(fast.count,2)
        for side in ("right","left"),quadrature in ("real","imag")
            v=zeros(fast.count);indices=side=="right" ? (1:fast.nr) : (fast.nr+1:half)
            offset=quadrature=="real" ? 0 : half
            v[offset .+ indices]=randn(rng,length(indices));v/=norm(v)
            old=slow.action(v);new=fast.action(v)
            err=norm(old.field-new.field)/norm(old.field)
            err<1e-8 || error("full derivative changed: $err")
            max(old.cap_response_error,new.cap_response_error)<1e-8 || error("cap response inaccurate")
            row=Dict{String,Any}("side"=>side,"quadrature"=>quadrature,
                "derivative_relative_error"=>err,"cap_response_error"=>new.cap_response_error,
                "stencils"=>[])
            push!(rows,row)
            for h in (1e-5,3e-6)
                p,m=slow.evaluate(h*v),slow.evaluate(-h*v)
                fastp=fast.evaluate(h*v)
                evaluatediff=norm(fastp.field-p.field)/norm(p.field)
                evaluatediff<1e-8 || error("retracted field changed")
                fd=(p.field-m.field)/(2h)
                push!(row["stencils"],Dict("h"=>h,"relative_error"=>norm(fd-new.field)/norm(new.field),
                    "retracted_field_relative_error"=>evaluatediff))
            end
            minimum(x["relative_error"] for x in row["stencils"])<1e-4 || error("full finite difference failed")
            if side=="right" && quadrature=="real"
                row["original_seconds"]=ma_time(()->slow.action(v))
                row["vector_seconds"]=ma_time(()->fast.action(v))
            end
            bv_write(out,report)
        end
        u=randn(rng,fast.count);u/=norm(u);v=randn(rng,fast.count);v/=norm(v)
        ju,jv=fast.action(u).field,fast.action(v).field
        linear=norm(fast.action(u+v).field-ju-jv)/max(norm(ju),norm(jv))
        report["real_linearity_error"]=linear
        linear<1e-8 || error("full response not real linear")
        report["passed"]=true
    catch err
        report["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        report["complete"]=true;bv_write(out,report)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    mg_audit(ARGS[1],ARGS[2],ARGS[3])
end
