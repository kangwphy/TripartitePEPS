# Derivatives at nonzero points of ONE fixed MPS retraction chart.
# Needed to differentiate projected gradients without mixing moving gauges.
# No existing solver/response method is overridden.
include("replay_mixed_vector_step.jl")

function ca_chart(R,L,O;refs=nothing)
    A,B=R.AL[1],L.AL[1];NR,NL=TensorKit.left_null(A),TensorKit.left_null(B)
    tr,tl=zero(NR'*A),zero(NL'*B);nr,nl=length(tr.data),length(tl.data);half=nr+nl
    function unpack(v)
        length(v)==2half || error("anchored coordinate dimension mismatch")
        z=complex.(v[1:half],v[half+1:end])
        TensorMap(copy(z[1:nr]),space(tr)),TensorMap(copy(z[nr+1:end]),space(tl))
    end
    base=mg_cache(A,B,O;refs)
    function point(v;responses=true)
        t,u=unpack(v)
        M=id(ComplexF64,domain(A))+t'*t;N=id(ComplexF64,domain(B))+u'*u
        W,Z=bn_invsqrt(M),bn_invsqrt(N)
        rawA,rawB=A+NR*t,B+NL*u;a,b=rawA*W,rawB*Z
        cache=mg_cache(a,b,O;responses,refs=base.refs)
        function action(w)
            responses || error("anchored response disabled")
            dt,du=unpack(w)
            dW=bn_frechet(M,dt'*t+t'*dt);dZ=bn_frechet(N,du'*u+u'*du)
            da,db=NR*dt*W+rawA*dW,NL*du*Z+rawB*dZ
            er=norm(a'*da+da'*a)/max(norm(da),eps())
            el=norm(b'*db+db'*b)/max(norm(db),eps())
            max(er,el)<1e-10 || error("anchored derivative leaves isometric manifold")
            response=cache.action(da,db)
            (;response...,right_isometric_tangent_error=er,left_isometric_tangent_error=el)
        end
        (;cache,action,field=cache.field)
    end
    (;base,point,nr,nl,count=2half)
end

function ca_input(geometry,snapshot)
    reportpath=joinpath(geometry,"report.toml");r=TOML.parsefile(reportpath)
    r["complete"] && r["passed"] && r["diagnostic_only"] && r["chi"]==16 || error("geometry diagnosis incomplete")
    hashes=copy(r["source_hashes"])
    for (p,h) in hashes
        bytes2hex(sha256(read(p)))==h || error("stale geometry source: $p")
    end
    pairpath=joinpath(snapshot,"independent_pair.jls")
    get(hashes,pairpath,"")==bytes2hex(sha256(read(pairpath))) || error("geometry does not cover snapshot")
    linpath=joinpath(geometry,"linearization.jls")
    bytes2hex(sha256(read(linpath)))==r["linearization_sha256"] || error("changed linearization")
    lin=deserialize(linpath);lin.snapshot_pair_sha256==hashes[pairpath] || error("linearization has another base")
    for p in (reportpath,linpath,@__FILE__)
        hashes[p]=bytes2hex(sha256(read(p)))
    end
    (;report=r,pair=deserialize(pairpath),lin,hashes)
end

function ca_audit(geometry,snapshot,out)
    haskey(ENV,"SLURM_JOB_ID") || error("submit via Slurm")
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    input=ca_input(geometry,snapshot);pair=input.pair;lin=input.lin;hashes=input.hashes
    checks=[];record=Dict{String,Any}("complete"=>false,"passed"=>false,
        "chi"=>16,"diagnostic_only"=>true,"accepted_entropy"=>false,"accepted_curve_point"=>false,
        "zero_optimization_steps"=>true,"source_hashes"=>hashes,"checks"=>checks,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>"preempt")
    bv_write(out,record)
    try
        chart=ca_chart(pair.R,pair.L,pair.transfer);original=mg_chart(pair.R,pair.L,pair.transfer)
        base=chart.point(zeros(chart.count))
        norm(base.field-lin.F)/norm(lin.F)<1e-8 || error("linearization field changed")
        rng=MersenneTwister(32512);half=div(chart.count,2)
        for (label,Q) in (("fixed_offset",lin.Q),("outside_offset",lin.P))
            v=Q*randn(rng,size(Q,2));v*=1e-3/norm(v)
            shifted=chart.point(v)
            same=original.evaluate(v)
            ferr=norm(shifted.field-same.field)/norm(same.field)
            ferr<1e-8 || error("anchored retraction changed field")
            for side in ("right","left"),quadrature in ("real","imag")
                w=zeros(chart.count);indices=side=="right" ? (1:chart.nr) : (chart.nr+1:half)
                offset=quadrature=="real" ? 0 : half
                w[offset .+ indices]=randn(rng,length(indices));w/=norm(w)
                baseerr=norm(base.action(w).field-lin.J*w)/norm(lin.J*w)
                baseerr<1e-8 || error("saved J uses a different coordinate basis")
                response=shifted.action(w);response.cap_response_error<1e-8 || error("inaccurate shifted response")
                stencils=[]
                for h in (1e-5,3e-6)
                    fd=(original.evaluate(v+h*w).field-original.evaluate(v-h*w).field)/(2h)
                    push!(stencils,Dict("h"=>h,"relative_error"=>norm(fd-response.field)/norm(response.field)))
                end
                minimum(s["relative_error"] for s in stencils)<1e-4 || error("nonzero-coordinate derivative failed")
                push!(checks,Dict("offset"=>label,"side"=>side,"quadrature"=>quadrature,
                    "field_relative_error"=>ferr,"saved_J_relative_error"=>baseerr,
                    "cap_response_error"=>response.cap_response_error,
                    "right_isometric_tangent_error"=>response.right_isometric_tangent_error,
                    "left_isometric_tangent_error"=>response.left_isometric_tangent_error,"stencils"=>stencils))
                bv_write(out,record)
            end
        end
        record["passed"]=true
    catch err
        record["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        record["complete"]=true;bv_write(out,record)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    ca_audit(ARGS...)
end
