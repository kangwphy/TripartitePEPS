# Derivative of the horizontal gradient AT the candidate, expressed in a
# smooth orthonormal complement. It differs away from a root from the Hessian
# of the scalar pulled back to the fixed polar-retraction coordinates.
include("paired_tangent_metric.jl")

function pg_chart(R,L,O;refs=nothing)
    A,B=R.AL[1],L.AL[1];base=bt_chart(A,B,O;refs)
    NR,NL=TensorKit.left_null(A),TensorKit.left_null(B)
    tr,tl=zero(NR'*A),zero(NL'*B);nr,nl=length(tr.data),length(tl.data);count=nr+nl
    function unpack(v)
        length(v)==2count || error("wrong point-gradient dimension")
        z=complex.(v[1:count],v[count+1:2count])
        TensorMap(copy(z[1:nr]),space(tr)),TensorMap(copy(z[nr+1:end]),space(tl))
    end
    KR=A'*base.base.GR;KL=B'*base.base.GL
    function action(v)
        VR,VL=unpack(v);d=base.base.derivative(NR*VR,NL*VL)
        bn_real(vcat((NR'*d.dGR-VR*KR).data,(NL'*d.dGL-VL*KL).data))
    end
    function evaluate(v)
        t=base.tensors(v);cache=bt_cache(t.R,t.L,O;responses=false,refs=base.base.refs)
        # Complement columns rotate with the point, with no change of their norm.
        PR=(NR-A*t.VR')*bn_invsqrt(id(ComplexF64,codomain(t.VR))+t.VR*t.VR')
        PL=(NL-B*t.VL')*bn_invsqrt(id(ComplexF64,codomain(t.VL))+t.VL*t.VL')
        frame_error=max(norm(t.R'*PR),norm(t.L'*PL),
            norm(PR'*PR-id(ComplexF64,domain(PR))),norm(PL'*PL-id(ComplexF64,domain(PL))))
        frame_error<1e-10 || error("moving complement is not orthonormal: $frame_error")
        gradient=bn_real(vcat((PR'*cache.GR).data,(PL'*cache.GL).data))
        ambient=hypot(norm(cache.GR-t.R*(t.R'*cache.GR)),norm(cache.GL-t.L*(t.L'*cache.GL)))
        norm_error=abs(norm(gradient)-ambient)/max(ambient,eps())
        norm_error<1e-8 || error("point-gradient norm depends on frame: $norm_error")
        (;R=t.R,L=t.L,q=cache.q,refs=cache.refs,reports=cache.reports,gradient,frame_error,
          ambient_norm_error=norm_error)
    end
    w=pt_weights(R,L;refs=base.base.refs);P=w.pushforward
    scaled_evaluate=v->begin
        p=evaluate(P(v))
        (;R=p.R,L=p.L,q=p.q,refs=p.refs,reports=p.reports,gradient=P(p.gradient),
          raw_gradient=p.gradient,frame_error=p.frame_error,ambient_norm_error=p.ambient_norm_error)
    end
    (;count=2count,nr,nl,NR,NL,unpack,raw_gradient=base.gradient,raw_action=action,
      raw_evaluate=evaluate,gradient=P(base.gradient),action=v->P(action(P(v))),
      evaluate=scaled_evaluate,pushforward=P,weights=w,base=base.base,scalar_chart=base,KR,KL)
end

function pg_audit(source,snapshot,out;perturb=false,proposal=nothing)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    files=[joinpath(source,"boundary_1.jls"),joinpath(source,"boundary_3.jls"),snapshot]
    proposal!==nothing && push!(files,proposal)
    append!(files,[joinpath(@__DIR__,f) for f in ("point_gradient_response.jl",
        "paired_tangent_metric.jl","antiunitary_newton_chart.jl","branch_response.jl",
        "newton_metric_response.jl","newton_response_fast.jl","core.jl")])
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in files)
    meta=Dict{String,Any}("complete"=>false,"passed"=>false,"diagnostic_only"=>true,
        "accepted_entropy"=>false,"independent_left_right"=>true,"direction_constraint"=>false,
        "source_hashes"=>hashes,"job_id"=>ENV["SLURM_JOB_ID"],"perturbed"=>perturb)
    bv_write(out,meta)
    try
        n,s=deserialize.(files[1:2]);pair=deserialize(snapshot);maps=ba_pair_maps(n,s)
        R,L=deepcopy(pair.R),deepcopy(pair.L);refs=hasproperty(pair,:refs) ? pair.refs : nothing
        if perturb
            states=[]
            for (i,state,K) in ((1,R,maps.right),(2,L,maps.left))
                symmetry=at_symmetry(state,K)
                A=state.AL[1]+0.002randn(MersenneTwister(31510+i),ComplexF64,space(state.AL[1]))
                A=(A+symmetry.apply(A))/2;A=A*bn_invsqrt(A'*A)
                candidate,_=bt_canonical(A);push!(states,candidate)
            end
            R,L=states;refs=nothing
        end
        c=pg_chart(R,L,pair.transfer;refs);Q=at_tangent(R,L,maps).Q
        norm(c.raw_gradient)>1e-7 || error("stationary zero is not a derivative control")
        meta["nonstationary"]=true;meta["raw_gradient_norm"]=norm(c.raw_gradient)
        origin=c.evaluate(zeros(c.count))
        meta["origin_error"]=norm(origin.gradient-c.gradient)/norm(c.gradient)
        meta["origin_error"]<1e-7 || error("point-gradient origin differs")
        meta["right_parallel_skew_norm"]=norm(c.KR-c.KR')/2
        meta["left_parallel_skew_norm"]=norm(c.KL-c.KL')/2
        directions=[];rng=MersenneTwister(31519);half=div(c.count,2)
        for side in ("right","left"),sample in 1:2
            v=zeros(c.count);inds=side=="right" ? (1:c.nr) : (c.nr+1:half)
            v[inds]=randn(rng,length(inds));v[half .+ inds]=randn(rng,length(inds));v=Q*(Q'*v)
            v/=norm(c.pushforward(v));push!(directions,(label=side*string(sample),v=v))
        end
        if proposal!==nothing
            trial=deserialize(proposal)
            VR=(c.NR'*trial.R.AL[1])*inv(R.AL[1]'*trial.R.AL[1])
            VL=(c.NL'*trial.L.AL[1])*inv(L.AL[1]'*trial.L.AL[1])
            v=bn_real(vcat(c.weights.WR\VR.data,c.weights.WL\VL.data))
            v/=norm(c.pushforward(v));push!(directions,(label="actual_old_merit_proposal",v=v))
        end
        rows=Dict{String,Any}[];meta["derivative_checks"]=rows
        for direction in directions
            v=direction.v;dx=c.pushforward(v);j=c.action(v);h=c.pushforward(c.scalar_chart.action(dx))
            VR,VL=c.unpack(dx)
            correction=c.pushforward(bn_real(vcat((-VR*(c.KR-c.KR')/2).data,(-VL*(c.KL-c.KL')/2).data)))
            identity=norm(j-h-correction)/max(norm(j),norm(h),eps())
            identity<1e-8 || error("point/scalar derivative difference identity failed")
            stencils=[]
            for step in (1e-4,3e-5,1e-5)
                p=c.evaluate(step*v);m=c.evaluate(-step*v);fd=(p.gradient-m.gradient)/(2step)
                push!(stencils,Dict("step"=>step,"point_jacobian_relative_error"=>norm(fd-j)/max(norm(j),eps()),
                    "old_scalar_hessian_relative_error"=>norm(fd-h)/max(norm(j),eps()),
                    "frame_error"=>max(p.frame_error,m.frame_error),
                    "ambient_norm_error"=>max(p.ambient_norm_error,m.ambient_norm_error)))
            end
            push!(rows,Dict("direction"=>direction.label,"difference_identity_error"=>identity,
                "point_vs_scalar_relative_difference"=>norm(j-h)/max(norm(j),eps()),"stencils"=>stencils))
            minimum(t["point_jacobian_relative_error"] for t in stencils)<1e-4 || error("point-gradient derivative mismatch")
            bv_write(out,meta)
        end
        u,v=directions[1].v,directions[3].v
        meta["jacobian_asymmetry_sample"]=abs(dot(u,c.action(v))-dot(v,c.action(u)))/max(norm(u)*norm(c.action(v)),norm(v)*norm(c.action(u)),eps())
        meta["passed"]=true
    catch err
        meta["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        meta["complete"]=true;bv_write(out,meta)
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    pg_audit(ARGS[1],ARGS[2],ARGS[3];perturb=parse(Bool,ARGS[4]),
        proposal=length(ARGS)>4 ? ARGS[5] : nothing)
end
