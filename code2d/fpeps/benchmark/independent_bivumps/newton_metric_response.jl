# A reversible change of tangent coordinates for the validated independent
# analytic Newton equations. Each state supplies its OWN Schmidt metric.
include("newton_response_fast.jl")

function nm_weight(C;power=0.5)
    rho=C*C';W=zero(rho);values=Float64[]
    for (q,b) in blocks(rho)
        e=eigen(Hermitian((b+b')/2));append!(values,e.values)
        minimum(e.values)>0 || error("Schmidt metric is rank deficient; no silent floor")
        block(W,q).=e.vectors*Diagonal(e.values.^(-power))*e.vectors'
    end
    (;W,minimum=minimum(values),maximum=maximum(values),power)
end

function nm_chart(R,L,O;power=0.5)
    base=bn_chart(R.AL[1],L.AL[1],O)
    wr=nm_weight(R.C[1];power);wl=nm_weight(L.C[1];power)
    tr=zero(TensorKit.left_null(R.AL[1])'*R.AL[1])
    tl=zero(TensorKit.left_null(L.AL[1])'*L.AL[1])
    function pushforward(v)
        count=div(base.count,2)
        length(v)==base.count || error("wrong real tangent dimension")
        z=complex.(v[1:count],v[count+1:end])
        vr=TensorMap(copy(z[1:base.nr]),space(tr))
        vl=TensorMap(copy(z[base.nr+1:end]),space(tl))
        bn_real(vcat((vr*wr.W).data,(vl*wl.W).data))
    end
    # W is Hermitian, hence the real-coordinate Jacobian P is self-adjoint.
    # Hold P fixed within a trial: g_y=P' g_x, H_y=P' H_x P.
    gradient=pushforward(base.gradient)
    action=v->pushforward(base.action(pushforward(v)))
    function evaluate(v)
        p=base.evaluate(pushforward(v))
        (;R=p.R,L=p.L,q=p.q,gradient=pushforward(p.gradient))
    end
    weights=Dict("right_density_min"=>wr.minimum,"right_density_max"=>wr.maximum,
        "left_density_min"=>wl.minimum,"left_density_max"=>wl.maximum,
        "inverse_density_power"=>power,"rank_truncated"=>false)
    (;R,L,O,count=base.count,nr=base.nr,nl=base.nl,gradient,action,evaluate,
      checks=base.checks,base=base.base,pushforward,weights)
end

function nm_audit(source,out)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");mkpath(out);BLAS.set_num_threads(2)
    n,s=deserialize.(joinpath.(source,("boundary_1.jls","boundary_3.jls")))
    rng=MersenneTwister(2412)
    A=n.state.AL[1];B=MPSKit.InfiniteMPS([FermionicPEPS._spatial_boundary_bra(s.state.AR[1])]).AL[1]
    A+=0.002randn(rng,ComplexF64,space(A));B+=0.003randn(rng,ComplexF64,space(B))
    R=MPSKit.InfiniteMPS([A];tol=1e-13,maxiter=1000)
    L=MPSKit.InfiniteMPS([B];tol=1e-13,maxiter=1000)
    c=nm_chart(R,L,n.transfer);directions=[];rows=Dict{String,Any}[]
    norm(c.evaluate(zeros(c.count)).gradient-c.gradient)/max(norm(c.gradient),eps())<1e-8 || error("metric origin mismatch")
    for side in ("right","left"),quadrature in ("real","imag")
        v=zeros(c.count);offset=quadrature=="real" ? 0 : div(c.count,2)
        inds=side=="right" ? (1:c.nr) : (c.nr+1:c.nr+c.nl)
        v[offset .+ inds].=randn(rng,length(inds));v/=norm(c.pushforward(v));push!(directions,v)
        hv=c.action(v);predicted=dot(c.gradient,v);stencils=[]
        for h in (1e-4,3e-5,1e-5)
            p=c.evaluate(h*v);m=c.evaluate(-h*v)
            scalar=real((p.q-m.q)/(2h*c.base.q));fd=(p.gradient-m.gradient)/(2h)
            push!(stencils,Dict("step"=>h,"scalar_error"=>abs(scalar-predicted),
                "hessian_relative_error"=>norm(fd-hv)/max(norm(hv),eps())))
        end
        abs(predicted)>1e-7 || error("stationary control direction")
        minimum(t["scalar_error"] for t in stencils)<1e-7 || error("metric scalar derivative failed")
        minimum(t["hessian_relative_error"] for t in stencils)<1e-4 || error("metric Hessian derivative failed")
        push!(rows,Dict("side"=>side,"quadrature"=>quadrature,"predicted"=>predicted,"stencils"=>stencils))
    end
    u,v=directions[1],directions[4];hu,hv=c.action(u),c.action(v)
    symmetry=abs(dot(u,hv)-dot(v,hu))/max(norm(hu)*norm(v),norm(hv)*norm(u),eps())
    linearity=norm(c.action(u+v)-hu-hv)/max(norm(hu),norm(hv),eps())
    duality=abs(dot(u,c.pushforward(v))-dot(c.pushforward(u),v))/max(norm(u)*norm(c.pushforward(v)),eps())
    max(symmetry,linearity,duality)<1e-8 || error("metric Hessian/duality control failed")
    bv_write(joinpath(out,"checks.toml"),Dict("complete"=>true,"passed"=>true,
        "independent_left_right"=>true,"direction_constraint"=>false,"checks"=>rows,
        "hessian_symmetry_error"=>symmetry,"hessian_linearity_error"=>linearity,
        "coordinate_duality_error"=>duality,"weights"=>c.weights,
        "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))),
        "base_response_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"newton_response_fast.jl")))),
        "response_sha256"=>bytes2hex(sha256(read(@__FILE__)))))
    println("independent Schmidt-coordinate response audit passed");flush(stdout)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    nm_audit(ARGS...)
end
