# Frozen-state audit of an invertible preconditioner from the local mixed
# tangent overlap. This local form is not the complete uniform-MPS fidelity
# metric, nor a biorthogonal canonical gauge for the entire boundary MPS.
include("antiunitary_newton_chart.jl")

function pt_weights(R,L;refs=nothing)
    A,B=R.AL[1],L.AL[1]
    NR,NL=TensorKit.left_null(A),TensorKit.left_null(B)
    tr,tl=zero(NR'*A),zero(NL'*B)
    rng=MersenneTwister(31401)
    target(k)=refs===nothing ? nothing : refs[k]
    l=bt_cap(x->MPSKit.transfer_left(x,A,B),
        randn(rng,ComplexF64,space(B,1)←space(A,1));target=target(:l0),responses=false)
    r=bt_cap(x->MPSKit.transfer_right(x,A,B),
        randn(rng,ComplexF64,dual(space(A,4))←dual(space(B,4)));target=target(:r0),responses=false)
    physical=id(ComplexF64,space(A,2))⊗id(ComplexF64,space(A,3))
    apply=X->(l.v⊗physical)*X*r.v
    denominator=dot(B,apply(A))
    abs(denominator)>1e-12 || error("local mixed overlap vanishes")
    M=bv_matrix(v->NL'*apply(NR*v)/denominator,tr)
    size(M,1)==size(M,2) || error("unequal independent tangent dimensions")
    F=svd(M);ratio=minimum(F.S)/maximum(F.S)
    ratio>1e-12 || error("local mixed tangent form rank deficient: $ratio; no rank cut")
    X=F.V*Diagonal(1 ./ sqrt.(F.S));Y=F.U*Diagonal(1 ./ sqrt.(F.S))
    # Positive normalizers preserve the original right/left coordinate frames.
    # Their mixed form is unitary. X,Y are the associated dual whitening bases.
    WR=X*F.V';WL=Y*F.U'
    white_error=norm(Y'*M*X-I)/sqrt(size(M,1))
    polar=WL'*M*WR
    unitary_error=norm(polar'*polar-I)/sqrt(size(M,1))
    max(white_error,unitary_error)<1e-8 || error("local mixed whitening inaccurate")
    nr,nl=length(tr.data),length(tl.data);count=nr+nl
    function pushforward(v)
        length(v)==2count || error("wrong real tangent dimension")
        z=complex.(v[1:count],v[count+1:end])
        bn_real(vcat(WR*z[1:nr],WL*z[nr+1:end]))
    end
    report=Dict("local_form_only"=>true,"rank_truncated"=>false,
        "singular_ratio"=>ratio,"singular_min"=>minimum(F.S),"singular_max"=>maximum(F.S),
        "dual_whitening_error"=>white_error,"normalized_form_unitarity_error"=>unitary_error,
        "normalizer_operator_norm"=>1/sqrt(minimum(F.S)),
        "right_normalizer_hermiticity"=>norm(WR-WR')/norm(WR),
        "left_normalizer_hermiticity"=>norm(WL-WL')/norm(WL),
        "left_cap"=>l.report,"right_cap"=>r.report)
    (;M,WR,WL,pushforward,report,tr,tl)
end

function pt_chart(R,L,O;refs=nothing)
    base=bt_chart(R.AL[1],L.AL[1],O;refs)
    w=pt_weights(R,L;refs=base.base.refs);P=w.pushforward
    function evaluate(v)
        p=base.evaluate(P(v))
        (;R=p.R,L=p.L,q=p.q,refs=p.refs,reports=p.reports,gradient=P(p.gradient))
    end
    (;gradient=P(base.gradient),action=v->P(base.action(P(v))),evaluate,
      pushforward=P,base=base.base,count=base.count,weights=w)
end

function pt_audit(source,snapshot,out;perturb=false)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    files=[joinpath(source,"boundary_1.jls"),joinpath(source,"boundary_3.jls"),snapshot]
    append!(files,[joinpath(@__DIR__,f) for f in ("paired_tangent_metric.jl",
        "antiunitary_newton_chart.jl","branch_response.jl","newton_metric_response.jl",
        "newton_response_fast.jl","core.jl")])
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in files)
    meta=Dict{String,Any}("complete"=>false,"passed"=>false,"diagnostic_only"=>true,
        "accepted_entropy"=>false,"independent_left_right"=>true,"direction_constraint"=>false,
        "source_hashes"=>hashes,"job_id"=>ENV["SLURM_JOB_ID"],"perturbed"=>perturb)
    bv_write(out,meta)
    try
        n,s=deserialize.(files[1:2]);pair=deserialize(snapshot);maps=ba_pair_maps(n,s)
        R,L=deepcopy(pair.R),deepcopy(pair.L)
        refs=hasproperty(pair,:refs) ? pair.refs : nothing
        if perturb
            states=[]
            for (i,state,K) in ((1,R,maps.right),(2,L,maps.left))
                symmetry=at_symmetry(state,K)
                raw=state.AL[1]+0.002randn(MersenneTwister(31410+i),ComplexF64,space(state.AL[1]))
                raw=(raw+symmetry.apply(raw))/2;raw=raw*bn_invsqrt(raw'*raw)
                candidate,_=bt_canonical(raw);push!(states,candidate)
            end
            R,L=states;refs=nothing
        end
        symmetry=at_tangent(R,L,maps);S,Q=symmetry.S,symmetry.Q
        paired=pt_chart(R,L,pair.transfer;refs)
        ordinary=btm_chart(R,L,pair.transfer;refs)
        norm(paired.gradient)>1e-7 || error("stationary zero is not a derivative control")
        meta["nonstationary"]=true
        meta["paired_metric"]=paired.weights.report;meta["ordinary_metric"]=ordinary.weights
        P=hcat((paired.pushforward(v) for v in eachcol(Matrix{Float64}(I,paired.count,paired.count)))...)
        commutator=norm(S*P-P*S)/norm(P)
        meta["symmetry_commutator"]=commutator
        meta["coordinate_self_adjoint_error"]=norm(P-P')/norm(P)
        for (name,c) in (("paired",paired),("ordinary",ordinary))
            g=c.gradient
            meta[name*"_gradient"]=Dict("norm"=>norm(g),
                "outside_absolute"=>norm(g-Q*(Q'*g)),
                "outside_fraction"=>norm(g-Q*(Q'*g))/max(norm(g),eps()))
        end
        max(commutator,meta["coordinate_self_adjoint_error"])<1e-8 || error("paired coordinates violate audited symmetry")
        # In the equal-boundary case the local overlap form is exactly the
        # ordinary right Schmidt density. Compare BEFORE inverting small values.
        diagonal=pt_weights(R,R);rho=R.C[1]*R.C[1]'/norm(R.C[1])^2
        expected=bv_matrix(v->v*rho,diagonal.tr)
        same_error=norm(diagonal.M-expected)/norm(expected)
        meta["equal_boundary_density_error"]=same_error
        same_error<1e-9 || error("equal-boundary overlap form differs from Schmidt metric")
        rng=MersenneTwister(31417);rows=Dict{String,Any}[]
        for sample in 1:3
            v=Q*randn(rng,size(Q,2));v/=norm(paired.pushforward(v))
            hv=paired.action(v);predicted=dot(paired.gradient,v);stencils=[]
            for h in (1e-4,3e-5,1e-5)
                p=paired.evaluate(h*v);m=paired.evaluate(-h*v)
                scalar=real((p.q-m.q)/(2h*paired.base.q))
                fd=(p.gradient-m.gradient)/(2h)
                push!(stencils,Dict("step"=>h,"scalar_error"=>abs(scalar-predicted),
                    "hessian_relative_error"=>norm(fd-hv)/max(norm(hv),eps())))
            end
            push!(rows,Dict("sample"=>sample,"predicted_scalar"=>predicted,"stencils"=>stencils))
            minimum(t["scalar_error"] for t in stencils)<1e-7 || error("paired scalar derivative mismatch")
            minimum(t["hessian_relative_error"] for t in stencils)<1e-4 || error("paired Hessian derivative mismatch")
        end
        meta["derivative_checks"]=rows;meta["passed"]=true
    catch err
        meta["error"]=sprint(showerror,err)
        rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        meta["complete"]=true;bv_write(out,meta)
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    pt_audit(ARGS[1],ARGS[2],ARGS[3];perturb=length(ARGS)>3 && parse(Bool,ARGS[4]))
end
