# Arnoldi for a point-gradient Jacobian, which need not be symmetric away
# from a stationary point. Use the preconditioned operator ONLY to build a
# search space; minimize the original horizontal residual in that space.
include("krylov_trust.jl")
include("ellipsoid_trust.jl")

function pk_basis(action,g;maxbasis=192,forcing=1e-3)
    beta=norm(g);beta>0 || error("zero gradient with unresolved outer equations")
    n=length(g);limit=min(n,maxbasis)
    Q=zeros(n,limit+1);K=zeros(limit+1,limit);Q[:,1]=g/beta
    rows=Dict{String,Any}[];used=0;reason="basis_limit"
    for j in 1:limit
        v=action(Q[:,j]);scale=norm(v)
        all(isfinite,v) || error("nonfinite point Jacobian response")
        for _ in 1:2
            h=Q[:,1:j]'*v;K[1:j,j]+=h;v-=Q[:,1:j]*h
        end
        next=norm(v);K[j+1,j]=next;used=j
        closed=next<=100eps(Float64)*max(scale,eps()) || j==n
        if !closed;Q[:,j+1]=v/next;end
        if j%8==0 || closed || j==limit
            F=svd(K[1:j+1,1:j]);rhs=zeros(j+1);rhs[1]=beta
            y=F.V*[s==0 ? 0.0 : -a/s for (s,a) in zip(F.S,F.U'*rhs)]
            rel=norm(rhs+K[1:j+1,1:j]*y)/beta
            asymmetry=norm(K[1:j,1:j]-K[1:j,1:j]')/max(norm(K[1:j,1:j]),eps())
            push!(rows,Dict("dimension"=>j,"unconstrained_relative_residual"=>rel,
                "projected_asymmetry"=>asymmetry,"next_basis_norm"=>next))
            println("  Point Jacobian Arnoldi dimension=",j," linear residual=",rel);flush(stdout)
            if closed;reason="closed_subspace";break;end
            if rel<forcing;reason="inexact_Newton_forcing_met";break;end
        end
    end
    basis=Q[:,1:used];orth=norm(basis'*basis-I)
    orth<1e-9 || error("point Arnoldi orthogonality failed")
    H=K[1:used+1,1:used];rhs=zeros(used+1);rhs[1]=beta
    (;basis,H,rhs,svd=svd(H),report=Dict("dimension"=>used,"maxbasis"=>maxbasis,
        "forcing"=>forcing,"stop_reason"=>reason,"orthogonality_error"=>orth,
        "symmetry_assumed"=>false,"rows"=>rows))
end

function pk_raw_model(b,images,g)
    length(images)==size(b.basis,2) || error("Jacobian images do not match Arnoldi basis")
    H=hcat(images...)
    size(H,1)==length(g) || error("raw gradient/image dimension mismatch")
    (;basis=b.basis,H,rhs=g)
end

function pk_control()
    # Deliberately nonnormal, rectangular raw least squares after restricting
    # the independent coordinates. No physical-state model enters this test.
    rng=MersenneTwister(31607);n=13;m=10
    U=Matrix(qr(randn(rng,n,n)).Q);V=Matrix(qr(randn(rng,n,n)).Q)
    J=U*(Diagonal(range(-2,3;length=n))+3triu(randn(rng,n,n),1))*V'
    P=V*Diagonal(10.0.^range(-0.7,0.7;length=n))*V'
    Q=Matrix(qr(randn(rng,n,m)).Q)[:,1:m];g=randn(rng,n)
    images=Vector{Float64}[]
    action=v->begin
        j=J*(P*(Q*v));push!(images,j);Q'*(P*j)
    end
    b=pk_basis(action,Q'*(P*g);maxbasis=m,forcing=0.0)
    raw=pk_raw_model(b,images,g);geometry=be_geometry(raw,v->P*(Q*v))
    D=J*P*Q;rows=[]
    image_error=norm(raw.H-D*b.basis)/norm(raw.H)
    image_error<1e-12 || error("raw Arnoldi image alignment failed")
    for radius in (0.01,0.2,5.0)
        rawradius=0.5;got=be_step(raw,geometry,radius;rawradius)
        G=I/radius^2+Q'*P'*P*Q/rawradius^2
        normal=D'*D;rhs=-D'*g;metric(v)=sqrt(real(dot(v,G*v)))
        reference=normal\rhs;mu=0.0
        if metric(reference)>1
            hi=1.0;lo=0.0
            while metric((normal+hi*G)\rhs)>1;hi*=10;end
            for _ in 1:100
                mid=(lo+hi)/2
                if metric((normal+mid*G)\rhs)>1;lo=mid;else;hi=mid;end
            end
            mu=hi;reference=(normal+mu*G)\rhs
        end
        err=norm(got.delta-reference)/max(norm(reference),eps())
        kkt=norm(normal*got.delta+mu*G*got.delta-rhs)/norm(rhs)
        predicted_error=abs(got.predicted_residual-norm(g+D*got.delta))/norm(g)
        max(err,kkt,predicted_error)<1e-8 || error("nonnormal raw-merit trust control failed")
        push!(rows,Dict("radius"=>radius,"step_relative_error"=>err,
            "KKT_relative_error"=>kkt,"predicted_residual_error"=>predicted_error,
            "metric_norm"=>metric(got.delta),"tensor_step_norm"=>norm(P*Q*got.delta)))
    end
    nonnormality=norm(J'*J-J*J')/norm(J)^2
    nonnormality>0.1 || error("control is not sufficiently nonnormal")
    Dict("passed"=>true,"raw_merit"=>true,"symmetry_assumed"=>false,
        "nonnormality"=>nonnormality,"image_alignment_error"=>image_error,
        "nonnormal_rectangular_control"=>rows)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    using LinearAlgebra,Random,TOML
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(ARGS[1]) && error("refusing overwrite")
    report=pk_control()
    report["complete"]=true;report["job_id"]=ENV["SLURM_JOB_ID"]
    open(ARGS[1],"w") do io;TOML.print(io,report);end
end
