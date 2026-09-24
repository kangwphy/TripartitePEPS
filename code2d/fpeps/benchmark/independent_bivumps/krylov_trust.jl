# Matrix-free least-squares Newton trust step. The operator is the already
# validated real analytic Hessian; no physical tensor or parity rank is cut.
function bk_basis(action,g;maxbasis=192,forcing=1e-3)
    beta=norm(g);beta>0 || error("zero gradient with unresolved outer equations")
    n=length(g);limit=min(n,maxbasis)
    Q=zeros(n,limit+1);K=zeros(limit+1,limit);Q[:,1]=g/beta
    rows=Dict{String,Any}[];used=0;reason="basis_limit"
    for j in 1:limit
        v=action(Q[:,j]);scale=norm(v)
        # Full two-pass reorthogonalization handles nearly closed subspaces.
        for _ in 1:2
            h=Q[:,1:j]'*v;K[1:j,j]+=h;v-=Q[:,1:j]*h
        end
        next=norm(v);K[j+1,j]=next;used=j
        closed=next<=100eps(Float64)*max(scale,eps()) || j==n
        if !closed;Q[:,j+1]=v/next;end
        if j%8==0 || closed || j==limit
            F=svd(K[1:j+1,1:j]);rhs=zeros(j+1);rhs[1]=beta
            b=F.U'*rhs
            y=F.V*[s==0 ? 0.0 : -a/s for (s,a) in zip(F.S,b)]
            rel=norm(rhs+K[1:j+1,1:j]*y)/beta
            symmetry=norm(K[1:j,1:j]-K[1:j,1:j]')/max(norm(K[1:j,1:j]),eps())
            push!(rows,Dict("dimension"=>j,"unconstrained_relative_residual"=>rel,
                "projected_symmetry_error"=>symmetry,"next_basis_norm"=>next))
            symmetry<1e-7 || error("analytic Hessian lost projected symmetry")
            println("  Newton Krylov dimension=",j," linear residual=",rel);flush(stdout)
            if closed;reason="closed_subspace";break;end
            if rel<forcing;reason="inexact_Newton_forcing_met";break;end
        end
    end
    basis=Q[:,1:used];orth=norm(basis'*basis-I)
    orth<1e-9 || error("Newton Krylov orthogonality failed")
    H=K[1:used+1,1:used];rhs=zeros(used+1);rhs[1]=beta
    (;basis,H,rhs,svd=svd(H),report=Dict("dimension"=>used,"maxbasis"=>maxbasis,
        "forcing"=>forcing,"stop_reason"=>reason,"orthogonality_error"=>orth,"rows"=>rows))
end

function bk_step(b,radius)
    F=b.svd;scale=maximum(F.S);scale>0 || error("zero projected Hessian")
    d=F.S/scale;rhs=(F.U'*b.rhs)/scale
    coords(eta)=[(s==0 && eta==0) ? 0.0 : -s*a/(s*s+eta) for (s,a) in zip(d,rhs)]
    z=coords(0.0);eta=0.0
    if !all(isfinite,z) || norm(z)>radius
        high=1.0
        while norm(coords(high))>radius;high*=10;isfinite(high) || error("trust overflow");end
        while high>1e-280 && norm(coords(high/10))<radius;high/=10;end
        low=high/10
        for _ in 1:80
            middle=(low+high)/2
            if norm(coords(middle))>radius;low=middle;else;high=middle;end
        end
        eta=high;z=coords(eta)
    end
    y=F.V*z;delta=b.basis*y
    # The full-space residual norm equals the small Hessenberg residual norm.
    predicted_residual=norm(b.rhs+b.H*y)
    (;delta,eta,predicted_residual)
end

function bk_control()
    rng=MersenneTwister(2471);U=Matrix(qr(randn(rng,8,8)).Q)
    H=U*Diagonal([-5.0,-1.7,-0.4,-0.09,0.05,0.3,1.1,4.0])*U'
    g=randn(rng,8);b=bk_basis(x->H*x,g;maxbasis=8,forcing=0.0)
    comparisons=[]
    for radius in (0.2,2.0,100.0)
        got=bk_step(b,radius)
        # Independent dense KKT multiplier: (H'H+mu I)d=-H'g.
        normal=H'*H;rhs=-H'*g
        mu=0.0;reference=normal\rhs
        if norm(reference)>radius
            lo=0.0;hi=1.0
            while norm((normal+hi*I)\rhs)>radius;hi*=10;end
            for _ in 1:100
                mid=(lo+hi)/2
                if norm((normal+mid*I)\rhs)>radius;lo=mid;else;hi=mid;end
            end
            mu=hi;reference=(normal+mu*I)\rhs
        end
        step_error=norm(got.delta-reference)/max(norm(reference),eps())
        residual_error=abs(got.predicted_residual-norm(g+H*got.delta))/norm(g)
        max(step_error,residual_error)<1e-10 || error("indefinite Newton trust control failed")
        push!(comparisons,Dict("radius"=>radius,"step_relative_error"=>step_error,
            "residual_norm_error"=>residual_error))
    end
    Dict("passed"=>true,"indefinite_control"=>comparisons)
end
