# Convex reduced Newton least squares with a trust ball and two half-spaces.
# Enumerate active sets; each affine slice reduces to a smaller trust ball.

function bd_ball(M,rhs,radius)
    n=size(M,2)
    n==0 && return (;z=Float64[],mu=0.0)
    F=svd(M);s=F.S;a=F.U'*rhs
    coords(mu)=[v==0 && mu==0 ? 0.0 : -v*w/(v*v+mu) for (v,w) in zip(s,a)]
    mu=0.0;t=coords(mu)
    if norm(t)>radius
        lo=0.0;hi=max(maximum(s)^2,eps())
        while norm(coords(hi))>radius;hi*=10;end
        for _ in 1:100
            mid=(lo+hi)/2
            if norm(coords(mid))>radius;lo=mid;else;hi=mid;end
        end
        mu=hi;t=coords(mu)
    end
    (;z=F.V*t,mu)
end

function bd_constrained_ball(M,rhs,A,d)
    n=size(M,2);m=length(d);size(A)==(m,n) || error("constraint shape mismatch")
    m<=4 || error("active-set enumeration bounded to four constraints")
    minimum(d)>=0 || error("zero step must be feasible")
    best=nothing
    for mask in 0:(2^m-1)
        active=[j for j in 1:m if ((mask>>(j-1))&1)==1]
        if isempty(active)
            z0=zeros(n);N=Matrix{Float64}(I,n,n)
        else
            F=svd(A[active,:];full=true)
            rank=count(s->s>1e-12*maximum(F.S),F.S)
            rank==length(active) || continue
            z0=F.V[:,1:rank]*((F.U'*(-d[active]))./F.S)
            norm(z0)<=1+1e-12 || continue
            N=F.V[:,rank+1:end]
        end
        radius=sqrt(max(0.0,1-sum(abs2,z0)))
        radius>1e-12 || continue
        sol=bd_ball(M*N,rhs+M*z0,radius);z=z0+N*sol.z
        margins=A*z+d
        minimum(margins)>=-1e-10*max(1.0,norm(d),norm(A)) || continue
        residual=norm(rhs+M*z)
        if best===nothing || residual<best.residual
            best=(;z,mu=sol.mu,active,residual,margins)
        end
    end
    best===nothing && error("no feasible dominance trust step")
    best
end

function bd_step(b,geometry,radius,constraints;rawradius=0.5)
    k=size(b.basis,2)
    U=cholesky(Symmetric(Matrix{Float64}(I,k,k)+(radius/rawradius)^2*geometry.gram)).U
    H=radius*(b.H/U);Q=radius*(b.basis/U)
    A=reduce(vcat,[permutedims(Q'*v) for v in constraints.gradients])
    d=constraints.gaps-constraints.floors
    scales=max.(sqrt.(sum(abs2,A;dims=2))[:,1],abs.(d),eps())
    scaledA=A./scales;scaledd=d./scales
    step=bd_constrained_ball(H,b.rhs,scaledA,scaledd)
    delta=Q*step.z;y=b.basis'*delta
    metricnorm=sqrt(sum(abs2,y)/radius^2+sum(abs2,geometry.Z*y)/rawradius^2)
    metricnorm<=1+1e-7 || error("dominance trust ellipsoid failed")
    predicted=norm(b.rhs+b.H*y)
    abs(predicted-step.residual)<1e-7*max(norm(b.rhs),eps()) || error("dominance coordinate mismatch")
    (;delta,eta=step.mu,predicted_residual=predicted,metricnorm,
      active=step.active,linearized_gaps=A*step.z+constraints.gaps)
end

function bd_control()
    rng=MersenneTwister(2513);rows=[]
    for trial in 1:12
        n=8;Q=Matrix(qr(randn(rng,n,n)).Q)
        H=Q*Diagonal(range(-2,3;length=n))*Q';g=randn(rng,n)
        A=randn(rng,2,n);d=trial%3==0 ? [1e-6,1e-5] : [0.1,0.2]
        sol=bd_constrained_ball(H,g,A,d)
        gradient=H'*(g+H*sol.z)+sol.mu*sol.z
        eta=isempty(sol.active) ? Float64[] : A[sol.active,:]'\gradient
        stationarity=gradient-(isempty(sol.active) ? zeros(n) : A[sol.active,:]'*eta)
        err=norm(stationarity)/max(norm(H'*g),eps())
        slackerr=abs(sol.mu*(sum(abs2,sol.z)-1))/max(norm(H'*g),eps())
        max(err,slackerr)<1e-8 || error("dominance KKT stationarity failed")
        isempty(eta) || minimum(eta)>-1e-9 || error("negative inequality multiplier")
        minimum(sol.margins)>-1e-9 && norm(sol.z)<=1+1e-9 || error("dominance feasibility failed")
        push!(rows,Dict("trial"=>trial,"active_constraints"=>sol.active,
            "KKT_stationarity_error"=>err,"ball_complementarity_error"=>slackerr,
            "inequality_multipliers"=>eta,"constraint_margins"=>sol.margins,
            "ball_multiplier"=>sol.mu,"step_norm"=>norm(sol.z)))
    end
    any(length(x["active_constraints"])==2 for x in rows) || error("missing two-active-constraint control")
    Dict("passed"=>true,"checks"=>rows)
end
