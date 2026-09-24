# Keep both the Schmidt-coordinate step and the actual tensor step small.
# Post-scaling a Euclidean solution wastes directions when only a few weak
# Schmidt modes saturate the tensor bound. Solve in the combined ellipsoid.
function be_geometry(b,pushforward)
    Z=hcat((pushforward(v) for v in eachcol(b.basis))...)
    (;Z,gram=Symmetric(Z'*Z))
end

function be_step(b,geometry,radius;rawradius=0.5)
    radius>0 && rawradius>0 || error("invalid trust radius")
    k=size(b.basis,2)
    # ||y/radius||^2 + ||P Q y/rawradius||^2 <= 1.
    # y=radius U^-1 z makes this exactly ||z||<=1. No rank is cut.
    M=Symmetric(Matrix{Float64}(I,k,k)+(radius/rawradius)^2*geometry.gram)
    U=cholesky(M).U
    H=radius*(b.H/U);Q=radius*(b.basis/U)
    transformed=(;basis=Q,H,rhs=b.rhs,svd=svd(H))
    step=bk_step(transformed,1.0)
    y=b.basis'*step.delta
    metricnorm=sqrt(sum(abs2,y)/radius^2+sum(abs2,geometry.Z*y)/rawradius^2)
    metricnorm<=1+1e-7 || error("ellipsoid constraint failed: $metricnorm")
    predicted=norm(b.rhs+b.H*y)
    abs(predicted-step.predicted_residual)<1e-7*max(norm(b.rhs),eps()) ||
        error("ellipsoid coordinate residual mismatch")
    (;delta=step.delta,eta=step.eta,predicted_residual=predicted,metricnorm,
      raw_norm=norm(geometry.Z*y),rawradius)
end

function be_control()
    rng=MersenneTwister(2479);n=12
    U=Matrix(qr(randn(rng,n,n)).Q)
    H=U*Diagonal(range(-3,4;length=n))*U'
    g=randn(rng,n);b=bk_basis(x->H*x,g;maxbasis=n,forcing=0.0)
    V=Matrix(qr(randn(rng,n,n)).Q)
    P=V*Diagonal(10.0.^range(-1,2;length=n))*V'
    geometry=be_geometry(b,x->P*x);rows=[]
    for radius in (0.01,0.2,5.0)
        rawradius=0.5;got=be_step(b,geometry,radius;rawradius)
        # Independently solve the full-space KKT equations with a multiplier.
        G=I/radius^2+P'*P/rawradius^2
        normal=H'*H;rhs=-H'*g
        metric(v)=sqrt(real(dot(v,G*v)))
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
        max(err,kkt)<1e-8 || error("ellipsoid dense KKT comparison failed")
        # A uniformly rescaled old step is a feasible competitor.
        old=bk_step(b,radius).delta;old/=max(1,metric(old))
        bestres=norm(g+H*got.delta);oldres=norm(g+H*old)
        bestres<=oldres+1e-10*norm(g) || error("ellipsoid step worse than feasible rescaling")
        push!(rows,Dict("radius"=>radius,"rawradius"=>rawradius,
            "step_relative_error"=>err,"KKT_relative_error"=>kkt,
            "metric_norm"=>metric(got.delta),"tensor_step_norm"=>norm(P*got.delta),
            "optimized_residual"=>bestres,"rescaled_residual"=>oldres))
    end
    Dict("passed"=>true,"ellipsoid_control"=>rows)
end
