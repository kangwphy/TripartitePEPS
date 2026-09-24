# Convex min-max of block least squares on a Euclidean trust ball.
# The simplex dual gives an explicit primal/dual certificate. Pairwise dual
# steps use curvature only as a proposal; actual dual improvement is checked.
# Requires LinearAlgebra and the original bk_step, supplied by the caller.

function mm_point(blocks,rhs,w,radius)
    minimum(w)>=0 && abs(sum(w)-1)<1e-12 || error("invalid dual weights")
    A=reduce(vcat,[sqrt(w[i])*blocks[i] for i in eachindex(w)])
    f=reduce(vcat,[sqrt(w[i])*rhs[i] for i in eachindex(w)])
    k=size(A,2);S=svd(A)
    maximum(S.S)>0 || error("zero weighted operator")
    model=(;basis=Matrix{Float64}(I,k,k),H=A,rhs=f,svd=S)
    step=bk_step(model,radius);x=step.delta;lambda=step.eta*maximum(S.S)^2
    residuals=[rhs[i]+blocks[i]*x for i in eachindex(w)]
    values=[dot(r,r)/2 for r in residuals]
    gradients=hcat([blocks[i]'*residuals[i] for i in eachindex(w)]...)
    stationarity=A'*(f+A*x)+lambda*x
    norm(x)<=radius*(1+1e-12) || error("infeasible inner trust point")
    # This lower bound remains conservative with an inexact inner KKT solve:
    # convexity gives q(y)>=q(x)+(-lambda*x+r)'(y-x) for every feasible y.
    dual=dot(w,values)
    correction=(radius+norm(x))*norm(stationarity)+lambda*max(0.0,radius*norm(x)-dot(x,x))
    lower=dual-correction;primal=maximum(values)
    gap=max(0.0,primal-lower)/max(primal,eps())
    (;x,S,lambda,values,gradients,dual,lower,primal,gap,correction,
        stationarity_norm=norm(stationarity))
end

function mm_solve(blocks,rhs,radius;tol=1e-6,maxiter=80,callback=nothing)
    n=length(blocks);n==length(rhs) && n>0 || error("invalid block problem")
    scale=maximum(norm,rhs);scale>0 || error("already zero residual")
    A=[a/scale for a in blocks];f=[b/scale for b in rhs]
    w=fill(1.0/n,n);point=mm_point(A,f,w,radius);rows=[];reason="maxiter"
    for iteration in 0:maxiter
        row=Dict{String,Any}("iteration"=>iteration,"weights"=>copy(w),
            "primal"=>point.primal,"dual_lower_bound"=>point.lower,
            "relative_duality_gap"=>point.gap,"inner_KKT_bound_correction"=>point.correction,
            "inner_stationarity_norm"=>point.stationarity_norm,"step_norm"=>norm(point.x))
        push!(rows,row);callback===nothing || callback(rows)
        if point.gap<tol;reason="duality_gap_passed";break;end
        iteration==maxiter && break
        j=argmax(point.values);active=findall(>(0.0),w)
        donor=active[argmin(point.values[active])]
        j!=donor || (reason="inner_bound_limits_certificate";break)
        slope=point.values[j]-point.values[donor]
        slope>0 || (reason="no_dual_ascent";break)
        g=point.gradients[:,j]-point.gradients[:,donor]
        den=point.S.S.^2 .+ point.lambda
        inverse(v)=point.S.V*[d==0 ? 0.0 : t/d for (t,d) in zip(point.S.V'*v,den)]
        p=inverse(g)
        if point.lambda>0
            u=inverse(point.x);d=dot(point.x,u)
            d>0 || error("invalid boundary sensitivity")
            p-=u*(dot(point.x,p)/d)
        end
        curvature=max(0.0,dot(g,p))
        alpha=curvature>0 ? min(w[donor],slope/curvature) : w[donor]
        accepted=nothing;lines=[];row["line_search"]=lines
        for attempt in 1:24
            trial=copy(w);trial[j]+=alpha;trial[donor]=max(0.0,trial[donor]-alpha)
            trial/=sum(trial)
            fresh=mm_point(A,f,trial,radius)
            gain=fresh.dual-point.dual
            push!(lines,Dict("alpha"=>alpha,"dual_gain"=>gain,"slope"=>slope))
            if gain>=1e-4alpha*slope
                accepted=(trial,fresh);break
            end
            alpha/=2
        end
        accepted===nothing && (reason="dual_line_search_failed";break)
        w,point=accepted
    end
    report=Dict("passed"=>point.gap<tol,"stop_reason"=>reason,"tolerance"=>tol,
        "relative_duality_gap"=>point.gap,"rows"=>rows,"weights"=>w,
        "predicted_block_maximum"=>scale*sqrt(2point.primal),
        "primal"=>point.primal*scale^2,"dual_lower_bound"=>point.lower*scale^2)
    (;delta=point.x,report)
end

function mm_control()
    rows=[]
    # Unequal slopes require nonuniform optimal dual weights (2/3,1/3).
    A=[reshape([1.0],1,1),reshape([2.0],1,1)];f=[[-2.0],[1.0]]
    for (radius,expected,maximum_residual) in ((1.0,1/3,5/3),(0.1,0.1,1.9))
        result=mm_solve(A,f,radius;tol=1e-9,maxiter=120)
        result.report["passed"] || error("analytic minimax control not certified")
        err=abs(only(result.delta)-expected)
        maxerr=abs(result.report["predicted_block_maximum"]-maximum_residual)
        max(err,maxerr)<1e-7 || error("analytic minimax control failed")
        push!(rows,Dict("radius"=>radius,"coordinate_error"=>err,"maximum_error"=>maxerr,
            "certificate"=>result.report))
    end
    # Reordering blocks must change weights, not the physical step.
    a=mm_solve(A,f,1.0;tol=1e-9,maxiter=120)
    b=mm_solve(reverse(A),reverse(f),1.0;tol=1e-9,maxiter=120)
    b.report["passed"] && norm(a.delta-b.delta)<1e-10 || error("minimax block ordering failed")
    Dict("passed"=>true,"checks"=>rows,"block_permutation_error"=>norm(a.delta-b.delta))
end
