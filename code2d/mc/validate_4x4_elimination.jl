# validate_4x4_elimination.jl -- exact six-layer contraction for the 4x4 test.
#
# Every replicated Ising spin is a binary variable.  We build the pair factors
# of K_lambda, choose a min-fill elimination order, and sum variables exactly.
# The largest intermediate factor for this geometry has only 13 binary indices;
# there is no SVD, bond-dimension cutoff, or Monte Carlo sampling here.

include(joinpath(@__DIR__, "validate_rect_exact.jl"))

struct BinaryFactor
    vars::Vector{Int}
    logv::Vector{Float64}
    dlogv::Vector{Float64}
end

"Log-sum-exp for two scalar contributions."
logadd2(a,b) = begin
    m=max(a,b)
    isfinite(m) ? m+log(exp(a-m)+exp(b-m)) : m
end

"Return merged pairs (J_D, Delta J), where J_lambda=J_D+lambda*Delta J."
function edge_couplings_4x4()
    Lx=4; Ly=4
    base=Dict{Tuple{Int,Int},Float64}()
    delta=Dict{Tuple{Int,Int},Float64}()
    vid(x,y,l) = NL_RECT*(y*Lx+x)+(l-1)
    for y in 0:Ly-1, x in 0:Lx-1
        for (xx,yy) in ((x+1,y),(x,y+1))
            (xx>=Lx || yy>=Ly) && continue
            tN,tD=endpoint_couplings(region_rect(x,y,Lx,Ly),
                                      region_rect(xx,yy,Lx,Ly),BETA_C_RECT)
            for (a,b,J) in tD
                u,v=vid(x,y,a),vid(xx,yy,b); u>v && ((u,v)=(v,u))
                base[(u,v)]=get(base,(u,v),0.0)+J
                delta[(u,v)]=get(delta,(u,v),0.0)-J
            end
            for (a,b,J) in tN
                u,v=vid(x,y,a),vid(xx,yy,b); u>v && ((u,v)=(v,u))
                delta[(u,v)]=get(delta,(u,v),0.0)+J
            end
        end
    end
    keys_all=union(keys(base),keys(delta))
    Dict(edge=>(get(base,edge,0.0),get(delta,edge,0.0)) for edge in keys_all)
end

"Greedy min-fill order and its induced width for the coupling graph."
function minfill_order(nvar,edges)
    adj=[Set{Int}() for _ in 1:nvar]
    for (u0,v0) in keys(edges); u=u0+1;v=v0+1; push!(adj[u],v);push!(adj[v],u); end
    alive=trues(nvar); order=Int[]; width=0
    while length(order)<nvar
        best=0; bestscore=(typemax(Int),typemax(Int))
        for v in 1:nvar
            alive[v] || continue
            ns=[u for u in adj[v] if alive[u]]
            missing=0
            for i in 1:length(ns)-1, j in i+1:length(ns); missing += !(ns[j] in adj[ns[i]]); end
            score=(missing,length(ns))
            if score<bestscore; best=v;bestscore=score; end
        end
        ns=[u for u in adj[best] if alive[u]]; width=max(width,length(ns))
        for i in 1:length(ns)-1, j in i+1:length(ns); push!(adj[ns[i]],ns[j]);push!(adj[ns[j]],ns[i]);end
        alive[best]=false; push!(order,best)
    end
    order,width
end

"Multiply all incident log factors and exactly sum the selected binary variable."
function eliminate_one(fs,var)
    hit=findall(f->var in f.vars,fs)
    unionvars=sort!(unique!(vcat((fs[i].vars for i in hit)...)))
    outvars=filter(!=(var),unionvars)
    out=zeros(1<<length(outvars)); dout=zeros(length(out))
    pos=Dict(v=>i-1 for (i,v) in enumerate(unionvars))
    outpos=Dict(v=>i-1 for (i,v) in enumerate(outvars))
    total=zeros(1<<length(unionvars)); dtotal=zeros(length(total))
    for i in hit
        f=fs[i]
        for state in eachindex(total)
            idx=0
            for (k,v) in enumerate(f.vars)
                idx |= (((state-1)>>pos[v])&1)<<(k-1)
            end
            total[state]+=f.logv[idx+1]
            dtotal[state]+=f.dlogv[idx+1]
        end
    end
    for outstate in 0:length(out)-1
        vals=Float64[]; ders=Float64[]
        for bit in 0:1
            full=0
            for v in unionvars
                b=v==var ? bit : ((outstate>>outpos[v])&1)
                full |= b<<pos[v]
            end
            push!(vals,total[full+1])
            push!(ders,dtotal[full+1])
        end
        out[outstate+1]=logadd2(vals[1],vals[2])
        m=max(vals[1],vals[2]); w1=exp(vals[1]-m); w2=exp(vals[2]-m)
        dout[outstate+1]=(w1*ders[1]+w2*ders[2])/(w1+w2)
    end
    keep=[f for (i,f) in enumerate(fs) if !(i in hit)]
    push!(keep,BinaryFactor(outvars,out,dout)); keep
end

"Compute log Z_lambda exactly for the 4x4 six-layer model."
function exact_logz_elimination_4x4(lam; verbose=false)
    edges=edge_couplings_4x4(); order,width=minfill_order(96,edges)
    fs=BinaryFactor[]
    for ((u,v),(JD,DK)) in edges
        # little-endian assignments: 00, 10, 01, 11 for sorted variables
        J=JD+lam*DK
        push!(fs,BinaryFactor([u+1,v+1],[J,-J,-J,J],[DK,-DK,-DK,DK]))
    end
    for v in order; fs=eliminate_one(fs,v); end
    @assert all(isempty(f.vars) for f in fs)
    verbose && println("min-fill induced width=$width")
    sum(f.logv[1] for f in fs),sum(f.dlogv[1] for f in fs)
end

function main_4x4_elimination()
    lD,_=exact_logz_elimination_4x4(0.0;verbose=true)
    lN,_=exact_logz_elimination_4x4(1.0)
    @printf("4x4 endpoint=%.15f logZD=%.15f logZN=%.15f\n",lD-lN,lD,lN)
    x,w=gl01_rect(12)
    ti=-sum(w[q]*exact_logz_elimination_4x4(x[q])[2] for q in eachindex(x))
    @printf("4x4 exact_GL12=%.15f endpoint_minus_GL12=%.2e\n",ti,(lD-lN)-ti)
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && main_4x4_elimination()
