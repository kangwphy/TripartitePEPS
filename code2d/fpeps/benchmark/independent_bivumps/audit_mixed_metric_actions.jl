# Isolated algebra/performance check on the actual chi=16 center pencils.
# No solver methods are replaced, and no boundary tensor is optimized.
include("mixed_center_response.jl")

function ma_metric(N)
    f=svd(N);s=f.S;U,V=f.U,f.V;a=sqrt.(s)
    ratio=minimum(s)/maximum(s)
    ratio>1e-12 || error("mixed metric lost rank; no rank cut: $ratio")
    WL=U*Diagonal(1 ./ a)*U';WR=V*Diagonal(1 ./ a)*V'
    K=[-1/(x*y*(x+y)*(x*x+y*y)) for x in a,y in a]
    function actions(dN,XL,XR)
        # E=U' dN V. In this basis the two Gram derivatives are
        # E Sigma + Sigma E' and E' Sigma + Sigma E, respectively.
        # Apply dW to the needed vectors without constructing dense dW.
        E=U'*dN*V
        left=K.*(E.*reshape(s,1,:)+reshape(s,:,1).*E')
        right=K.*(E'.*reshape(s,1,:)+reshape(s,:,1).*E)
        (U*(left*(U'*XL)),V*(right*(V'*XR)))
    end
    (;WL,WR,actions,ratio)
end

function ma_pencil(H,N,r,l)
    Hr,Nr=H*r,N*r;Hl,Nl=H'*l,N'*l
    ht,nt=dot(l,Hr),dot(l,Nr)
    min(abs(ht),abs(nt))>1e-12 || error("center contraction vanishes")
    hR,nR=Hr/ht,Nr/nt;hL,nL=Hl/conj(ht),Nl/conj(nt)
    metric=ma_metric(N)
    hs=(hR,hL,metric.WL*hR,metric.WR*hL)
    ns=(nR,nL,metric.WL*nR,metric.WR*nL)
    scales=[i<=2 ? sqrt((norm(hs[i])^2+norm(ns[i])^2)/2) : norm(hs[i]) for i in 1:4]
    minimum(scales)>1e-12 || error("center residual normalization vanishes")
    fields=[(hs[i]-ns[i])/scales[i] for i in 1:4]
    function derivative(dH,dN,dr,dl)
        dHr=dH*r+H*dr;dNr=dN*r+N*dr
        dHl=dH'*l+H'*dl;dNl=dN'*l+N'*dl
        dht=dot(dl,Hr)+dot(l,dHr);dnt=dot(dl,Nr)+dot(l,dNr)
        dhR=dHr/ht-hR*(dht/ht);dnR=dNr/nt-nR*(dnt/nt)
        dhL=dHl/conj(ht)-hL*conj(dht/ht)
        dnL=dNl/conj(nt)-nL*conj(dnt/nt)
        left,right=metric.actions(dN,hcat(hR,nR),hcat(hL,nL))
        dhs=(dhR,dhL,left[:,1]+metric.WL*dhR,right[:,1]+metric.WR*dhL)
        dns=(dnR,dnL,left[:,2]+metric.WL*dnR,right[:,2]+metric.WR*dnL)
        result=Vector{ComplexF64}[]
        for i in 1:4
            ds=i<=2 ? real(dot(hs[i],dhs[i])+dot(ns[i],dns[i]))/(2scales[i]) :
                real(dot(hs[i],dhs[i]))/scales[i]
            push!(result,(dhs[i]-dns[i])/scales[i]-fields[i]*(ds/scales[i]))
        end
        result
    end
    (;fields,derivative,metric,z=ht/nt)
end

function ma_time(f;repeats=3)
    f() # warm the exact closure before measuring
    samples=Float64[]
    for _ in 1:repeats
        start=time_ns();f();push!(samples,(time_ns()-start)/1e9)
    end
    samples
end

function ma_audit(source,out,controlpath)
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("preempt required")
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite")
    BLAS.set_num_threads(2)
    control=TOML.parsefile(controlpath)
    control["complete"] && control["passed"] && !control["perturbed"] || error("exact-input derivative audit required")
    hashes=copy(control["source_hashes"])
    for (p,h) in hashes
        bytes2hex(sha256(read(p)))==h || error("stale source: $p")
    end
    paths=joinpath.(source,["boundary_1.jls","boundary_3.jls","independent_pair.jls"])
    for p in paths
        get(hashes,p,"")==bytes2hex(sha256(read(p))) || error("control does not cover source: $p")
    end
    hashes[controlpath]=bytes2hex(sha256(read(controlpath)))
    hashes[@__FILE__]=bytes2hex(sha256(read(@__FILE__)))
    n=deserialize(paths[1]);sum(n.chi)==16 || error("this diagnostic targets chi=16")
    pair=deserialize(paths[3]);rows=[]
    report=Dict{String,Any}("complete"=>false,"passed"=>false,"chi"=>16,
        "job_id"=>ENV["SLURM_JOB_ID"],"partition"=>ENV["SLURM_JOB_PARTITION"],
        "source_hashes"=>hashes,"zero_optimization_steps"=>true,
        "diagnostic_only"=>true,"accepted_entropy"=>false,
        "scope"=>"center-pencil algebra and timing, not a new full solver control",
        "rows"=>rows)
    bv_write(out,report)
    try
        cache=mw_cache(pair.R.AL[1],pair.L.AL[1],pair.transfer;responses=false)
        rng=MersenneTwister(32416)
        for (name,P,r,l) in (("AC",cache.ac,cache.R.AC[1].data,cache.L.AC[1].data),
                             ("C",cache.c,cache.R.C[1].data,cache.L.C[1].data))
            H,N=P.H,P.N;slow=mw_pencil(H,N,r,l);fast=ma_pencil(H,N,r,l)
            field_error=norm(vcat(slow.fields...)-vcat(fast.fields...))/norm(vcat(slow.fields...))
            field_error<1e-8 || error("base field changed: $field_error")
            pencilrow=Dict{String,Any}("center"=>name,"matrix_size"=>size(N,1),
                "metric_singular_ratio"=>fast.metric.ratio,"field_relative_error"=>field_error,
                "directions"=>[])
            push!(rows,pencilrow)
            for direction in 1:4
                dH=randn(rng,ComplexF64,size(H));dH*=norm(H)/norm(dH)
                dN=randn(rng,ComplexF64,size(N));dN*=norm(N)/norm(dN)
                dr=randn(rng,ComplexF64,length(r));dr*=norm(r)/norm(dr)
                dl=randn(rng,ComplexF64,length(l));dl*=norm(l)/norm(dl)
                old=vcat(slow.derivative(dH,dN,dr,dl)...)
                new=vcat(fast.derivative(dH,dN,dr,dl)...)
                err=norm(new-old)/norm(old)
                err<1e-8 || error("derivative action changed: $err")
                stencils=[]
                for h in (1e-4,1e-5,1e-6)
                    plus=mw_pencil(H+h*dH,N+h*dN,r+h*dr,l+h*dl)
                    minus=mw_pencil(H-h*dH,N-h*dN,r-h*dr,l-h*dl)
                    fd=(vcat(plus.fields...)-vcat(minus.fields...))/(2h)
                    push!(stencils,Dict("h"=>h,"relative_error"=>norm(fd-new)/norm(new)))
                end
                minimum(x["relative_error"] for x in stencils)<1e-4 || error("independent pencil finite difference failed")
                row=Dict{String,Any}("direction"=>direction,"derivative_relative_error"=>err,"stencils"=>stencils)
                push!(pencilrow["directions"],row)
                if direction==1
                    row["original_seconds"]=ma_time(()->slow.derivative(dH,dN,dr,dl))
                    row["action_seconds"]=ma_time(()->fast.derivative(dH,dN,dr,dl))
                end
                bv_write(out,report)
            end
        end
        report["passed"]=true
    catch err
        report["error"]=sprint(showerror,err);rethrow()
    finally
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
        report["complete"]=true;bv_write(out,report)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    ma_audit(ARGS[1],ARGS[2],ARGS[3])
end
