# Correlation propagation channel WITH the PEPS column; separate from edge-pair xi.
include("mortier_ctm_correlations.jl")
using KrylovKit
function channel_xi(peps,env)
    O=MPSKit.FiniteMPO(signed_operators(space(peps[1],1))["normal"])
    vn,vo=PEPSKit.start_correlator_left(CartesianIndex(1,1),peps,O[1],peps,env)
    T=PEPSKit.edge_transfermatrix(1,1,peps,peps,env)
    result=Dict{String,Any}();spectra=Dict{String,Any}()
    for (name,initial) in (("even",vn),("odd",vo))
        initial=randn(MersenneTwister(991),ComplexF64,space(initial))
        initial=initial/norm(initial)
        vals,vecs,info=KrylovKit.eigsolve(v->v*T,initial,4,:LM;tol=1e-11,krylovdim=64,maxiter=1000)
        residuals=[norm(v*T-val*v)/max(norm(v*T),eps()) for (v,val) in zip(vecs,vals)]
        result[name*"_lambda_real"]=real.(vals);result[name*"_lambda_imag"]=imag.(vals)
        result[name*"_relative_residuals"]=residuals;result[name*"_eigenpairs_converged"]=info.converged
        spectra[name]=vals
    end
    normvalue=first(spectra["even"])
    for name in ("even","odd")
        ratios=spectra[name]./normvalue;i=name=="even" ? 2 : 1
        if length(ratios)<i
            result[name*"_xi_x_raw"]=NaN;result[name*"_decaying_resolved"]=false
            continue
        end
        r=abs(ratios[i])
        result[name*"_xi_x_raw"]=-1/log(r)
        result[name*"_decaying_resolved"]=0<r<1-1e-9
        result[name*"_ratio_abs"]=r
    end
    result
end
function caps_for(path)
    n=deserialize(joinpath(path,"boundary_1.jls"));s=deserialize(joinpath(path,"boundary_3.jls"))
    p=paired_boundary_observable(n,s,id(ComplexF64,space(n.peps[1],1)));L,R=p.left,p.right
    cs=reshape([id(ComplexF64,space(L,4)'),id(ComplexF64,space(R,1)),id(ComplexF64,space(R,4)'),id(ComplexF64,space(L,1))],4,1,1)
    es=reshape([n.state.AL[1],R,s.state.AL[1],L],4,1,1)
    n.peps,PEPSKit.CTMRGEnv(cs,es)
end
function main_xi()
    rows=Dict{String,Any}[];root="data/mortier_ctm_scan"
    paths=filter(p->isfile(joinpath(root,p,"environment.jls")),readdir(root))
    for path in paths
        metadata=TOML.parsefile(joinpath(root,path,"result.toml"))
        if get(metadata,"postcheck_residual",metadata["residual"])>1e-6
            push!(rows,Dict("method"=>"CTMRG","path"=>joinpath(root,path,"result.toml"),
                "xi"=>Dict{String,Any}(),"skipped"=>"CTMRG residual exceeds 1e-6; edge-pair raw diagnostic retained separately"))
            continue
        end
        c=deserialize(joinpath(root,path,"environment.jls"))
        values=channel_xi(c.peps,c.environment)
        row=Dict("method"=>"CTMRG","path"=>joinpath(root,path,"result.toml"),"xi"=>values)
        push!(rows,row);println(path," physical xi even=",values["even_xi_x_raw"]," odd=",values["odd_xi_x_raw"]);flush(stdout)
        open(joinpath(root,"physical_xi.toml"),"w") do io;TOML.print(io,Dict("cases"=>rows));end
    end
    for row in TOML.parsefile(joinpath(root,"vumps.toml"))["cases"]
        peps,caps=caps_for(row["source"]);values=channel_xi(peps,caps)
        record=Dict("method"=>"VUMPS","path"=>row["source"],"xi"=>values)
        push!(rows,record);println(row["source"]," physical xi even=",values["even_xi_x_raw"]," odd=",values["odd_xi_x_raw"]);flush(stdout)
        open(joinpath(root,"physical_xi.toml"),"w") do io;TOML.print(io,Dict("cases"=>rows));end
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    main_xi()
end
