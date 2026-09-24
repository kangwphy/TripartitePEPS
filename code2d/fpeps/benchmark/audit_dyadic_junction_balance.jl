# Frozen endpoint diagnostic only. Reciprocal powers of two on each edge
# preserve every binary input entry exactly (checked by immediate inversion).
# All sectors/indices remain; no production contraction or entropy is changed.
include("../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, LinearAlgebra, Serialization, TOML, SHA

function port_weights(E,leg)
    weights=Dict(q=>zeros(Float64,dim(space(E,leg),q)) for q in sectors(space(E,leg)))
    for (fo,fi) in fusiontrees(E)
        A=E[fo,fi];q=fo.uncoupled[leg]
        for index in eachindex(weights[q])
            weights[q][index]+=sum(abs2,selectdim(A,leg,index))
        end
    end
    weights
end

function exact_port_scale!(E,leg,exponents)
    for (fo,fi) in fusiontrees(E)
        A=E[fo,fi];exps=exponents[fo.uncoupled[leg]]
        for index in eachindex(exps)
            exponent=exps[index];exponent==0 && continue
            part=selectdim(A,leg,index)
            for i in eachindex(part)
                old=part[i]
                new=complex(ldexp(real(old),exponent),ldexp(imag(old),exponent))
                isfinite(new) || error("dyadic scaling overflow")
                restored=complex(ldexp(real(new),-exponent),ldexp(imag(new),-exponent))
                restored==old || error("dyadic scaling lost an input bit")
                part[i]=new
            end
        end
    end
end

function second_plan(network,firstplan)
    adjacent(i,j)=length(intersect(network[i],network[j]))==1
    for j in 2:5,k in j+1:6
        left=[1,j,k];right=setdiff(1:6,left)
        all(adjacent(a,b) for a in left for b in left if a<b) || continue
        all(adjacent(a,b) for a in right for b in right if a<b) || continue
        for a in left,b in right,label in intersect(network[a],network[b])
            label==firstplan.label && continue
            lr=setdiff(left,[a]);rr=setdiff(right,[b])
            tree=[[[a,lr[1]],lr[2]],[[b,rr[1]],rr[2]]]
            order=first(TensorKit.TensorOperations.tree2indexorder(tree,network))
            return (;label,a,b,order)
        end
    end
    error("no alternative edge")
end

function balance_frozen_endpoints(endpoints,network;sweeps=40)
    tensors=deepcopy(endpoints)
    occurrences=Dict(label=>[(j,i) for (j,ls) in enumerate(network)
                            for (i,l) in enumerate(ls) if l==label]
                     for label in sort(unique(vcat(network...))))
    history=Dict{String,Any}[];updates=0
    initial_log_envelope=sum(log(norm(E)) for E in tensors)
    for sweep in 1:sweeps
        changed=0
        for label in sort(collect(keys(occurrences)))
            ports=occurrences[label];length(ports)==2 || error("edge multiplicity")
            (a,i),(b,j)=ports
            space(tensors[a],i)==dual(space(tensors[b],j)) || error("edge arrows")
            wa=port_weights(tensors[a],i);wb=port_weights(tensors[b],j)
            na=sum(sum(v) for v in values(wa));nb=sum(sum(v) for v in values(wb))
            exponents=Dict(q=>[if x>0 && y>0
                    clamp(round(Int,(log2(y/nb)-log2(x/na))/4),-16,16)
                else
                    0
                end for (x,y) in zip(wa[q],wb[q])] for q in keys(wa))
            nchanged=sum(count(x->!iszero(x),v) for v in values(exponents))
            nchanged==0 && continue
            exact_port_scale!(tensors[a],i,exponents)
            exact_port_scale!(tensors[b],j,Dict(q=>-v for (q,v) in exponents))
            changed+=nchanged
        end
        updates+=changed
        push!(history,Dict("sweep"=>sweep,"changed_index_scales"=>changed,
            "log_frobenius_envelope_ratio"=>sum(log(norm(E)) for E in tensors)-initial_log_envelope))
        changed==0 && break
    end
    (;tensors,history,updates)
end

function audit_dyadic_junction(source,out,indices)
    ispath(out) && error("refusing overwrite");mkpath(out)
    BLAS.set_num_threads(parse(Int,get(ENV,"SLURM_CPUS_PER_TASK","2")))
    sourcehash=bytes2hex(sha256(read(source)));saved=deserialize(source)
    sectors=hasproperty(saved,:samples) ? last(saved.samples).sectors : saved
    rows=Dict{String,Any}[]
    report=Dict("complete"=>false,"source"=>source,"source_sha256"=>sourcehash,
        "depth"=>sectors.Z4.depth,"rows"=>rows,"accepted_entropy"=>false,
        "zero_optimization_steps"=>true,"all_input_bits_preserved_by_each_gauge"=>true,
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "sixr_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../src/sixr.jl")))),
        "interpretation"=>"Reciprocal parity-preserving dyadic edge gauges only, with exact binary roundtrips for every changed entry. Frozen Float64 endpoints and all their indices remain. Reclosure agreement diagnoses contraction arithmetic only, not propagation, environment accuracy, or entropy.")
    function checkpoint()
        open(joinpath(out,"report.toml.tmp"),"w") do io;TOML.print(io,report);end
        mv(joinpath(out,"report.toml.tmp"),joinpath(out,"report.toml");force=true)
    end
    for index in indices
        r=sectors.Z4.sectors[index].result
        owner=Dict((X,c)=>3(c-1)+x for (x,X) in enumerate((:A,:B,:C)) for c in 1:4)
        network=[[owner[(X,c)] for c in spec.copies for X in (spec.X,spec.Y)] for spec in r.specs]
        length(network)==6 && all(n->length(n)==4,network) || error("six-endpoint Z4 required")
        dimensions=Dict(label=>Float64(dim(space(E,i))) for (E,ls) in zip(r.endpoints,network) for (i,label) in enumerate(ls))
        firstplan=FermionicPEPS._junction_slice_plan(network,dimensions)
        plans=(firstplan,second_plan(network,firstplan))
        balanced=balance_frozen_endpoints(r.endpoints,network)
        checks=[];closed=ComplexF64[]
        for plan in plans
            result=FermionicPEPS._sliced_junction_close(balanced.tensors,network,plan)
            push!(closed,result.value)
            push!(checks,Dict("sliced_edge"=>plan.label,"slice_sum_cancellation"=>result.cancellation,
                "value_real"=>real(result.value),"value_imag"=>imag(result.value),
                "relative_difference_from_original"=>abs(result.value-r.value)/max(abs(r.value),floatmin(Float64))))
        end
        disagreement=abs(closed[1]-closed[2])/max(abs(closed[1]),abs(closed[2]),floatmin(Float64))
        row=Dict("term"=>index,"gauge_history"=>balanced.history,"scale_updates"=>balanced.updates,
            "checks"=>checks,"balanced_reclosure_relative_disagreement"=>disagreement,
            "original_value_real"=>real(r.value),"original_value_imag"=>imag(r.value))
        push!(rows,row);checkpoint()
        println("term=",index," envelope ratio=",exp(last(balanced.history)["log_frobenius_envelope_ratio"]),
            " balanced closures disagreement=",disagreement);flush(stdout)
    end
    bytes2hex(sha256(read(source)))==sourcehash || error("source changed")
    report["complete"]=true
    report["balanced_reclosure_gate_passed"]=all(r["balanced_reclosure_relative_disagreement"]<1e-10 for r in rows)
    checkpoint()
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    audit_dyadic_junction(ARGS[1],ARGS[2],parse.(Int,split(ARGS[3],',')))
end
