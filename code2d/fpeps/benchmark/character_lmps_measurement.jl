# Exact even ket/bra-support character, not numerical parity-term truncation.
# The old signed expansion remains the default production measurement path.
include("even_replica_support.jl")
using Serialization, SHA
import TensorKit

function character_geometry_fingerprint(seams,caps;centers=nothing)
    buffer=IOBuffer()
    serialize(buffer,"exact-even-character-v1")
    for name in keys(seams)
        serialize(buffer,name)
        for tensor in (getproperty(seams,name)...,getproperty(caps,name))
            # Canonical numerical content, not object identities or internal
            # TensorKit dictionaries, so restart in another process is safe.
            data=sort(collect(TensorKit.blocks(tensor));by=x->string(first(x)))
            serialize(buffer,(string(TensorKit.space(tensor)),
                [(string(q),Array(block)) for (q,block) in data]))
        end
    end
    if centers!==nothing
        serialize(buffer,"explicit-even-centers-v1")
        for (name,tensor) in pairs(centers)
            data=sort(collect(TensorKit.blocks(tensor));by=x->string(first(x)))
            serialize(buffer,(name,string(TensorKit.space(tensor)),
                [(string(q),Array(block)) for (q,block) in data]))
        end
    end
    bytes2hex(SHA.sha256(take!(buffer)))
end
function character_lmps_sectors(seams,caps;depth,factorized=true,propagation_cache=nothing,centers=nothing)
    definitions=(Z1=((1,),(1,),(1,)),Z2A=((2,1),(1,2),(1,2)),
        Z2B=((1,2),(2,1),(1,2)),Z2C=((1,2),(1,2),(2,1)),
        Z4=((1,2,3,4),(2,1,4,3),(3,4,1,2)))
    results=map(values(definitions)) do permutations
        character=even_replica_character(permutations)
        result=finite_graded_sixr(seams,permutations,caps;depth,factorized,
            insertions=character.insertions,propagation_cache,centers)
        terms=[(;weight=1.0,insertions=character.insertions,result)]
        total=FermionicPEPS._scaled_complex_sum([(;value=result.value,logscale=result.logscale)])
        (;total...,sectors=terms,depth,factorized,character_mask=character.mask)
    end
    NamedTuple{keys(definitions)}(results)
end

"Extend seam depth until the original three-point length/phase gates pass."
function measure_character_lmps_adaptive(seams,caps;initial_depths=(8,16,32,64),
        max_depth=1024,on_sample=(_->nothing),checkpoint_dir=nothing,centers=nothing,kwargs...)
    ds=collect(initial_depths)
    max_depth>=last(ds) || throw(ArgumentError("max_depth is below initial depths"))
    shared_cache=Dict{Any,Any}()
    sector_cache=Dict{Int,Any}()
    samples=Dict{Int,Any}()
    fingerprint=if checkpoint_dir===nothing
        nothing
    else
        mkpath(checkpoint_dir)
        character_geometry_fingerprint(seams,caps;centers)
    end
    builder=function (s,c;depth,propagation_cache=nothing)
        get!(sector_cache,depth) do
            path=checkpoint_dir===nothing ? nothing : joinpath(checkpoint_dir,"depth_$depth.jls")
            if path!==nothing && isfile(path)
                saved=deserialize(path)
                saved.fingerprint==fingerprint && saved.depth==depth ||
                    error("incompatible saved replica sectors: $path")
                # The current phase and length gates are reapplied below.
                saved.sectors
            else
                sectors=character_lmps_sectors(s,c;depth,propagation_cache=shared_cache,centers)
                if path!==nothing
                    temporary=path*".tmp_$(getpid())"
                    serialize(temporary,(;fingerprint,depth,sectors))
                    mv(temporary,path)
                end
                sectors
            end
        end
    end
    record=function (sample)
        if !haskey(samples,sample.depth)
            samples[sample.depth]=sample
            on_sample(sample)
        end
    end
    while true
        result=measure_lmps_stilde(seams,caps;depths=Tuple(ds),
            accept_unconverged=true,sector_builder=builder,on_sample=record,kwargs...)
        if result.converged
            return merge(result,(;all_samples=[samples[d] for d in sort(collect(keys(samples)))],
                                 adaptive_depth=true))
        end
        last(ds)<max_depth || error("character LMPS length gate failed at max_depth=$max_depth: $(result.diagnostics)")
        ds=[ds[end-1],ds[end],min(2last(ds),max_depth)]
    end
end
