const _REGIONAL_SEAMS=((:AB,:A,:B),(:AC,:A,:C),(:BC,:B,:C))

function _relative_cycles(pX,pY)
    q=invperm(collect(pX))[collect(pY)]
    visited=falses(length(q)); cycles=Vector{Int}[]
    for start in eachindex(q)
        visited[start] && continue
        cycle=Int[]; i=start
        while !visited[i]
            push!(cycle,i); visited[i]=true; i=q[i]
        end
        i==start || error("invalid relative permutation")
        push!(cycles,cycle)
    end
    cycles
end

"Finite propagation with positive norm scaling; complex phases are never removed."
function _propagate_seam(graph,cap,depth)
    depth>=0 || throw(ArgumentError("nonnegative seam depth required"))
    scale=norm(cap)
    scale>0 || throw(ArgumentError("far cap must be nonzero"))
    endpoint=cap/scale; logscale=log(scale)
    for _ in 1:depth
        endpoint=_apply_seam_graph(graph,endpoint)
        scale=norm(endpoint)
        isfinite(scale) || error("seam propagation overflowed")
        iszero(scale) && return (;endpoint,logscale=0.0,zero=true,lambda=0.0im,residual=0.0)
        endpoint/=scale
        logscale+=log(scale)
    end
    image=_apply_seam_graph(graph,endpoint)
    lambda=dot(endpoint,image)
    residual=norm(image-lambda*endpoint)/max(norm(image),eps())
    (;endpoint,logscale,zero=false,lambda,residual)
end

"Restrict one output port to one degeneracy index, retaining its parity wire."
function _junction_port_slice(E,leg,q,index)
    numin(E)==0 || throw(ArgumentError("junction slice requires all-output endpoints"))
    V=space(E,leg)
    W=Vect[FermionParity](q=>1)
    isdual(V) && (W=dual(W))
    ports=ntuple(i->i==leg ? W : space(E,i),numout(E))
    sliced=zeros(ComplexF64,foldl(⊗,ports)←one(W))
    for (fo,fi) in fusiontrees(sliced)
        source=E[fo,fi]
        selection=ntuple(i->i==leg ? (index:index) : Colon(),ndims(source))
        sliced[fo,fi] .= view(source,selection...)
    end
    sliced
end

"Two disjoint triangles of an octahedron, with a sliced edge between them."
function _junction_slice_plan(network,dimensions)
    length(network)==6 && all(ls->length(ls)==4,network) || return nothing
    adjacent(i,j)=length(intersect(network[i],network[j]))==1
    for j in 2:5,k in j+1:6
        left=[1,j,k]; right=setdiff(1:6,left)
        all(adjacent(a,b) for a in left for b in left if a<b) || continue
        all(adjacent(a,b) for a in right for b in right if a<b) || continue
        candidates=[(label,a,b) for a in left for b in right
                    for label in intersect(network[a],network[b])]
        isempty(candidates) && continue
        sort!(candidates;by=x->(-dimensions[x[1]],x[1]))
        label,a,b=first(candidates)
        lrest=setdiff(left,[a]); rrest=setdiff(right,[b])
        tree=[[[a,lrest[1]],lrest[2]],[[b,rrest[1]],rrest[2]]]
        order=first(TensorKit.TensorOperations.tree2indexorder(tree,network))
        return (;label,a,b,order)
    end
    nothing
end

"Exact sector/degeneracy sum on one edge; the odd wire is never removed."
function _sliced_junction_close(tensors,network,plan)
    i=findfirst(==(plan.label),network[plan.a])
    j=findfirst(==(plan.label),network[plan.b])
    V=space(tensors[plan.a],i)
    space(tensors[plan.b],j)==dual(V) || throw(DimensionMismatch("sliced edge arrows"))
    work=copy(tensors)
    parts=ComplexF64[]
    for q in sectors(V), index in 1:dim(V,q)
        work[plan.a]=_junction_port_slice(tensors[plan.a],i,q,index)
        work[plan.b]=_junction_port_slice(tensors[plan.b],j,q,index)
        push!(parts,ComplexF64(ncon(work,network;order=plan.order)))
    end
    # Pairwise summation after ordering by magnitude retains all complex
    # contributions. Slicing is exact algebra, not a Schmidt truncation.
    sort!(parts;by=abs)
    value=sum(parts)
    (;value,slices=length(parts),cancellation=sum(abs,parts)/max(abs(value),floatmin(Float64)))
end

"Close an explicitly directed regional junction; every edge occurs twice."
function _close_replica_junction(endpoints,specs,k)
    tensors=Any[]; network=Vector{Int}[]
    owner=Dict((X,c)=>3(c-1)+x for (x,X) in enumerate((:A,:B,:C)) for c in 1:k)
    occurrences=Dict{Int,Vector{Tuple{Int,Int}}}()
    for (j,(endpoint,spec)) in enumerate(zip(endpoints,specs))
        labels=[owner[(X,c)] for c in spec.copies for X in (spec.X,spec.Y)]
        numout(endpoint)==length(labels) && numin(endpoint)==0 || throw(DimensionMismatch("junction endpoint ports"))
        for (i,label) in enumerate(labels)
            push!(get!(occurrences,label,Tuple{Int,Int}[]),(j,i))
        end
        push!(tensors,endpoint); push!(network,labels)
    end
    length(occurrences)==3k || error("junction has missing regional replica edges")
    for ports in values(occurrences)
        length(ports)==2 || error("junction edge does not occur twice")
        (j,i),(jj,ii)=ports
        space(tensors[j],i)==dual(space(tensors[jj],ii)) || throw(DimensionMismatch("junction arrows do not match"))
    end
    # Replica-number order can join antipodal vertices too early and create
    # unnecessarily large intermediates. Optimize only the contraction tree;
    # all tensors, graded arrows, crossings and logical edges stay intact.
    dimensions=Dict(label=>Float64(dim(space(t,j))) for (t,ls) in zip(tensors,network)
                    for (j,label) in enumerate(ls))
    # A triangle contraction otherwise carries six full chi indices. Keep
    # the parity charge of a selected edge, and sum its degeneracy exactly,
    # reducing those intermediates to five large indices for large chi.
    if k==4 && maximum(values(dimensions))^6>2.0^26
        plan=_junction_slice_plan(network,dimensions)
        plan!==nothing && return _sliced_junction_close(tensors,network,plan).value
    end
    tree,_=TensorKit.TensorOperations.optimaltree(network,dimensions)
    order=first(TensorKit.TensorOperations.tree2indexorder(tree,network))
    ComplexF64(ncon(tensors,network;order))
end

"""Finite-depth graded replica network, using full seams or cycle endpoints.

`caps` supplies one even TWO-PORT far cap per seam (AB,AC,BC). Every replica
starts from a copy of that same cap. With factorized=true the cube uses the
six endpoints AB12,AB34,AC13,AC24,BC14,BC23; false keeps three eight-port
endpoints as an independent reference. The finite depth and norm prefactors
are retained. No infinite endpoint phase or sector weight is guessed.

This defines a particular finite LMPS rail network. Its physical far-cap
choice and connection to the infinite PEPS still require separate validation.
Optional `centers=(A=...,B=...,C=...)` supplies explicit even two-port
regional centers instead of the default direct junction closure. Each
center's ports follow AB/AC for A, AB/BC for B and AC/BC for C.
"""
function finite_graded_sixr(seams,permutations,caps;depth::Int,factorized=true,
                            insertions=fill(false,3length(first(permutations))),
                            propagation_cache=nothing,centers=nothing)
    length(permutations)==3 || throw(ArgumentError("three regional permutations required"))
    k=length(first(permutations))
    length(insertions)==3k || throw(DimensionMismatch("regional parity insertions"))
    specs=NamedTuple[]; endpoints=Any[]; diagnostics=NamedTuple[]
    logscale=0.0
    for (name,X,Y) in _REGIONAL_SEAMS
        x=findfirst(==(X),(:A,:B,:C)); y=findfirst(==(Y),(:A,:B,:C))
        pX,pY=permutations[x],permutations[y]
        blocks=factorized ? _relative_cycles(pX,pY) : [collect(1:k)]
        railX,railY=getproperty(seams,name)
        onecap=getproperty(caps,name)
        numout(onecap)==2 && numin(onecap)==0 || throw(DimensionMismatch("two-port even far cap required"))
        for copies in blocks
            graph=_graded_seam_graph(railX,railY,pX,pY,copies;
                parity_X=[insertions[3(r-1)+x] for r in 1:k],
                parity_Y=[insertions[3(r-1)+y] for r in 1:k])
            cap=foldl((a,_)->a⊗onecap,2:length(copies);init=onecap)
            key=(objectid(railX),objectid(railY),objectid(onecap),name,Tuple(pX),Tuple(pY),Tuple(copies),
                 Tuple(insertions[3(r-1)+x] for r in copies),
                 Tuple(insertions[3(r-1)+y] for r in copies))
            result=if propagation_cache===nothing
                _propagate_seam(graph,cap,depth)
            else
                previous=get(propagation_cache,key,nothing)
                if previous===nothing || previous.depth>depth
                    _propagate_seam(graph,cap,depth)
                elseif previous.depth==depth
                    previous.result
                elseif previous.result.zero
                    previous.result
                else
                    advanced=_propagate_seam(graph,previous.result.endpoint,depth-previous.depth)
                    merge(advanced,(logscale=advanced.logscale+previous.result.logscale,))
                end
            end
            propagation_cache!==nothing && (propagation_cache[key]=(;depth,result))
            push!(specs,(;name,X,Y,copies))
            push!(endpoints,result.endpoint)
            push!(diagnostics,(;name,copies,lambda=result.lambda,residual=result.residual,zero=result.zero))
            logscale+=result.logscale
        end
    end
    value=centers===nothing ? _close_replica_junction(endpoints,specs,k) :
                             _mps_junction_close(endpoints,specs,centers,k)
    (;value,logscale,endpoints,specs,diagnostics,depth,factorized)
end

"Combine scaled complex values without discarding destructive interference."
function _scaled_complex_sum(values)
    active=filter(x->!iszero(x.value),values)
    isempty(active) && return (value=0.0im,logscale=0.0,cancellation_condition=Inf)
    scale=maximum(x.logscale+log(abs(x.value)) for x in active)
    value=sum(x.value*exp(x.logscale-scale) for x in active)
    cancellation_condition=sum(abs(x.value)*exp(x.logscale-scale) for x in active)/abs(value)
    (;value,logscale=scale,cancellation_condition)
end

"Five scaled sectors of the same finite rail network and far caps."
function finite_lmps_sectors(seams,caps;depth::Int,factorized=true,propagation_cache=nothing)
    definitions=(Z1=((1,),(1,),(1,)),Z2A=((2,1),(1,2),(1,2)),
        Z2B=((1,2),(2,1),(1,2)),Z2C=((1,2),(1,2),(2,1)),
        Z4=((1,2,3,4),(2,1,4,3),(3,4,1,2)))
    NamedTuple{keys(definitions)}(Tuple(finite_occupation_sixr(seams,p,caps;depth,factorized,propagation_cache)
                                       for p in values(definitions)))
end

"""Read entropies only from nonzero, phase-consistent, well-resolved sectors.

This numerical gate does not certify a physical far cap or the PEPS/LMPS
approximation. It prevents complex/negative contractions being turned into
plausible entropies by blindly taking log(abs(Z)).
"""
function lmps_sector_entropies(sectors;phase_tolerance=1e-8,max_cancellation=1e10)
    isfinite(phase_tolerance) && phase_tolerance>0 || throw(ArgumentError("positive finite phase tolerance required"))
    isfinite(max_cancellation) && max_cancellation>=1 || throw(ArgumentError("finite cancellation bound >= 1 required"))
    all(s->!iszero(s.value)&&isfinite(abs(s.value))&&isfinite(s.logscale),values(sectors)) ||
        error("finite LMPS has a zero or nonfinite sector")
    all(s->s.cancellation_condition<=max_cancellation,values(sectors)) ||
        error("parity-sector cancellation is numerically unresolved")
    logmag(s)=log(abs(s.value))+s.logscale
    phase(s)=s.value/abs(s.value)
    z1=sectors.Z1
    two=(sectors.Z2A,sectors.Z2B,sectors.Z2C)
    phase2=[phase(z)/phase(z1)^2 for z in two]
    phase4=phase(sectors.Z4)/phase(z1)^4
    max(maximum(abs.(phase2 .- 1)),abs(phase4-1))<=phase_tolerance ||
        error("LMPS sectors have invalid physical phases: Z2=$phase2, Z4=$phase4")
    S2=[2logmag(z1)-logmag(z) for z in two]
    S3=(4logmag(z1)-logmag(sectors.Z4))/2
    (;S2,S3,stilde=2S3-sum(S2),phase2,phase4)
end

"""Occupation parity-sector sum on the explicitly supplied finite rail graph.

This inserts virtual ket parity strings on both incident seams of each region.
It is a candidate PEPS pull-through construction, not a certification of a
physical outer cap. The direct/full and six-R/cycle routes share this definition.
"""
function finite_occupation_sixr(seams,permutations,caps;depth::Int,factorized=true,propagation_cache=nothing,centers=nothing)
    expansion=replica_parity_expansion(permutations)
    sectors=NamedTuple[]
    for term in expansion.terms
        result=finite_graded_sixr(seams,permutations,caps;depth,factorized,insertions=term.insertions,propagation_cache,centers)
        push!(sectors,(;weight=term.weight,insertions=term.insertions,result))
    end
    total=_scaled_complex_sum([(;value=s.weight*s.result.value,logscale=s.result.logscale) for s in sectors])
    (;total...,sectors,depth,factorized)
end
