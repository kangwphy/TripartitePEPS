# Finite-window region contraction. All physical cut legs remain
# separate ket/bra FermionParity factors; only auxiliary MPS bonds are fused.

function _finite_vacuum_frontier(A,width)
    V=Vect[FermionParity](0=>1,1=>0)
    vk=_vacuum(dual(space(A,2)))
    vb=_vacuum(space(A,2))
    u=id(ComplexF64,V)
    @tensor M[l k b;r] := u[l;r]*vk[k]*vb[b]
    MPSKit.FiniteMPS([copy(M) for _ in 1:width];normalize=false)
end

function _finite_grow_frontier(frontier,A,width;keep_east=true)
    old=[i==length(frontier) ? frontier.AC[i] : frontier.AL[i] for i in 1:length(frontier)]
    grown=typeof(first(old))[]
    tail=nothing
    for i in 1:width
        D=_grow_boundary(old[i],A,A)
        # Parallel ket/bra cups become an auxiliary TensorMap composition.
        # Its pivotal correction is on the dual strand; after a 180-degree
        # rotation that is the bra strand, not the ket strand.
        D=PEPSKit.twistdual(D,(2,3))
        Q=permute(D,((1,2,3,4,5),(6,7,8)))
        L=space(Q,1)⊗space(Q,2)⊗space(Q,3)
        UL=if i==1
            id(ComplexF64,space(Q,1))⊗_vacuum(space(Q,2))'⊗
                _vacuum(space(Q,3))'
        else
            isomorphism(ComplexF64,fuse(L)←L)
        end
        R=domain(Q)
        if i==width && !keep_east
            VR=id(ComplexF64,R[1])⊗_vacuum(R[2])⊗_vacuum(R[3])
            push!(grown,(UL⊗id(ComplexF64,space(Q,4))⊗id(ComplexF64,space(Q,5)))*Q*VR)
        else
            UR=isomorphism(ComplexF64,fuse(R)←R)
            push!(grown,(UL⊗id(ComplexF64,space(Q,4))⊗id(ComplexF64,space(Q,5)))*Q*UR')
            if i==width
                N=permute(id(ComplexF64,R),((1,2,3,5,6),(4,)))
                tail=(UR⊗id(ComplexF64,space(N,4))⊗id(ComplexF64,space(N,5)))*N
            end
        end
    end
    keep_east && push!(grown,tail)
    append!(grown,old[width+1:end])
    MPSKit.FiniteMPS(grown;normalize=false)
end

"""Contract a rectangular norm region as an L-shaped finite boundary MPS.

The external north/west boundaries are virtual vacuum. The frontier is the
south edge left-to-right followed by the east edge bottom-to-top. Set
`keep_east=false` to close the east edge in vacuum. `chi` is the total retained
even-plus-odd auxiliary dimension; `nothing` disables SVD truncation.
`discarded` records the fractional squared-norm loss per row, not an entropy
error bound. All ket/bra cut legs remain separate FermionParity factors.
"""
function finite_region_mps(A,width::Int,height::Int;keep_east=true,chi=nothing)
    width>0 && height>0 || throw(ArgumentError("positive region dimensions required"))
    (chi===nothing || (chi isa Int && chi>0)) || throw(ArgumentError("chi must be a positive total rank or nothing"))
    numout(A)==1 && numin(A)==4 && sectortype(A)==FermionParity ||
        throw(ArgumentError("expected a FermionParity PEPS tensor physical ← north,east,south,west"))
    all(i->norm(_vacuum(space(A,i)))==1,2:5) ||
        throw(ArgumentError("virtual vacuum must have a unique even basis state"))
    state=_finite_vacuum_frontier(A,width)
    logscale=0.0; discarded=Float64[]; ranks=Int[]
    for row in 1:height
        state=_finite_grow_frontier(state,A,width;keep_east)
        before=norm(state)
        isfinite(before) && before>0 || error("nonfinite or zero norm region after row $row")
        if chi!==nothing
            MPSKit.changebonds!(state,MPSKit.SvdCut(;trscheme=truncrank(chi));normalize=false)
        end
        after=norm(state)
        isfinite(after) && after>0 || error("nonfinite or zero norm region after truncating row $row")
        after<=before*(1+1e-10) || error("SVD truncation increased region norm at row $row")
        push!(discarded,max(0,1-(after/before)^2))
        push!(ranks,maximum(dim(space(state.AL[i],1)) for i in 1:length(state)))
        logscale+=log(after)
        normalize!(state)
    end
    (;state,logscale,discarded,ranks,width,height,keep_east,chi)
end

# Finite, position-dependent seam contraction with explicit regional centers.
function mps_junction_geometry(a,b,c)
    a.keep_east && b.keep_east && !c.keep_east || throw(ArgumentError("A/B need an east frontier; C needs an east vacuum"))
    a.height==b.width==c.height && c.width==a.width+b.height ||
        throw(DimensionMismatch("expected upper-left/right rectangles and a lower-half rectangle"))
    A,B,C=a.state,b.state,c.state
    left,right,halfheight=a.width,b.height,a.height
    flip(M)=permute(M,((4,2,3),(1,)))
    seams=(AB=[(A.AR[left+halfheight-i+1],flip(B.AL[i])) for i in 1:halfheight],
           AC=[(flip(A.AL[i]),C.AR[left+right-i+1]) for i in 1:left],
           BC=[(B.AR[halfheight+right-i+1],flip(C.AL[i])) for i in 1:right])
    centers=(A=permute(A.C[left],((2,1),())),
             B=permute(B.C[halfheight],((1,2),())),
             C=permute(C.C[right],((2,1),())))
    (;seams,centers,logscale=a.logscale+b.logscale+c.logscale)
end

"""Physical occupation-replica entropy on a width×L virtual-vacuum window.

Uses a translation-invariant 1×1 iPEPS tensor, explicit finite region centers,
and position-dependent seams. A/B are the upper-left/upper-right quadrants;
C is the lower half. This returns a finite-window, finite-chi measurement.
`L` is even; `width=L` and `cut=width÷2` by default. Odd widths and an unequal
upper partition support closures such as the nonzero critical 3×2 KSVC patch.
Convergence of size and chi must be established separately; a phase check alone
does not establish either limit. There is one tripartite junction.
"""
function measure_finite_fpeps_stilde(peps=ksvc_ipeps();L::Int,width::Int=L,cut::Int=width÷2,chi=nothing,
        phase_tolerance=1e-8,on_region=(_...)->nothing,on_sector=(_...)->nothing)
    iseven(L) && L>=2 || throw(ArgumentError("even L>=2 required"))
    width>=2 && 0<cut<width || throw(ArgumentError("width>=2 and 0<cut<width required"))
    isfinite(phase_tolerance) && phase_tolerance>0 || throw(ArgumentError("positive finite phase tolerance required"))
    size(peps)==(1,1) || throw(ArgumentError("finite-window sewing currently requires a 1×1 unit cell"))
    regions=map((:A,:B,:C)) do X
        A=X==:A ? peps[1] : X==:B ? rotl90(peps)[1] : rot180(peps)[1]
        region_width=X==:A ? cut : X==:B ? L÷2 : width
        region_height=X==:B ? width-cut : L÷2
        m=finite_region_mps(A,region_width,region_height;keep_east=X!=:C,chi)
        on_region(X,m)
        m
    end
    geometry=mps_junction_geometry(regions...)
    sectors=finite_mps_sectors(geometry;on_sector)
    entropies=lmps_sector_entropies(sectors;phase_tolerance)
    norm_phase_error=abs(sectors.Z1.value/abs(sectors.Z1.value)-1)
    norm_phase_error<=phase_tolerance || error("finite physical norm is not real and positive")
    all(entropies.S2 .>= -phase_tolerance) || error("finite purity exceeds one")
    diagnostics=(max_row_discarded=maximum(maximum(m.discarded) for m in regions),
        max_retained_rank=maximum(maximum(m.ranks) for m in regions),
        norm_phase_error,
        max_phase_error=max(maximum(abs.(entropies.phase2.-1)),abs(entropies.phase4-1)),
        max_cancellation=maximum(z.cancellation_condition for z in values(sectors)),
        size_converged=false,chi_converged=false)
    (;stilde=entropies.stilde,entropies,sectors,regions,diagnostics,L,width,cut,chi,
      method=:finite_region_mps,regulator=:virtual_vacuum_obc,junctions=1)
end

function _mps_junction_close(endpoints,specs,centers,k)
    owner(X,c,side)=6(c-1)+2(findfirst(==(X),(:A,:B,:C))-1)+side
    ports=Dict((:AB,:A)=>1,(:AB,:B)=>1,(:AC,:A)=>2,(:AC,:C)=>1,(:BC,:B)=>2,(:BC,:C)=>2)
    ts=Any[]; labels=Vector{Int}[]
    for (R,s) in zip(endpoints,specs)
        push!(ts,R)
        push!(labels,[owner(X,c,ports[(s.name,X)]) for c in s.copies for X in (s.X,s.Y)])
    end
    for c in 1:k,X in (:A,:B,:C)
        push!(ts,getproperty(centers,X))
        push!(labels,[owner(X,c,1),owner(X,c,2)])
    end
    # First absorb each rank-two center into one endpoint. Then optimize the
    # remaining (at most six) endpoint graph. Interleaving the two center legs
    # by replica number builds a chi^12 intermediate already at k=4.
    reduced=deepcopy(labels[1:length(endpoints)])
    order=Int[]
    for pair in labels[length(endpoints)+1:end]
        a,b=pair
        i=findfirst(xs->a in xs,reduced)
        reduced[i][findfirst(==(a),reduced[i])]=b
        push!(order,a)
    end
    dimensions=Dict(label=>Float64(dim(space(t,j))) for (t,ls) in zip(ts,labels)
                    for (j,label) in enumerate(ls))
    tree,_=TensorKit.TensorOperations.optimaltree(reduced,dimensions)
    append!(order,first(TensorKit.TensorOperations.tree2indexorder(tree,reduced)))
    ComplexF64(ncon(ts,labels;order))
end

function _finite_mps_graded(geometry,permutations,insertions,cache)
    k=length(first(permutations)); endpoints=Any[]; specs=NamedTuple[]
    logscale=k*geometry.logscale
    for (name,X,Y) in _REGIONAL_SEAMS
        x,y=findfirst(==(X),(:A,:B,:C)),findfirst(==(Y),(:A,:B,:C))
        px,py=permutations[x],permutations[y]
        q=invperm(collect(px))[collect(py)]
        chain=getproperty(geometry.seams,name)
        for copies in _relative_cycles(px,py)
            localq=Tuple(findfirst(==(q[c]),copies) for c in copies)
            parx=Tuple(insertions[3(c-1)+x] for c in copies)
            pary=Tuple(insertions[3(c-1)+y] for c in copies)
            key=(name,localq,parx,pary)
            result=get!(cache,key) do
                n=length(copies); idperm=Tuple(1:n)
                firstX,firstY=first(chain)
                onecap=_vacuum(dual(space(firstX,4)))⊗
                       _vacuum(dual(space(firstY,4)))
                endpoint=foldl((a,_)->a⊗onecap,2:n;init=onecap)
                scale=0.0
                for (railX,railY) in chain
                    graph=_graded_seam_graph(railX,railY,idperm,localq,collect(1:n);
                        parity_X=parx,parity_Y=pary,stationary=false)
                    endpoint=_apply_seam_graph(graph,endpoint)
                    s=norm(endpoint)
                    iszero(s) && return (;endpoint,logscale=0.0,zero=true)
                    endpoint/=s; scale+=log(s)
                end
                (;endpoint,logscale=scale,zero=false)
            end
            result.zero && return (value=0.0im,logscale=0.0)
            push!(endpoints,result.endpoint)
            push!(specs,(;name,X,Y,copies))
            logscale+=result.logscale
        end
    end
    value=_mps_junction_close(endpoints,specs,geometry.centers,k)
    (;value,logscale)
end

function finite_mps_sectors(geometry;on_sector=(_...)->nothing)
    definitions=(Z1=((1,),(1,),(1,)),Z2A=((2,1),(1,2),(1,2)),
        Z2B=((1,2),(2,1),(1,2)),Z2C=((1,2),(1,2),(2,1)),
        Z4=((1,2,3,4),(2,1,4,3),(3,4,1,2)))
    cache=Dict{Any,Any}()
    results=map(collect(pairs(definitions))) do (name,permutations)
        terms=map(replica_parity_expansion(permutations).terms) do term
            r=_finite_mps_graded(geometry,permutations,term.insertions,cache)
            (;weight=term.weight,insertions=term.insertions,result=r)
        end
        total=_scaled_complex_sum([(;value=t.weight*t.result.value,
                                                logscale=t.result.logscale) for t in terms])
        result=(;total...,sectors=terms)
        on_sector(name,result)
        result
    end
    NamedTuple{keys(definitions)}(Tuple(results))
end
