"""Compile the directed *graded* seam graph for a closed replica subset.

Rails have ports (chi_left, ket, bra; chi_right). Opposite rails must
already have dual physical spaces; no reflection or arrow change is guessed.
The returned rail order is X_i,Y_i for i in copies. Regional occupation
parity corrections and the physical far cap are separate inputs to the full
calculation, not silently replaced by a generic seed.
"""
function _graded_seam_graph(X,Y,sigmaX,sigmaY,copies;
                            parity_X=fill(false,length(sigmaX)),
                            parity_Y=fill(false,length(sigmaY)),stationary=true)
    k=length(sigmaX)
    sort(collect(sigmaX))==sort(collect(sigmaY))==collect(1:k) ||
        throw(ArgumentError("invalid regional permutations"))
    !isempty(copies) && allunique(copies) && all(i->i in 1:k,copies) ||
        throw(ArgumentError("invalid copy subset"))
    q=invperm(collect(sigmaX))[collect(sigmaY)]
    Set(q[collect(copies)])==Set(copies) || throw(ArgumentError("subset is not closed under relative permutation"))
    length(parity_X)==length(parity_Y)==k || throw(DimensionMismatch("parity insertions"))
    for rail in (X,Y)
        numout(rail)==3 && numin(rail)==1 || throw(DimensionMismatch("rail ports must be (chi,ket,bra;chi)"))
        sectortype(rail)==FermionParity || throw(ArgumentError("fermionic rail required"))
        (!stationary || space(rail,1)==dual(space(rail,4))) ||
            throw(DimensionMismatch("stationary chi spaces required"))
    end
    all(i->space(X,i)==dual(space(Y,i)),(2,3)) ||
        throw(DimensionMismatch("opposite rail physical spaces must be dual"))
    n=2length(copies)
    wire=Dict{Tuple{Symbol,Int},Int}()
    for c in copies,label in ((:ket,c),(:bra,sigmaX[c]))
        wire[label]=n+length(wire)+1
    end
    tensors=Any[]; labels=Vector{Vector{Int}}()
    for c in copies,(rail,sigma,parities) in ((X,sigmaX,parity_X),(Y,sigmaY,parity_Y))
        i=length(tensors)+1
        push!(tensors,parities[c] ? twist(rail,2) : rail)
        push!(labels,[-i,wire[(:ket,c)],wire[(:bra,sigma[c])],i])
    end
    (;tensors,labels,n,contraction_order=collect(1:n+length(wire)))
end

"""Apply a full graded seam transfer to an explicitly supplied cap/endpoint.

No chi^(2n) transfer matrix is formed: seed bonds are contracted first.
Parity insertions act on ket virtual strands and retain their actual physical
legs. This primitive does not by itself supply occupation-sector weights or
prove that the selected infinite endpoint factors into two cycles.
"""
function graded_seam_apply(X,Y,sigmaX,sigmaY,endpoint;
                           copies=collect(eachindex(sigmaX)),kwargs...)
    graph=_graded_seam_graph(X,Y,sigmaX,sigmaY,copies;kwargs...)
    _apply_seam_graph(graph,endpoint)
end

function _apply_seam_graph(graph,endpoint)
    numin(endpoint)==0 && numout(endpoint)==graph.n || throw(DimensionMismatch("endpoint ports"))
    all(i->space(endpoint,i)==dual(space(graph.tensors[i],4)),1:graph.n) ||
        throw(DimensionMismatch("endpoint order must be X_i,Y_i for i in copies"))
    tensors=Any[endpoint]
    append!(tensors,graph.tensors)
    network=Vector{Vector{Int}}([collect(1:graph.n)])
    append!(network,graph.labels)
    ncon(tensors,network;order=graph.contraction_order)
end

"Explicit coordinate identity far cap, with directed spaces and parity blocks."
function seam_identity_cap(X,Y)
    cap=zeros(ComplexF64,space(X,1)⊗space(Y,1)←one(space(X,1)))
    for (fo,fi) in fusiontrees(cap)
        data=cap[fo,fi]
        for i in CartesianIndices(data)
            data[i]=i[1]==i[2] ? 1 : 0
        end
    end
    norm(cap)>0 || throw(ArgumentError("no common sector for identity cap"))
    cap
end

"""Leading graded seam eigenvectors with explicit gap and phase diagnostics.

The phase of the leading vector is tied to the supplied seed. `isolated`
reports separation in eigenvalue magnitude: a degenerate leading subspace
must be resolved by the physical far cap, not by this routine's vector order.
"""
function leading_seam_endpoint(X,Y,pX,pY,seed;copies=collect(eachindex(pX)),
                                tolerance=1e-10,maxiter=300,krylovdim=30,
                                gap_tolerance=1e-8,dense_threshold=128,kwargs...)
    graph=_graded_seam_graph(X,Y,pX,pY,copies;kwargs...)
    isfinite(tolerance) && tolerance>0 || throw(ArgumentError("positive finite seam tolerance required"))
    norm(seed)>0 || throw(ArgumentError("nonzero seam seed required"))
    action=endpoint->_apply_seam_graph(graph,endpoint)
    # Count the even intertwiner space, not the product of dense leg dimensions.
    dimension=dim(space(seed))
    nev=min(2,dimension)
    eigenvalues,vectors,info=if dimension<=dense_threshold
        # Matrix coordinates of the EVEN morphism space. Every basis vector
        # is a typed TensorMap; no ungraded physical-array contraction is used.
        # A single Krylov seed can miss eigenvalues in an invariant subspace.
        matrix=zeros(ComplexF64,dimension,dimension)
        for j in 1:dimension
            basis=zeros(ComplexF64,space(seed)); basis.data[j]=1
            matrix[:,j]=action(basis).data
        end
        decomposition=eigen(matrix)
        order=sortperm(abs.(decomposition.values);rev=true)
        vals=decomposition.values[order]
        vecs=[TensorMap(copy(decomposition.vectors[:,j]),space(seed)) for j in order]
        (vals,vecs,(converged=dimension,numiter=1,numops=dimension))
    else
        KrylovKit.eigsolve(action,seed/norm(seed),nev,:LM;
            ishermitian=false,tol=tolerance,maxiter,krylovdim=min(krylovdim,dimension))
    end
    isempty(eigenvalues) && error("no seam eigenpair returned")
    endpoint=vectors[1]/norm(vectors[1]); lambda=eigenvalues[1]
    isfinite(abs(lambda)) && abs(lambda)>0 || error("seam has no nonzero finite leading eigenvalue")
    overlap=dot(seed,endpoint)
    abs(overlap)>1e-12*norm(seed) || error("seam leading vector has negligible overlap with supplied seed")
    endpoint*=conj(overlap)/abs(overlap)
    residual=norm(action(endpoint)-lambda*endpoint)/max(abs(lambda),eps())
    gap=dimension==1 ? 1.0 : length(eigenvalues)>=2 ? 1-abs(eigenvalues[2])/abs(lambda) : NaN
    converged=info.converged>=1 && residual<=max(10tolerance,1e-10)
    converged || error("seam eigenvector failed residual check: $residual")
    (;endpoint,lambda,residual,gap,isolated=isfinite(gap)&&gap>gap_tolerance,
      eigenvalues,iterations=info.numiter,operations=info.numops)
end
