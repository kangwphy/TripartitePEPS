"""Opt-in CUDA bare/direct experiment. Never imported by LMPSVUMPS itself."""
module BareCUDA
using CUDA, Adapt, ITensors, LinearAlgebra, Random, KrylovKit
import cuTENSOR, TensorKit
import LMPSVUMPS
import LMPSVUMPS: theind, pairtag, ket, bra
include(joinpath(@__DIR__, "../sixr.jl"))
function device_array(t, ix...)
    a = ITensors.array(t, ix...)
    a isa CuArray || error("unexpected host tensor in CUDA contraction: $(typeof(a))")
    a
end
# Explicit named contractions preserve a[north,west,south,east] and the
# transpose (not adjoint) convention of the authoritative CPU algorithm.
function device_direct_maps(ML, MR, a)
    all(x -> x isa CuArray, (ML,MR,a)) || error("CUDA inputs required")
    chi, dp, _ = size(ML)
    l=Index(chi); r=Index(chi); lo=Index(chi); ro=Index(chi)
    n=Index(dp); w=Index(dp); s=Index(dp); e=Index(dp)
    L=ITensor(ML,l,w,lo); R=ITensor(MR,r,e,ro)
    bulk=ITensor(a,n,w,s,e)
    Lg=ITensor(ML,l,w,lo); Rg=ITensor(MR,r,w,ro)
    apply_G = X -> device_array((Lg*ITensor(X,l,r))*Rg,lo,ro)
    apply_B = X -> device_array(((L*ITensor(X,l,n,r))*bulk)*R,lo,s,ro)
    apply_G, apply_B
end
include("sixr_cuda.jl")
include("direct_cuda.jl")
include("bridge_cuda.jl")
function gpu_sliced_octahedron(Rs,vertices;hinges=Dict{String,Matrix{Float64}}())
    out=m2_shared_tensors(Rs,vertices;hinges)
    li=[m2_vertex_index(vertices,n) for n in (:AB12,:AC13,:BC14)]
    ri=[m2_vertex_index(vertices,n) for n in (:AB34,:AC24,:BC23)]
    unmatched(a,b)=symdiff(Set(inds(a)),Set(inds(b)))
    edges=intersect(unmatched(out[li[1]],out[li[2]]),unmatched(out[ri[1]],out[ri[2]]))
    edge=first(sort!(collect(edges);by=m2_edge_key))
    owners=findall(R->hasind(R,edge),out)
    length(owners)==2 || error("sliced edge ownership")
    z=CUDA.zeros(promote_type(map(eltype,out)...),1); peak=0
    for value in 1:dim(edge)
        sliced=copy(out)
        selector=ITensor(Float64.(CuArray(1:dim(edge)) .== value),edge)
        for owner in owners
            sliced[owner] *= selector
        end
        left=sliced[li[1]]*sliced[li[2]]
        peak=max(peak,prod(dim.(inds(left))))
        left*=sliced[li[3]]
        peak=max(peak,prod(dim.(inds(left))))
        right=sliced[ri[1]]*sliced[ri[2]]
        peak=max(peak,prod(dim.(inds(right))))
        right*=sliced[ri[3]]
        peak=max(peak,prod(dim.(inds(right))))
        result=left*right
        order(result)==0 || error("open closure")
        z .+= reshape(device_array(result),1)
    end
    only(Array(z)),peak # one scalar result transfer after all device contractions
end
end
