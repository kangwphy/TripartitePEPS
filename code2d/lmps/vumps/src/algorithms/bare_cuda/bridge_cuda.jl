# Dense TensorKit extraction for the ungraded ComplexSpace TFIM only.
# t[] exposes all uncoupled axes in TensorKit's Trivial-sector implementation.
function device_tensorkit(t)
    TensorKit.sectortype(t) === TensorKit.Trivial || error("only ungraded TFIM supported")
    t.data isa CuArray || error("TensorKit backing storage is not on the GPU")
    # TensorKit exposes a StridedView over a one-dimensional CUDA backing
    # buffer. Base.copy would iterate that view on the host; reshape the parent
    # buffer directly instead, preserving TensorKit's linear sector order.
    view=t[]
    buf=parent(view)
    buf isa CuArray || error("TensorKit view parent is not device storage: $(typeof(buf))")
    a=reshape(buf,size(view))
    a isa CuArray || error("TensorKit reshape returned host storage: $(typeof(a))")
    a
end
function boundary_array(b)
    length(b.transfer)==1 || error("one-site PEPS required")
    raw=device_tensorkit(b.state.AL[1])
    ndims(raw) in (3,4) || error("unexpected boundary rank")
    reshape(raw,size(raw,1),prod(size(raw)[2:end-1]),size(raw,ndims(raw)))
end
function transfer_array(b)
    top,bottom=b.transfer[1]
    A=device_tensorkit(top); B=device_tensorkit(bottom)
    size(A)==size(B) || error("ket/bra shape mismatch")
    d,n,e,s,w=size(A)
    n==e==s==w || error("uniform bonds required")
    p=Index(d); nk=Index(n); nb=Index(n); ek=Index(n); eb=Index(n)
    sk=Index(n); sb=Index(n); wk=Index(n); wb=Index(n)
    dbl=ITensor(A,p,nk,ek,sk,wk)*ITensor(conj(B),p,nb,eb,sb,wb)
    reshape(device_array(dbl,nk,nb,wk,wb,sk,sb,ek,eb),n^2,n^2,n^2,n^2)
end
