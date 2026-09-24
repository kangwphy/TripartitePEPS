"Multiply exterior polynomials with canonical generators [c†,δ,β,γ,α]."
function _exterior_product(a, b)
    out = Dict{UInt,ComplexF64}()
    for (u, x) in a, (v, y) in b
        iszero(u & v) || continue
        inversions = sum(count_ones(v & ((UInt(1) << i) - 1))
                         for i in 0:4 if !iszero(u & (UInt(1) << i)); init=0)
        word = u | v
        out[word] = get(out, word, 0.0im) + (isodd(inversions) ? -x*y : x*y)
    end
    filter!(p -> !iszero(last(p)), out)
    out
end

"Exact unnormalized coefficients Q = Σ A[k,d,r,u,l] c†ᵏ δᵈ βʳ γᵘ αˡ."
function ksvc_coefficients()
    # Canonically sorted quadratic monomials in F, derived from the defining Q.
    terms = ((0,1,1im), (0,2,-1), (0,3,-1), (0,4,-1im),
             (1,2,-1im), (1,3,-1), (1,4,1), (2,3,-1), (2,4,-1), (3,4,1im))
    F = Dict((UInt(1)<<i)|(UInt(1)<<j) => ComplexF64(c) for (i,j,c) in terms)
    F2 = _exterior_product(F, F)
    @assert isempty(_exterior_product(F2, F))
    Q = mergewith(+, Dict(UInt(0)=>1.0+0im), F, Dict(w=>c/2 for (w,c) in F2))
    A = zeros(ComplexF64, 2,2,2,2,2)
    for (w,c) in Q
        indices = ntuple(i -> Int((w >> (i-1)) & 1) + 1, 5)
        A[indices...] = c
    end
    A
end

"""Local even map physical ← (α,β,γ,δ) in ascending virtual Fock order.

The annihilation word δ β γ α differs from the reversed Fock dual
δ γ β α by the β/γ crossing: (-1)^(r*u). No bond has been absorbed here.
"""
function ksvc_projector()
    V = Vect[FermionParity](0=>1, 1=>1)
    A = ksvc_coefficients()
    Q = zeros(ComplexF64, V ← V ⊗ V ⊗ V ⊗ V)
    for (fo, fi) in fusiontrees(Q)
        k = Int(fo.uncoupled[1].isodd)
        l,r,u,d = Int.(getproperty.(fi.uncoupled, :isodd))
        Q[fo,fi] .= (-1)^(r*u) * A[k+1,d+1,r+1,u+1,l+1]
    end
    Q
end

"Unnormalized even Bell pair (1 + f₁† f₂†)|0⟩, with the written wire order."
function _bell(V,weight=1)
    B = zeros(ComplexF64, V ⊗ V ← one(V))
    for (fo,fi) in fusiontrees(B)
        B[fo,fi] .= fo.uncoupled[1].isodd ? weight : 1
    end
    B
end

"""KSVC PEPSKit tensor physical ← (north,east,south,west).

Absorb the oriented β→α and δ→γ Bell bonds using typed contractions.
Domain spaces are (V,V',V',V); the arrows are retained by TensorKit.
`bond_gauge=:balanced` shares each positive real bond weight equally between
its two endpoints. The historical default is `:outgoing`.
"""
function ksvc_tensor(;bond_weight=(1,1),bond_gauge=:outgoing)
    if bond_gauge==:balanced
        length(bond_weight)==2 && all(w->isreal(w)&&isfinite(w)&&real(w)>0,bond_weight) ||
            throw(ArgumentError("balanced gauge requires two positive finite real weights"))
        A=ksvc_tensor()
        for (fo,fi) in fusiontrees(A)
            n,e,s,w=Int.(getproperty.(fi.uncoupled,:isodd))
            A[fo,fi] .*= real(bond_weight[1])^((e+w)/2)*real(bond_weight[2])^((n+s)/2)
        end
        return A
    end
    bond_gauge==:outgoing || throw(ArgumentError("bond_gauge must be :outgoing or :balanced"))
    Q = ksvc_projector()
    Bx,By = (_bell(space(Q,1),w) for w in bond_weight)
    @tensor A[p; n e s w] := Q[p; w x n y] * Bx[x e] * By[y s]
    A
end

ksvc_ipeps(; unitcell=(1,1),bond_weight=(1,1),bond_gauge=:outgoing) =
    InfinitePEPS(ksvc_tensor(;bond_weight,bond_gauge);unitcell)

"Vacuum vector of a single virtual mode, including its dual orientation."
function _vacuum(V)
    t = zeros(ComplexF64, V ← one(V))
    for (fo,fi) in fusiontrees(t)
        t[fo,fi] .= 1
    end
    t
end

"""Exact finite typed PEPS contraction: open vacuum boundary or periodic torus.

Physical output order is row-major (x fastest), with mode s stored as bit s-1.
Array conversion occurs only after all virtual legs have been contracted.
Periodic dimensions must be at least three; ±1 twists are applied to outgoing
wrap bonds. A closure may produce the zero vector and must not be normalized.
"""
function finite_tensor(A, nx::Int, ny::Int; maxsites=8,periodic=false,twists=(1,1))
    nx > 0 && ny > 0 || throw(ArgumentError("positive patch dimensions required"))
    nx*ny <= maxsites || throw(ArgumentError("finite oracle exceeds maxsites=$maxsites"))
    periodic && min(nx,ny)<3 && throw(ArgumentError("periodic dimensions must be >= 3"))
    all(x->x in (-1,1),twists) || throw(ArgumentError("twists must be ±1"))
    site(x,y) = x + nx*(y-1)
    tensors = Any[]
    labels = Vector{Vector{Int}}()
    edges = Dict{Tuple{Int,Int},Int}()
    nextedge = 0
    for y in 1:ny, x in 1:nx
        s = site(x,y)
        ix = [-s]
        for (leg,(dx,dy)) in enumerate(((0,-1),(1,0),(0,1),(-1,0)))
            xx,yy = x+dx,y+dy
            if periodic
                xx,yy=mod1(xx,nx),mod1(yy,ny)
            end
            if 1 <= xx <= nx && 1 <= yy <= ny
                ss = site(xx,yy)
                key = minmax(s,ss)
                if !haskey(edges,key)
                    nextedge += 1
                    edges[key] = nextedge
                end
                push!(ix,edges[key])
            else
                nextedge += 1
                push!(ix,nextedge)
                push!(tensors,_vacuum(dual(space(A,leg+1))))
                push!(labels,[nextedge])
            end
        end
        localtensor=A
        periodic && x==nx && twists[1]==-1 && (localtensor=twist(localtensor,3))
        periodic && y==ny && twists[2]==-1 && (localtensor=twist(localtensor,4))
        push!(tensors,localtensor)
        push!(labels,ix)
    end
    ncon(tensors,labels)
end

finite_tensor_state(A,nx,ny;kwargs...) = vec(convert(Array,finite_tensor(A,nx,ny;kwargs...)))
