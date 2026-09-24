"Apply one ordered CAR word to a sparse Fock vector (rightmost operator first)."
function _apply_word(state, word)
    out = Dict{UInt,ComplexF64}()
    for (mask,amplitude) in state
        target = mask
        c = amplitude
        for (i,creation) in reverse(word)
            bit = UInt(1) << i
            occupied = !iszero(target & bit)
            if occupied == creation
                c = 0
                break
            end
            isodd(count_ones(target & (bit-1))) && (c = -c)
            target ⊻= bit
        end
        iszero(c) || (out[target] = get(out,target,0.0im) + c)
    end
    out
end

"""Finite Fock particle-hole transformation, optionally followed by exp(iθN).

The CAR product ∏(c+c†) implements a uniform particle-hole map up to an
irrelevant global sign on c. phase=im also reverses the pairing sign, mapping
the native projector convention to Gaussian/code/cirac2_model.py (up to scale).
"""
function particle_hole_fock(psi::AbstractVector;phase=1)
    N=trailing_zeros(length(psi))
    1<<N==length(psi) || throw(DimensionMismatch("Fock vector dimension"))
    abs(abs(phase)-1)<1e-12 || throw(ArgumentError("phase must have unit modulus"))
    state=Dict(UInt(i-1)=>ComplexF64(c) for (i,c) in enumerate(psi) if !iszero(c))
    for j in 0:N-1
        state=mergewith(+,_apply_word(state,[(j,false)]),_apply_word(state,[(j,true)]))
    end
    out=zeros(ComplexF64,length(psi))
    for (mask,c) in state
        out[Int(mask)+1]=c*phase^count_ones(mask)
    end
    out
end

"""Independent Q/bond CAR oracle, with site order [c,α,β,γ,δ].

This applies the defining operator product, without any TensorKit contraction
or local leg conversion. Open virtual legs are projected onto vacuum.
"""
function finite_fock_state(nx::Int, ny::Int; maxsites=8, periodic=false, twists=(1,1),bond_weight=(1,1))
    nx > 0 && ny > 0 || throw(ArgumentError("positive patch dimensions required"))
    N = nx*ny
    N <= maxsites && 5N < 8sizeof(UInt) || throw(ArgumentError("patch too large for CAR oracle"))
    all(t -> t in (-1,1),twists) || throw(ArgumentError("twists must be ±1"))
    periodic && min(nx,ny)<3 && throw(ArgumentError("periodic CAR oracle requires nx,ny>=3"))
    site(x,y) = x+nx*(y-1)
    mode(s,i) = 5*(s-1)+i
    state = Dict(UInt(0)=>1.0+0im)
    for y in 1:ny, x in 1:nx
        s = site(x,y)
        for (valid,first,last,phase) in (
            (x<nx || periodic,mode(s,2),mode(site(mod1(x+1,nx),y),1),bond_weight[1]*(x==nx ? twists[1] : 1)),
            (y<ny || periodic,mode(s,4),mode(site(x,mod1(y+1,ny)),3),bond_weight[2]*(y==ny ? twists[2] : 1)))
            valid || continue
            created = _apply_word(state,[(first,true),(last,true)])
            if phase != 1
                for mask in keys(created)
                    created[mask] *= phase
                end
            end
            mergewith!(+,state,created)
        end
    end
    A = ksvc_coefficients()
    for s in 1:N
        newstate = Dict{UInt,ComplexF64}()
        for ix in CartesianIndices(A)
            coeff = A[ix]
            iszero(coeff) && continue
            word = Tuple{Int,Bool}[]
            for (power,localmode,creation) in zip(Tuple(ix).-1,(0,4,2,3,1),(true,false,false,false,false))
                power == 1 && push!(word,(mode(s,localmode),creation))
            end
            term = _apply_word(state,word)
            for (mask,c) in term
                newstate[mask] = get(newstate,mask,0.0im) + coeff*c
            end
        end
        virtualmask = sum(UInt(1)<<mode(s,i) for i in 1:4)
        # No later Q acts on this site's virtual modes; project their vacuum
        # immediately to keep this independent global CAR oracle small.
        filter!(p -> !iszero(last(p)) && iszero(first(p)&virtualmask),newstate)
        state = newstate
    end
    physicalmask = sum(UInt(1)<<mode(s,0) for s in 1:N)
    psi = zeros(ComplexF64,1<<N)
    for (mask,c) in state
        iszero(mask & ~physicalmask) || continue
        out = sum(Int((mask>>mode(s,0))&1)<<(s-1) for s in 1:N)
        psi[out+1] += c
    end
    psi
end

"Defining KSVC parent Hamiltonian applied in Fock space (no BdG convention ambiguity)."
function ksvc_hamiltonian_action(psi,nx::Int,ny::Int;periodic=false,twists=(1,1),
                                  pairing_scale=1,hopping_scale=1)
    nx*ny == trailing_zeros(length(psi)) || throw(DimensionMismatch("patch/state mismatch"))
    state = Dict(UInt(i-1)=>ComplexF64(c) for (i,c) in enumerate(psi) if !iszero(c))
    out = zeros(ComplexF64,length(psi))
    site(x,y) = mod1(x,nx)+nx*(mod1(y,ny)-1)-1
    for y in 1:ny,x in 1:nx
        i = site(x,y)
        for (dx,dy,coeff,pairing) in ((0,1,2im,true),(1,0,-2im,true),
                                    (1,1,-1,false),(1,-1,-1,false))
            xx,yy=x+dx,y+dy
            (!periodic && !(1<=xx<=nx && 1<=yy<=ny)) && continue
            phase = (1<=xx<=nx ? 1 : twists[1])*(1<=yy<=ny ? 1 : twists[2])
            phase *= pairing ? pairing_scale : hopping_scale
            j = site(xx,yy)
            word = [(i,true),(j,pairing)]
            adjword = [(j,!pairing),(i,false)]
            for (w,c) in ((word,phase*coeff),(adjword,phase*conj(coeff)))
                for (mask,a) in _apply_word(state,w)
                    out[Int(mask)+1] += c*a
                end
            end
        end
    end
    out
end

"Change Fock mode order, including the occupation-dependent inversion sign."
function reorder_fock(psi::AbstractVector, order::AbstractVector{Int})
    N = length(order)
    sort(order) == collect(1:N) || throw(ArgumentError("order must be a permutation"))
    length(psi) == 1<<N || throw(DimensionMismatch("Fock vector length"))
    destination = invperm(order)
    out = similar(psi)
    for mask in 0:length(psi)-1
        bits = [(mask>>(i-1))&1 for i in 1:N]
        nswap = sum(bits[i]*bits[j] for i in 1:N for j in i+1:N if destination[i]>destination[j]; init=0)
        target = sum(bits[old]<<(new-1) for (new,old) in enumerate(order))
        out[target+1] = (-1)^nswap * psi[mask+1]
    end
    out
end

"Covariance Γ_ab=(i/2)⟨[γ_a,γ_b]⟩; γ_(2j)=i(c_j-c_j†)."
function majorana_covariance(psi::AbstractVector)
    N = trailing_zeros(length(psi))
    1<<N == length(psi) || throw(DimensionMismatch("Fock vector length is not a power of two"))
    real(dot(psi,psi)) > 0 || throw(ArgumentError("zero Fock state"))
    state = Dict(UInt(i-1)=>ComplexF64(c) for (i,c) in enumerate(psi) if !iszero(c))
    images = zeros(ComplexF64,length(psi),2N)
    for i in 0:N-1
        for (creation,c1,c2) in ((false,1,1im),(true,1,-1im))
            for (mask,c) in _apply_word(state,[(i,creation)])
                images[Int(mask)+1,2i+1] += c1*c
                images[Int(mask)+1,2i+2] += c2*c
            end
        end
    end
    gram = images' * images / real(dot(psi,psi))
    real.(1im*(gram-transpose(gram))/2)
end

"Exact five-sector occupation contraction, grouping region Fock modes first."
function occupation_sectors(psi::AbstractVector, regions::AbstractVector{Symbol}; maxsites=6)
    N = length(regions)
    N <= maxsites || throw(ArgumentError("occupation oracle exceeds maxsites=$maxsites"))
    all(x -> x in (:A,:B,:C),regions) || throw(ArgumentError("regions must be A/B/C"))
    order = sortperm(regions; by=x->findfirst(==(x),(:A,:B,:C)))
    v = reorder_fock(psi,order)
    regs = regions[order]
    Z1 = real(dot(v,v))
    Z1 > 0 || throw(ArgumentError("zero Fock state"))
    counts = [count(==(X),regs) for X in (:A,:B,:C)]
    wave = reshape(v,Tuple(1 .<< counts))
    purity = map(1:3) do r
        matrix = reshape(permutedims(wave,(r,filter(!=(r),1:3)...)),1<<counts[r],:)
        rho = matrix*matrix'
        real(tr(rho*rho))
    end
    # Plain occupation contraction, deliberately NOT a graded physical swap.
    Z4 = 0.0im
    # Reference uses the direct ket quartet for N<=6; skip zero amplitudes.
    nonzero = findall(!iszero,v)
    masks = [sum(1<<(i-1) for i in 1:N if regs[i]==X; init=0) for X in (:A,:B,:C)]
    sigma = ((1,2,3,4),(2,1,4,3),(3,4,1,2))
    for indices in Iterators.product(ntuple(_->nonzero,4)...)
        ketm = indices .- 1
        value = prod(v[i] for i in indices)
        for r in 1:4
            bra = sum(ketm[sigma[x][r]] & masks[x] for x in 1:3)
            value *= conj(v[bra+1])
        end
        Z4 += value
    end
    ratio = Z4*Z1^2/prod(purity)
    abs(imag(ratio)) <= 1e-11*max(abs(ratio),eps()) || error("nonreal cube ratio")
    real(ratio)>0 || error("nonpositive cube ratio")
    (; Z1, Z2A=purity[1], Z2B=purity[2], Z2C=purity[3], Z4,
       S2=-log.(purity./Z1^2), S3=-log(real(Z4)/Z1^4)/2,
       stilde=-log(real(ratio)), order)
end
