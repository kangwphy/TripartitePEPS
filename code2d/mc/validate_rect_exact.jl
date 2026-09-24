# validate_rect_exact.jl -- exact 3x3 and 3x4 validation of the six-layer TI.
#
# Two independent routes are compared at beta_c:
#   1. enumerate all 2^(Lx*Ly) physical RK configurations, build psi exactly,
#      and evaluate the five replica contractions with the dense cube kernel;
#   2. construct the N and D six-layer classical models directly and sum every
#      row-boundary state with an untruncated transfer calculation.  This gives
#      log Z_lambda and <Delta K>_lambda exactly, including at all GL12 nodes.
#
# The second route avoids the impossible flat enumeration of 2^(6*Lx*Ly)
# configurations, but it is still an exact sum: its boundary vector contains
# all 2^(6*Lx) row states and no singular-value or bond-dimension truncation.

using LinearAlgebra, Printf

include(joinpath(@__DIR__, "support", "cube.jl"))

const BETA_C_RECT = 0.5 * log(1 + sqrt(2.0))
const NL_RECT = 6
const SA_RECT = [1, 2, 3, 4]
const SB_RECT = [2, 1, 4, 3]       # (12)(34)
const SC_RECT = [3, 4, 1, 2]       # (13)(24)
const SWAP2_RECT = [2, 1]
const ID2_RECT = [1, 2]

"Region convention used for odd and rectangular validation lattices."
region_rect(x, y, Lx, Ly) = y >= Ly ÷ 2 ? 'C' : (x < Lx ÷ 2 ? 'A' : 'B')

"Open-boundary nearest-neighbor bonds, with zero-based site indices."
function bonds_rect(Lx, Ly)
    site(x, y) = y * Lx + x
    out = Tuple{Int,Int}[]
    for y in 0:Ly-1, x in 0:Lx-1
        x < Lx-1 && push!(out, (site(x,y), site(x+1,y)))
        y < Ly-1 && push!(out, (site(x,y), site(x,y+1)))
    end
    out
end

"Directly enumerate the physical RK wavefunction and evaluate its replica sums."
function direct_rk_value(Lx, Ly, beta; order=('A','B','C'))
    N = Lx * Ly
    regs = [region_rect(x, y, Lx, Ly) for y in 0:Ly-1 for x in 0:Lx-1]
    pos = Dict(r => findall(==(r), regs) .- 1 for r in ('A','B','C'))
    # The cube contraction is symmetric under relabeling A, B, and C.  Put a
    # large region in the middle slot when desired to keep the dense kernel
    # small (its cost is controlled by the two outer dimensions).
    dA, dB, dC = (1 << length(pos[r]) for r in order)
    psi = zeros(Float64, 1 << N)
    B = bonds_rect(Lx, Ly)
    spin(s, i) = isodd(s >> i) ? 1.0 : -1.0
    for s in 0:(1<<N)-1
        E = sum(spin(s,i) * spin(s,j) for (i,j) in B)
        abc = Int[]
        for r in order
            z = 0
            for (k,i) in enumerate(pos[r]); z |= ((s >> i) & 1) << (k-1); end
            push!(abc, z)
        end
        a,b,c = abc
        psi[(a+1) + dA*b + dA*dB*c] = exp(beta * E / 2)
    end
    res = cube_dense(psi, dA, dB, dC)
    (tildeS=res.tildeS, sizes=(length(pos['A']),length(pos['B']),length(pos['C'])))
end

"Coupling triples for a replica block; coincident ket/bra edges are combined."
function block_rect(pi, pj, off, beta)
    T = Dict{Tuple{Int,Int},Float64}()
    for r in eachindex(pi)
        T[(off+r,off+r)] = get(T,(off+r,off+r),0.0) + beta/2
        T[(off+pi[r],off+pj[r])] = get(T,(off+pi[r],off+pj[r]),0.0) + beta/2
    end
    [(a,b,J) for ((a,b),J) in T]
end

"Return exact numerator and denominator couplings for one oriented spatial bond."
function endpoint_couplings(r1, r2, beta)
    if r1 == r2
        t = [(l,l,beta) for l in 1:NL_RECT]
        return t, t
    end
    p4 = Dict('A'=>SA_RECT, 'B'=>SB_RECT, 'C'=>SC_RECT)
    tN = vcat(block_rect(p4[r1],p4[r2],0,beta), [(5,5,beta),(6,6,beta)])
    pA = Dict('A'=>SWAP2_RECT, 'B'=>ID2_RECT, 'C'=>ID2_RECT)
    pB = Dict('A'=>ID2_RECT, 'B'=>SWAP2_RECT, 'C'=>ID2_RECT)
    pC = Dict('A'=>ID2_RECT, 'B'=>ID2_RECT, 'C'=>SWAP2_RECT)
    tD = vcat(block_rect(pA[r1],pA[r2],0,beta),
              block_rect(pB[r1],pB[r2],2,beta),
              block_rect(pC[r1],pC[r2],4,beta))
    tN, tD
end

"Energy and Delta-K tables for a bond whose endpoint layer states are u and v."
function bond_tables(r1, r2, beta)
    q = 1 << NL_RECT
    tN, tD = endpoint_couplings(r1,r2,beta)
    EN = zeros(q,q); ED = zeros(q,q)
    spin(s,l) = isodd(s >> (l-1)) ? 1.0 : -1.0
    for u in 0:q-1, v in 0:q-1
        EN[u+1,v+1] = sum(J*spin(u,a)*spin(v,b) for (a,b,J) in tN)
        ED[u+1,v+1] = sum(J*spin(u,a)*spin(v,b) for (a,b,J) in tD)
    end
    ED, EN-ED
end

"Apply W' to one tensor axis; optionally propagate its lambda derivative."
function apply_axis(v, dv, W, dW, axis)
    n = ndims(v)
    perm = vcat(axis, collect(1:axis-1), collect(axis+1:n))
    invp = invperm(perm)
    vp = permutedims(v, perm); dp = permutedims(dv, perm)
    oldshape = size(vp)
    V = reshape(vp, oldshape[1], :); D = reshape(dp, oldshape[1], :)
    Vn = transpose(W) * V
    Dn = transpose(W) * D + transpose(dW) * V
    permutedims(reshape(Vn, oldshape), invp), permutedims(reshape(Dn, oldshape), invp)
end

"Exact log Z and derivative with scale bookkeeping (same transfer contraction)."
function exact_logz_dlogz(Lx, Ly, beta, lam)
    q = 1 << NL_RECT; shape = ntuple(_->q,Lx)
    h = Vector{Array{Float64}}(undef,Ly); dh = similar(h)
    for y in 0:Ly-1
        E=zeros(shape); DE=zeros(shape)
        tabs=[bond_tables(region_rect(x,y,Lx,Ly),region_rect(x+1,y,Lx,Ly),beta) for x in 0:Lx-2]
        for I in CartesianIndices(shape)
            ed=0.0; dk=0.0
            for x in 1:Lx-1; ed+=tabs[x][1][I[x],I[x+1]]; dk+=tabs[x][2][I[x],I[x+1]]; end
            E[I]=ed+lam*dk; DE[I]=dk
        end
        h[y+1]=exp.(E); dh[y+1]=h[y+1].*DE
    end
    v=copy(h[1]); dv=copy(dh[1]); logscale=0.0
    for y in 0:Ly-2
        for x in 0:Lx-1
            ED,DK=bond_tables(region_rect(x,y,Lx,Ly),region_rect(x,y+1,Lx,Ly),beta)
            W=exp.(ED.+lam.*DK); v,dv=apply_axis(v,dv,W,W.*DK,x+1)
        end
        dv=h[y+2].*dv+dh[y+2].*v; v=h[y+2].*v
        scale=maximum(v); v./=scale; dv./=scale; logscale+=log(scale)
    end
    Z=sum(v); log(Z)+logscale, sum(dv)/Z
end

"Gauss--Legendre nodes and weights on [0,1]."
function gl01_rect(n)
    x=zeros(n); w=zeros(n)
    for i in 1:n
        t=cos(pi*(i-0.25)/(n+0.5)); dp=0.0
        for _ in 1:100
            p0,p1=1.0,t
            for j in 2:n; p0,p1=p1,((2j-1)*t*p1-(j-1)*p0)/j; end
            dp=n*(t*p1-p0)/(t^2-1); dt=p1/dp; t-=dt
            abs(dt)<1e-15 && break
        end
        x[i]=0.5*(1-t); w[i]=1/((1-t^2)*dp^2)
    end
    x,w
end

function validate_size(Lx,Ly,beta)
    direct=direct_rk_value(Lx,Ly,beta)
    lD,_=exact_logz_dlogz(Lx,Ly,beta,0.0)
    lN,_=exact_logz_dlogz(Lx,Ly,beta,1.0)
    x,w=gl01_rect(12)
    ti=-sum(w[q]*exact_logz_dlogz(Lx,Ly,beta,x[q])[2] for q in eachindex(x))
    endpoint=lD-lN
    @printf("%dx%d sizes=%s  direct=%.15f endpoint=%.15f GL12=%.15f  diffs=(%.2e, %.2e)\n",
            Lx,Ly,string(direct.sizes),direct.tildeS,endpoint,ti,
            endpoint-direct.tildeS,ti-direct.tildeS)
    (direct=direct.tildeS,endpoint=endpoint,ti=ti)
end

function main_rect_exact(args)
    println("EXACT RECTANGULAR VALIDATION beta_c=$(BETA_C_RECT)")
    if isempty(args)
        validate_size(3,3,BETA_C_RECT)
        validate_size(3,4,BETA_C_RECT)
    elseif args[1] == "direct"
        Lx=parse(Int,args[2]); Ly=parse(Int,args[3])
        # C is the largest region for the validation geometry, so use it as
        # the middle argument of the symmetric dense cube contraction.
        result=direct_rk_value(Lx,Ly,BETA_C_RECT; order=('A','C','B'))
        @printf("%dx%d sizes=%s direct=%.15f\n",Lx,Ly,string(result.sizes),result.tildeS)
    elseif args[1] == "endpoint"
        Lx=parse(Int,args[2]); Ly=parse(Int,args[3])
        lD,_=exact_logz_dlogz(Lx,Ly,BETA_C_RECT,0.0)
        lN,_=exact_logz_dlogz(Lx,Ly,BETA_C_RECT,1.0)
        @printf("%dx%d endpoint=%.15f logZD=%.15f logZN=%.15f\n",Lx,Ly,lD-lN,lD,lN)
    else
        error("usage: validate_rect_exact.jl [direct|endpoint Lx Ly]")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_rect_exact(ARGS)
