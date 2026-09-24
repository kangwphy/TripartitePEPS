# validate_rect_mc.jl -- Swendsen--Wang/TI validation on a rectangular lattice.
#
# This is a small-lattice regression driver for the production algorithm.  It
# supports the 3x3 and 3x4 geometries used by validate_rect_exact.jl, uses the
# same six-layer N/D endpoints and GL12 integration, and never appends to the
# production data2d/mc_results.csv file.

using Random, Printf, Statistics

const Lx = parse(Int, get(ENV,"VRLX","3"))
const Ly = parse(Int, get(ENV,"VRLY","3"))
const NSW = parse(Int, get(ENV,"VRSW","20000"))
const NEQ = parse(Int, get(ENV,"VREQ","1000"))
const SEED = parse(Int, get(ENV,"VRSEED","1"))
const BETA = 0.5*log(1+sqrt(2.0))
const NL = 6
const rng = Xoshiro(SEED)

region(x,y) = y >= Ly÷2 ? 'C' : (x < Lx÷2 ? 'A' : 'B')
site(x,y) = y*Lx+x+1
nid(l,s) = (s-1)*NL+l

function block_J(pi,pj,off)
    T=Dict{Tuple{Int,Int},Float64}()
    for r in eachindex(pi)
        T[(off+r,off+r)]=get(T,(off+r,off+r),0.0)+BETA/2
        T[(off+pi[r],off+pj[r])]=get(T,(off+pi[r],off+pj[r]),0.0)+BETA/2
    end
    [(a,b,J) for ((a,b),J) in T]
end

function couplings(r1,r2)
    if r1==r2
        t=[(l,l,BETA) for l in 1:NL]; return t,t
    end
    id4=[1,2,3,4]; b4=[2,1,4,3]; c4=[3,4,1,2]
    p4=Dict('A'=>id4,'B'=>b4,'C'=>c4)
    tN=vcat(block_J(p4[r1],p4[r2],0),[(5,5,BETA),(6,6,BETA)])
    sw=[2,1]; id=[1,2]
    pA=Dict('A'=>sw,'B'=>id,'C'=>id)
    pB=Dict('A'=>id,'B'=>sw,'C'=>id)
    pC=Dict('A'=>id,'B'=>id,'C'=>sw)
    tD=vcat(block_J(pA[r1],pA[r2],0),block_J(pB[r1],pB[r2],2),block_J(pC[r1],pC[r2],4))
    tN,tD
end

struct Bond
    i::Int; j::Int
    tN::Vector{Tuple{Int,Int,Float64}}
    tD::Vector{Tuple{Int,Int,Float64}}
    equal::Bool
end

const bonds = let list=Bond[]
    for y in 0:Ly-1,x in 0:Lx-1
        if x<Lx-1
            tN,tD=couplings(region(x,y),region(x+1,y))
            push!(list,Bond(site(x,y),site(x+1,y),tN,tD,sort(tN)==sort(tD)))
        end
        if y<Ly-1
            tN,tD=couplings(region(x,y),region(x,y+1))
            push!(list,Bond(site(x,y),site(x,y+1),tN,tD,sort(tN)==sort(tD)))
        end
    end
    list
end

const NN=NL*Lx*Ly
const spins=ones(Int8,NN)
const parent=collect(1:NN)
function root(i)
    while parent[i]!=i; parent[i]=parent[parent[i]]; i=parent[i]; end
    i
end
unite(i,j)=(parent[root(i)]=root(j))

function sweep!(lam)
    for i in eachindex(parent); parent[i]=i; end
    for b in bonds
        if b.equal
            for (a,c,J) in b.tN
                u=nid(a,b.i);v=nid(c,b.j)
                spins[u]==spins[v] && rand(rng)<1-exp(-2J) && unite(u,v)
            end
        else
            for (a,c,J) in b.tD
                u=nid(a,b.i);v=nid(c,b.j);Jl=(1-lam)*J
                spins[u]==spins[v] && rand(rng)<1-exp(-2Jl) && unite(u,v)
            end
            for (a,c,J) in b.tN
                u=nid(a,b.i);v=nid(c,b.j);Jl=lam*J
                spins[u]==spins[v] && rand(rng)<1-exp(-2Jl) && unite(u,v)
            end
        end
    end
    flip=Dict{Int,Int8}()
    for i in eachindex(spins)
        r=root(i); f=get!(()->rand(rng)<0.5 ? Int8(-1) : Int8(1),flip,r); spins[i]*=f
    end
end

function deltaK()
    e=0.0
    for b in bonds
        b.equal && continue
        for (a,c,J) in b.tN; e+=J*spins[nid(a,b.i)]*spins[nid(c,b.j)]; end
        for (a,c,J) in b.tD; e-=J*spins[nid(a,b.i)]*spins[nid(c,b.j)]; end
    end
    e
end

function gl01(n)
    x=zeros(n);w=zeros(n)
    for i in 1:n
        t=cos(pi*(i-.25)/(n+.5));dp=0.0
        for _ in 1:100
            p0,p1=1.0,t
            for j in 2:n;p0,p1=p1,((2j-1)*t*p1-(j-1)*p0)/j;end
            dp=n*(t*p1-p0)/(t^2-1);dt=p1/dp;t-=dt;abs(dt)<1e-15&&break
        end
        x[i]=.5*(1-t);w[i]=1/((1-t^2)*dp^2)
    end
    x,w
end

function run_validation()
    @assert NSW%20==0
    x,w=gl01(12); integ=0.0; variance=0.0
    for q in eachindex(x)
        for _ in 1:NEQ; sweep!(x[q]); end
        bins=zeros(20); bs=NSW÷20
        for k in 1:20, _ in 1:bs; sweep!(x[q]); bins[k]+=deltaK(); end
        bins./=bs; m=mean(bins); e=std(bins)/sqrt(20)
        integ+=w[q]*m; variance+=(w[q]*e)^2
    end
    -integ, sqrt(variance)
end

value,error=run_validation()
@printf("RECT_MC L=%dx%d beta=crit sweeps=%d seed=%d tildeS=%.10f err=%.10f\n",
        Lx,Ly,NSW,SEED,value,error)
