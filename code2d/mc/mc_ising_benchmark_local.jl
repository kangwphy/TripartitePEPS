# Single-layer 2D Ising local-Metropolis benchmark.
using Random, Printf, Statistics

const L = parse(Int, ENV["MCL"])
const bstr = ENV["MCB"]
const beta = bstr == "crit" ? 0.5*log(1+sqrt(2)) : parse(Float64, bstr)
const NEQ = parse(Int, get(ENV, "MCEQ", "100000"))
const NSW = parse(Int, get(ENV, "MCSW", "100000"))
const seed = parse(Int, get(ENV, "MCSEED", "1"))
const outdir = ENV["MCOUT"]
const rng = Xoshiro(seed)
const N = L*L
const spins = ones(Int8, N)
const total_magnetization = Ref(0)
const total_energy = Ref(0)
site(x,y) = y*L + x + 1

"One checkerboard Metropolis sweep of the periodic square-lattice Ising model."
function sweep!()
    for parity in 0:1, y in 0:L-1, x in 0:L-1
        ((x+y)&1)==parity || continue
        i = site(x,y)
        h = (x < L-1 ? spins[site(x+1,y)] : 0) +
            (x > 0 ? spins[site(x-1,y)] : 0) +
            (y < L-1 ? spins[site(x,y+1)] : 0) +
            (y > 0 ? spins[site(x,y-1)] : 0)
        old_spin = spins[i]
        dK = -2.0*beta*old_spin*h
        if dK >= 0.0 || rand(rng) < exp(dK)
            total_magnetization[] -= 2*old_spin
            total_energy[] += 2*old_spin*h
            spins[i] = -old_spin
        end
    end
end

"Return |m|, m^2, energy/site, and signed m for the current configuration."
function observables()
    m = total_magnetization[]/N
    (abs(m), m*m, total_energy[]/N, m)
end

for i in eachindex(spins); spins[i] = rand(rng) < 0.5 ? Int8(-1) : Int8(1); end
total_magnetization[] = sum(spins)
for y in 0:L-1, x in 0:L-1
    x < L-1 && (total_energy[] -= spins[site(x,y)] * spins[site(x+1,y)])
    y < L-1 && (total_energy[] -= spins[site(x,y)] * spins[site(x,y+1)])
end
for _ in 1:NEQ; sweep!(); end
NBIN=20; @assert NSW % NBIN == 0
binsz=NSW ÷ NBIN; vals=zeros(4,NBIN)
for b in 1:NBIN
    acc=zeros(4)
    for _ in 1:binsz; sweep!(); acc .+= collect(observables()); end
    vals[:,b] .= acc/binsz
end
means=vec(mean(vals,dims=2)); errs=vec(std(vals,dims=2))/sqrt(NBIN)
chi=beta*N*(means[2]-means[4]^2)
chierr=beta*N*sqrt(errs[2]^2+(2*abs(means[4])*errs[4])^2)
mkpath(outdir); betakey=replace(bstr,"."=>"p")
file=@sprintf("L%03d_b%s_n%d_eq%d_seed%d.csv",L,betakey,NSW,NEQ,seed)
open(joinpath(outdir,file),"w") do io
    println(io,"L,beta,n_sweeps,n_eq,seed,m_abs,m_abs_error,m2,m2_error,energy_per_site,energy_error,m_mean,m_mean_error,chi,chi_error,update,quantity,tag")
    @printf(io,"%d,%s,%d,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,local,benchmark,LOCAL_ISING_OPEN_V2\n",L,bstr,NSW,NEQ,seed,means[1],errs[1],means[2],errs[2],means[3],errs[3],means[4],errs[4],chi,chierr)
end
@printf("DONE benchmark L=%d beta=%s seed=%d m_abs=%.6f m2=%.6f e=%.6f chi=%.6f\n",L,bstr,seed,means[1],means[2],means[3],chi)
