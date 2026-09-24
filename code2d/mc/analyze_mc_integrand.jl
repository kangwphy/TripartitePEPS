# analyze_mc_integrand.jl -- pool all MC per-node integrand data, fit a smooth
# curve, and separate quadrature error from statistical error.
# =============================================================================
# Reads every logs2d/mc_*.out, extracts (L, beta, NGL, seed, lambda, m, err) for
# all completed lambda-nodes.  For each (L,beta) group:
#   1. pool ALL points from all seeds and GL orders into one dataset
#   2. weighted least-squares polynomial fit in lambda, degree scan 6..14,
#      pick by chi^2/dof ~ 1 (stop when adding degree stops helping)
#   3. tS_fit = -integral_0^1 fit  (exact integral of the polynomial)
#   4. quadrature check ON THE FIT: |sum_i w_i f(x_i) - int f| for GL12 & GL24
#      -> the true quadrature error, free of MC noise
#   5. statistical error: parametric bootstrap (resample each point from
#      N(m,err), refit, reintegrate) x 400
# Output: one line per (L,beta): tS_fit +- stat, quadGL12, quadGL24, npts, deg.
# This arbitrates: if quadGL12 << observed GL12-vs-GL24 spread, the spread is
# statistics, not quadrature.
# =============================================================================
using Printf, Statistics, Random, LinearAlgebra

logdir = joinpath(@__DIR__, "..", "..", "logs2d")
pat_hdr  = r"mc_tildeS: L=(\d+) beta=([\d.]+) sweeps=(\d+) seed=(\d+)"
pat_node = r"lam=([\d.]+)\s+<dE/dl>=(-?[\d.]+) \+- ([\d.]+)"

# group key -> vector of (lam, m, e, seed, ngl)
groups = Dict{Tuple{Int,String},Vector{NTuple{5,Float64}}}()
BETA_C = 0.5*log(1+sqrt(2))

for f in sort(readdir(logdir))
    startswith(f, "mc_") || continue
    L = 0; bstr = ""; seed = 0; nodes = NTuple{3,Float64}[]
    for line in eachline(joinpath(logdir, f))
        m = match(pat_hdr, line)
        if m !== nothing
            L = parse(Int, m.captures[1])
            b = parse(Float64, m.captures[2])
            bstr = abs(b - BETA_C) < 1e-6 ? "crit" : @sprintf("%.2f", b)
            seed = parse(Int, m.captures[4])
        end
        m = match(pat_node, line)
        if m !== nothing
            push!(nodes, (parse(Float64,m.captures[1]), parse(Float64,m.captures[2]),
                          parse(Float64,m.captures[3])))
        end
    end
    L == 0 && continue
    ngl = length(nodes) # completed nodes; only keep full sets (12/24/48)
    ngl in (12,24,48) || continue
    v = get!(groups, (L,bstr), NTuple{5,Float64}[])
    for (lam,m,e) in nodes
        push!(v, (lam, m, e, Float64(seed), Float64(ngl)))
    end
end

function gauss_legendre01(n)
    x = zeros(n); w = zeros(n)
    for i in 1:n
        t = cos(pi*(i-0.25)/(n+0.5)); local dp = 0.0
        for _ in 1:100
            p0, p1 = 1.0, t
            for j in 2:n; p0, p1 = p1, ((2j-1)*t*p1-(j-1)*p0)/j; end
            dp = n*(t*p1-p0)/(t^2-1); dt = p1/dp; t -= dt
            abs(dt) < 1e-15 && break
        end
        x[i] = 0.5*(1-t); w[i] = 1.0/((1-t^2)*dp^2)
    end
    (x,w)
end

"weighted poly fit; returns coeffs (c0..cd)."
function polyfit_w(lam, m, e, deg)
    A = [lam[i]^j for i in eachindex(lam), j in 0:deg]
    W = Diagonal(1 ./ e.^2)
    (A'*W*A) \ (A'*W*m)
end
polyint01(c) = sum(c[j+1]/(j+1) for j in 0:length(c)-1)   # int_0^1
polyeval(c, x) = sum(c[j+1]*x^j for j in 0:length(c)-1)

rng = Xoshiro(7)
println("group            npts deg  tS_fit      stat      quadGL12   quadGL24   chi2/dof")
for key in sort(collect(keys(groups)))
    v = groups[key]
    lam = [p[1] for p in v]; m = [p[2] for p in v]; e = [p[3] for p in v]
    npts = length(v)
    # degree scan
    best = nothing
    for deg in 6:14
        npts <= deg+2 && break
        c = polyfit_w(lam, m, e, deg)
        chi2 = sum(((polyeval(c,lam[i])-m[i])/e[i])^2 for i in 1:npts)/(npts-deg-1)
        if best === nothing || chi2 < best[2] - 0.05
            best = (c, chi2, deg)
        end
    end
    c, chi2, deg = best
    tS = -polyint01(c)
    # quadrature error of GL12/GL24 evaluated ON the fitted smooth curve
    q = Dict{Int,Float64}()
    for n in (12,24)
        gx, gw = gauss_legendre01(n)
        q[n] = abs(sum(gw[i]*polyeval(c,gx[i]) for i in 1:n) - polyint01(c))
    end
    # parametric bootstrap for the statistical error of tS_fit
    boots = Float64[]
    for _ in 1:400
        mb = m .+ e .* randn(rng, npts)
        cb = polyfit_w(lam, mb, e, deg)
        push!(boots, -polyint01(cb))
    end
    @printf("L=%-3d %-5s  %4d  %2d  %+.6f  %.6f  %.2e  %.2e  %.2f\n",
            key[1], key[2], npts, deg, tS, std(boots), q[12], q[24], chi2)
end
println("DONE_ANALYZE")
