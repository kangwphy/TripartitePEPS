# fit_profile.jl -- fit the MCPROF per-bond integrand profiles.
# Model per arm (seam of M=L/2 bonds, r=1 at the junction, r=M at the open edge):
#   I(r) = -cJ * r^(-alpha)  -  cB * (M+1-r)^(-alphaB)
# (junction tail + boundary-endpoint tail). Pools seeds per (L, arm).
# Fits: (a) per-L joint over all three arms with shared alpha;
#       (b) junction-window fit (r <= M/2, boundary tail subtracted iteratively);
#       (c) L-collapse check: cJ, alpha across L=32/48/64.
# Grid search over (alpha, alphaB) + linear least squares for (cJ, cB).
using Printf, Statistics

logdir = joinpath(@__DIR__, "..", "..", "logs2d")
pat_hdr  = r"mc_tildeS: L=(\d+) beta=0\.44068.*seed=(\d+)"
WANT_GAUGE = get(ENV, "FITGAUGE", "user")   # "user" -> only gauge=user logs; "sym" -> only legacy logs
pat_prof = r"^PROF (\w) +(\d+) ([+-]?\d+\.\d+) (\d+\.\d+)"

# (L, arm, r) -> list of (I, err)
data = Dict{Tuple{Int,Char,Int},Vector{Tuple{Float64,Float64}}}()
for f in sort(readdir(logdir))
    startswith(f, "mc_") || continue
    L = 0; gaugeok = false
    for line in eachline(joinpath(logdir, f))
        m = match(pat_hdr, line)
        if m !== nothing
            L = parse(Int, m.captures[1])
            hasg = occursin("gauge=user", line)
            gaugeok = (WANT_GAUGE == "user") ? hasg : !hasg
        end
        gaugeok || continue
        m = match(pat_prof, line)
        if m !== nothing && L > 0
            key = (L, m.captures[1][1], parse(Int, m.captures[2]))
            push!(get!(data, key, Tuple{Float64,Float64}[]),
                  (parse(Float64, m.captures[3]), parse(Float64, m.captures[4])))
        end
    end
end
isempty(data) && (println("no PROF data"); exit())

Ls = sort(unique(k[1] for k in keys(data)))
println("profiles found for L = ", Ls)

"pooled I(r) with error for one (L,arm)."
function pooled(L, arm)
    M = L ÷ 2
    rs = Int[]; Is = Float64[]; Es = Float64[]
    for r in 1:M
        v = get(data, (L, arm, r), nothing); v === nothing && continue
        w = [1/e^2 for (_, e) in v]
        push!(rs, r)
        push!(Is, sum(v[i][1]*w[i] for i in eachindex(v)) / sum(w))
        push!(Es, 1/sqrt(sum(w)))
    end
    (rs, Is, Es)
end

"chi2 of the two-tail model at (alpha, alphaB) with LSQ (cJ,cB); returns (chi2, cJ, cB)."
function fit_two_tail(rs, Is, Es, M, alpha, alphaB)
    # design: I = -cJ * r^-a - cB * (M+1-r)^-aB
    A11=A12=A22=b1=b2=0.0
    for (r, I, e) in zip(rs, Is, Es)
        f1 = -Float64(r)^(-alpha); f2 = -Float64(M+1-r)^(-alphaB); w = 1/e^2
        A11 += w*f1*f1; A12 += w*f1*f2; A22 += w*f2*f2
        b1  += w*f1*I;  b2  += w*f2*I
    end
    det = A11*A22 - A12^2
    cJ = ( A22*b1 - A12*b2) / det
    cB = (-A12*b1 + A11*b2) / det
    chi2 = 0.0
    for (r, I, e) in zip(rs, Is, Es)
        pred = -cJ*Float64(r)^(-alpha) - cB*Float64(M+1-r)^(-alphaB)
        chi2 += ((I-pred)/e)^2
    end
    (chi2, cJ, cB)
end

println("\n== junction-window log-log fits: I(r) = -c * r^-alpha on r in [2, M/3] ==")
println("L arm  alpha   c       chi2/dof  npts")
for L in Ls, arm in ('W','E','V')
    rs, Is, Es = pooled(L, arm)
    M = L ÷ 2
    win = [i for i in eachindex(rs) if 2 <= rs[i] <= max(4, M ÷ 3)]
    length(win) < 3 && continue
    # only keep points with definite sign (junction side W/E negative, V positive)
    sgn = sign(sum(Is[i] for i in win))
    ok = [i for i in win if sign(Is[i]) == sgn && abs(Is[i]) > Es[i]]
    length(ok) < 3 && continue
    x = [log(Float64(rs[i])) for i in ok]
    y = [log(abs(Is[i])) for i in ok]
    w = [(abs(Is[i])/Es[i])^2 for i in ok]          # log-space weights
    Sw = sum(w); Sx = sum(w.*x); Sy = sum(w.*y); Sxx = sum(w.*x.*x); Sxy = sum(w.*x.*y)
    slope = (Sw*Sxy - Sx*Sy)/(Sw*Sxx - Sx^2)
    inter = (Sy - slope*Sx)/Sw
    chi2 = sum(w[j]*(y[j] - (inter + slope*x[j]))^2 for j in eachindex(ok))
    @printf("%-3d %c  %+.3f  %+.4f   %.2f   %d\n", L, arm, -slope, sgn*exp(inter), chi2/max(1,length(ok)-2), length(ok))
end

println("\n== raw pooled profiles (for the note figure) ==")
for L in Ls, arm in ('W','E','V')
    rs, Is, Es = pooled(L, arm)
    isempty(rs) && continue
    print("PROFILE_POOLED L=", L, " arm=", arm)
    for i in eachindex(rs); @printf(" %d:%.5f", rs[i], Is[i]); end
    println()
end

println("\n== arm sums (empirical descent budget; sum over all r) ==")
println("L  sum_W      sum_E      sum_V      total(=tS)")
for L in Ls
    tot = 0.0; sums = Dict{Char,Float64}()
    for arm in ('W','E','V')
        rs, Is, _ = pooled(L, arm)
        sums[arm] = sum(Is); tot += sums[arm]
    end
    @printf("%-3d %+.4f  %+.4f  %+.4f  %+.4f\n", L, sums['W'], sums['E'], sums['V'], tot)
end
println("DONE_FITPROF")
