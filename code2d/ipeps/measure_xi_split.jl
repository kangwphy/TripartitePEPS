# measure_xi_split.jl -- xi ONLY, for the split-replica one-replica environment.
# =============================================================================
# The tildeS_J measurement is NOT repeated here.  The one-replica environment
# (C1,T1,R1) evolves independently of the seam edge Td2 in
# coevolved_pair_environment (Td2 is a passenger: it never feeds back), so the
# same C1/T1 trajectory is reproduced by running the one-replica loop alone --
# at a fraction of the cost, because the expensive part of the production run
# is the gauge-polluted seam-edge stopping rule, not the corner fixed point.
#
# Reported per (beta,chi1):
#   xi_conn   degeneracy-aware connected length (skips the ordered-phase
#             symmetric/antisymmetric partner of the leading eigenvalue)
#   xi_naive  the old estimator -1/log(lambda2/lambda1)  (1e15 / Inf when ordered)
#   z2split   1 - lambda2/lambda1: the order-parameter diagnostic itself
#   xi_exact  analytic Ising spin correlation length, for validation
# Gate: in the disordered phase xi_conn must reproduce the xi1 already stored in
# data/rk_ising/ctmrg/tables/beta_chi_scan_split/summary.csv.
#
# Env: XBETAS (colon list, default 0.30:0.40:0.46:0.48:0.60)
#      XCHIS  (colon list, default 4:8:16:24)
#      XNMAX / XTOL  corner-spectrum stopping rule (gauge invariant)
# =============================================================================
include(joinpath(@__DIR__, "yvx_core.jl"))

const XBETAS = parse.(Float64, split(get(ENV, "XBETAS", "0.30:0.40:0.46:0.48:0.60"), ":"))
const XCHIS  = parse.(Int,     split(get(ENV, "XCHIS",  "4:8:16:24"), ":"))
const XNMAX  = parse(Int,      get(ENV, "XNMAX", "20000"))
const XTOL   = parse(Float64,  get(ENV, "XTOL",  "1e-14"))

# ---- verbatim copies from split_replica_ctmrg.jl (one-replica sector only) ----
function ctm_step_Z_flip(C, T, a, chimax, R)
    chi = size(C,1); dd = size(a,1); cd = chi*dd
    dd == 2 || error("ctm_step_Z_flip requires the one-replica d=2 tensor")
    CTl = reshape(C*reshape(T, chi, chi*dd), chi, chi, dd)
    res = reshape(reshape(permutedims(T,(2,1,3)), chi, chi*dd)' * reshape(CTl, chi, chi*dd),
                  chi, dd, chi, dd)
    Cb  = reshape(reshape(permutedims(res,(1,3,2,4)), chi*chi, dd*dd) * reshape(a, dd*dd, dd*dd),
                  chi, chi, dd, dd)
    Cbig = reshape(permutedims(Cb,(1,4,2,3)), cd, cd)
    Cbig = (Cbig+Cbig')/2
    Pflip = [0.0 1.0; 1.0 0.0]
    U = kron(Pflip,(R+R')/2)
    Cbig = (Cbig + U*Cbig*U')/2
    FU = eigen(Symmetric((U+U')/2))
    vectors = Vector{Vector{Float64}}(); values = Float64[]; parities = Int[]
    for parity in (-1,1)
        ids = findall(x -> parity*x > 0, FU.values)
        isempty(ids) && continue
        V = FU.vectors[:,ids]
        FB = eigen(Symmetric((V'*Cbig*V + (V'*Cbig*V)')/2))
        for j in eachindex(FB.values)
            v = V*FB.vectors[:,j]
            pivot = argmax(abs.(v)); v[pivot] < 0 && (v .*= -1)
            push!(vectors,v); push!(values,FB.values[j]); push!(parities,parity)
        end
    end
    order = sortperm(eachindex(values), by=i->(-abs(values[i]),-parities[i]))
    keep = order[1:min(chimax,cd)]
    Z = hcat(vectors[keep]...)
    Cn = Z'*Cbig*Z; Cn = (Cn+Cn')/2; Cn ./= maximum(abs,Cn)
    Rn = Matrix(Diagonal(Float64.(parities[keep])))
    Cn,Z,Rn
end

function sym_flip_edge(T,R)
    chi = size(T,1); dd = size(T,3)
    A = reshape(R*reshape(T,chi,chi*dd),chi,chi,dd)
    B = permutedims(A,(2,1,3))
    B = reshape(R*reshape(B,chi,chi*dd),chi,chi,dd)
    B = permutedims(B,(2,1,3))
    (T+B[:,:,2:-1:1])/2
end

function grow_T_raw(T, adr, Z)
    chi = size(T,1); dd = size(adr,1); cd = chi*dd
    res5 = reshape(reshape(permutedims(T,(3,1,2)), dd, chi*chi)' * reshape(adr, dd, dd*dd*dd),
                   chi, chi, dd, dd, dd)
    Tbig = reshape(permutedims(res5,(1,3,2,5,4)), cd, cd, dd)
    chin = size(Z,2)
    Tmid = reshape(Z' * reshape(Tbig, cd, cd*dd), chin, cd, dd)
    permutedims(reshape(Z' * reshape(permutedims(Tmid,(2,1,3)), cd, chin*dd),
                        chin, chin, dd), (2,1,3))
end

function normalize_edge(T; reference=nothing)
    T ./= maximum(abs, T)
    if reference !== nothing && size(reference) == size(T)
        sum(T .* reference) < 0 && (T .*= -1)
    end
    T
end

# ---- transfer-channel spectrum with Z2 parity labels -------------------------
"Leading channel spectrum of E = sum_p T_p (x) T_p together with its parity
 under R (x) R.  T_p is symmetric, so E is symmetric and the spectrum real."
function channel_spectrum(T, R; nlevels=8)
    chi = size(T,1); d = size(T,3)
    E = zeros(chi*chi, chi*chi)
    for p in 1:d
        A = T[:,:,p]; E .+= kron(A,A)
    end
    E = (E+E')/2
    F = eigen(Symmetric(E))
    order = sortperm(abs.(F.values), rev=true)[1:min(nlevels, chi*chi)]
    U = kron(R,R)
    values = abs.(F.values[order])
    parity = [F.vectors[:,i]' * U * F.vectors[:,i] for i in order]
    values, parity
end

function xi_exact(beta)
    bdual = -0.5*log(tanh(beta))
    beta < BETA_C ? 1/(2*(bdual-beta)) : 1/(4*(beta-bdual))
end

@printf("MEASURE-XI betas=%s chis=%s (tildeS is NOT recomputed)\n",
        string(XBETAS), string(XCHIS)); flush(stdout)

for beta in XBETAS
    a1, _ = ising_tensors(beta)
    for chi1 in XCHIS
        C1,T1 = init_CT(a1)
        R1 = [0.0 1.0; 1.0 0.0]
        spectrum_old = Float64[]; drift = Inf; itdone = XNMAX
        t0 = time()
        for it in 1:XNMAX
            Cn,Z1,R1n = ctm_step_Z_flip(C1,T1,a1,chi1,R1)
            T1n = normalize_edge(grow_T_raw(T1,a1,Z1);reference=T1)
            T1n = (T1n+permutedims(T1n,(2,1,3)))/2
            T1n = sym_flip_edge(T1n,R1n)
            T1n = normalize_edge(T1n;reference=T1)
            spectrum = sort(abs.(eigvals(Symmetric(Cn))),rev=true); spectrum ./= spectrum[1]
            drift = length(spectrum)==length(spectrum_old) ?
                    maximum(abs.(spectrum-spectrum_old)) : Inf
            C1,T1,R1 = Cn,T1n,R1n
            spectrum_old = spectrum
            if drift < XTOL
                itdone = it; break
            end
        end
        values, parity = channel_spectrum(T1,R1)
        ratios = values ./ values[1]
        z2split = 1 - ratios[2]
        xi_naive = ratios[2] < 1 ? -1/log(ratios[2]) : Inf
        k = findfirst(r -> r < 1 - 1e-10, ratios[2:end])
        xi_conn = k === nothing ? Inf : -1/log(ratios[k+1])
        @printf("SUMMARY-XI beta=%.6f chi1=%3d it=%6d drift=%.1e xi_conn=%10.5f xi_naive=%10.3e z2split=%9.2e xi_exact=%8.5f ratio=%6.3f par=%s sec=%.1f\n",
                beta, chi1, itdone, drift, xi_conn, xi_naive, z2split,
                xi_exact(beta), xi_conn/xi_exact(beta),
                join([@sprintf("%+.2f",p) for p in parity[1:min(4,end)]],","),
                time()-t0)
        flush(stdout)
    end
end
println("DONE MEASURE-XI")
