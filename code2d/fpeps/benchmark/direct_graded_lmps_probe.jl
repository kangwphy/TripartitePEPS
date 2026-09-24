# Independent TFIM-style grown-tail audit. This does not certify replica sewing.
include(joinpath(@__DIR__, "..", "src", "FermionicPEPS.jl"))
using .FermionicPEPS, TensorKit, MPSKit, PEPSKit, LinearAlgebra, Random
using Serialization, TOML

function direct_probe(boundary; seed=610, tolerance=1e-11, maxiter=8000)
    length(boundary.state)==length(boundary.transfer)==1 ||
        throw(ArgumentError("one-site direct tails cannot discard sites of a periodic boundary"))
    boundary.converged || error("unconverged VUMPS boundary")
    AL, AR = boundary.state.AL[1], boundary.state.AR[1]
    O = boundary.transfer[1]
    raw = FermionicPEPS._grow_boundary(AL, PEPSKit.ket(O), PEPSKit.bra(O))
    # Same pivotal conversion as the independently CAR-validated finite row
    # contraction: only the auxiliary cuts are fused, never physical cut legs.
    corrected = PEPSKit.twistdual(raw, (2, 3))
    Q = permute(corrected, ((1, 2, 3, 4, 5), (6, 7, 8)))
    Vl = space(Q, 1) ⊗ space(Q, 2) ⊗ space(Q, 3)
    Vr = domain(Q)
    UL = isomorphism(ComplexF64, fuse(Vl) ← Vl)
    UR = isomorphism(ComplexF64, fuse(Vr) ← Vr)
    physical = id(ComplexF64, space(Q, 4)) ⊗ id(ComplexF64, space(Q, 5))
    D = (UL ⊗ physical) * Q * UR'
    rng = MersenneTwister(seed)
    L = randn(rng, ComplexF64, space(AL, 1) ← space(D, 1))
    R = randn(rng, ComplexF64, dual(space(D, 4)) ← dual(space(AR, 4)))
    normalize!(L); normalize!(R)
    fl = x -> MPSKit.transfer_left(x, D, AL)
    fr = x -> MPSKit.transfer_right(x, D, AR)
    residuals = (Inf, Inf)
    used = maxiter
    for iteration in 1:maxiter
        Ln, Rn = fl(L), fr(R)
        residuals = (FermionicPEPS._scaled_residual(Ln, L),
                     FermionicPEPS._scaled_residual(Rn, R))
        normalize!(Ln); normalize!(Rn)
        for (next, previous) in ((Ln, L), (Rn, R))
            overlap = dot(previous, next)
            abs(overlap) > eps() && rmul!(next, conj(overlap) / abs(overlap))
        end
        L, R = Ln, Rn
        if iteration >= 4 && maximum(residuals) <= tolerance
            used = iteration
            break
        end
    end
    G = L * R
    B = (L ⊗ physical) * D * R
    target = AL * G
    kappa = dot(target, B) / dot(target, target)
    bg_residual = norm(B - kappa * target) / norm(B)
    old = lmps_objects(boundary)
    # Compare only up to independently normalized tail scalar; no gauge or
    # entropy-reference fit is inserted in either contraction.
    old_metric_residual = FermionicPEPS._scaled_residual(G, old.G)
    old_block_residual = FermionicPEPS._scaled_residual(B, old.B)
    singular = reduce(vcat, (svdvals(b) for (_, b) in blocks(G)))
    diagnostics = Dict("seed"=>seed, "iterations"=>used,
        "left_residual"=>residuals[1], "right_residual"=>residuals[2],
        "bg_residual"=>bg_residual, "old_metric_residual"=>old_metric_residual,
        "old_block_residual"=>old_block_residual,
        "metric_min_relative_singular"=>minimum(singular)/maximum(singular),
        "kappa_real"=>real(kappa), "kappa_imag"=>imag(kappa),
        "tail_converged"=>maximum(residuals)<=tolerance,
        "physical_replica_benchmark_certified"=>false)
    (; L, R, D, UL, UR, G, B, diagnostics)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    boundary = deserialize(ARGS[1])
    mkpath(ARGS[2])
    for seed in (610, 611)
        result = direct_probe(boundary; seed)
        serialize(joinpath(ARGS[2], "seed$(seed).jls"), result)
        open(joinpath(ARGS[2], "seed$(seed).toml"), "w") do io
            TOML.print(io, result.diagnostics)
        end
        println(result.diagnostics); flush(stdout)
    end
end
