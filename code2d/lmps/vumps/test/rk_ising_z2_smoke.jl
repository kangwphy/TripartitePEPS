# Opt-in RK-Ising Z2 adapter checks.  The normal LMPSVUMPS entry point is not
# modified or reloaded.  Run only through Slurm's preempt partition at Pitt CRC.
get(ENV, "SLURM_JOB_PARTITION", "") == "preempt" ||
    error("Submit this numerical test with Slurm --partition=preempt")

using Test
using LinearAlgebra
using Random
using SHA
using LMPSVUMPS
import TensorKit

const VUMPS_ROOT = dirname(@__DIR__)
const LEGACY_FILES = [
    joinpath(VUMPS_ROOT, "src", "LMPSVUMPS.jl"),
    joinpath(VUMPS_ROOT, "src", "algorithms", "boundary.jl"),
    joinpath(VUMPS_ROOT, "src", "algorithms", "direct_route.jl"),
    joinpath(VUMPS_ROOT, "src", "algorithms", "sixr_closure.jl"),
    joinpath(VUMPS_ROOT, "src", "models", "rk_ising.jl"),
    joinpath(VUMPS_ROOT, "src", "models", "rk_ising_observables.jl"),
]
file_sha(path) = bytes2hex(open(sha256, path))
const LEGACY_SHA_BEFORE = Dict(path => file_sha(path) for path in LEGACY_FILES)

include(joinpath(VUMPS_ROOT, "src", "models", "rk_ising_z2.jl"))
using .RKIsingZ2

# Intentionally independent elementary implementations for small chi=4 tests;
# none of the adapter's internal transformation/spectrum routines are reused.
function transform_every_leg(A, H)
    B = zeros(ComplexF64, size(A))
    for J in CartesianIndices(B), I in CartesianIndices(A)
        coefficient = prod(H[J[k], I[k]] for k in 1:ndims(A))
        B[J] += coefficient * A[I]
    end
    B
end

function dense_transfer_spectrum(M)
    chi = size(M, 1)
    E = zeros(ComplexF64, chi^2, chi^2)
    for p in axes(M, 2)
        A = Matrix(@view M[:, p, :])
        E .+= kron(conj(A), A)
    end
    eigvals(E)
end

"Maximum error after greedily matching two small eigenvalue multisets."
function spectrum_multiset_error(a, b)
    length(a) == length(b) || return Inf
    remaining = ComplexF64.(b)
    err = 0.0
    for z in sort(ComplexF64.(a); by=abs, rev=true)
        delta, index = findmin(abs.(remaining .- z))
        err = max(err, delta)
        deleteat!(remaining, index)
    end
    err
end

function canonical_residuals(state)
    AL0, AR0, AC0 = (convert(Array, state.AL[1]),
                    convert(Array, state.AR[1]),
                    convert(Array, state.AC[1]))
    C = convert(Array, state.C[1])
    chi = size(C, 1)
    AL = reshape(AL0, chi, :, chi)
    AR = reshape(AR0, chi, :, chi)
    AC = reshape(AC0, chi, :, chi)
    left_gram = zeros(ComplexF64, chi, chi)
    right_gram = zeros(ComplexF64, chi, chi)
    ac_left, ac_right = similar(AC), similar(AC)
    for p in axes(AL, 2)
        A = Matrix(@view AL[:, p, :])
        R = Matrix(@view AR[:, p, :])
        left_gram .+= A' * A
        right_gram .+= R * R'
        ac_left[:, p, :] = A * C
        ac_right[:, p, :] = C * R
    end
    (; left=norm(left_gram - I), right=norm(right_gram - I),
       center_left=norm(ac_left - AC) / norm(AC),
       center_right=norm(ac_right - AC) / norm(AC))
end

@testset "Opt-in Z2 RK-Ising adapter and legacy regression" begin
    beta, chi, chi_even = 0.30, 4, 2
    model = rk_ising_z2_peps(beta)
    Aoriginal = permutedims(rk_ising(beta).A, (1, 2, 5, 4, 3))
    Aparity = convert(Array, model.peps[1, 1])
    H = model.hadamard

    @testset "Exact tensor and physical basis" begin
        @test H' * H ≈ Matrix{ComplexF64}(I, 2, 2) atol=1e-14
        @test size(Aparity) == (2, 2, 2, 2, 2)
        @test Aparity ≈ transform_every_leg(Aoriginal, H) atol=1e-13
        @test transform_every_leg(Aparity, H') ≈ Aoriginal atol=1e-13
        @test model.source.A ≈ rk_ising(beta).A atol=1e-14
        forbidden_norm2 = sum(abs2(Aparity[I]) for I in CartesianIndices(Aparity)
                              if isodd(sum(Tuple(I)) - 5))
        @test sqrt(forbidden_norm2) / norm(Aparity) < 1e-13
        @test norm(Aparity) ≈ norm(Aoriginal) atol=1e-13
    end

    Random.seed!(0x5a2)
    symmetric = solve_rk_ising_z2(beta, chi; chi_even,
        tolerance=1e-9, diagnostic_tolerance=1e-7,
        max_iterations=1000, verbosity=1, spectrum_num=8,
        spectrum_tol=1e-12)
    boundary = symmetric.boundary
    observables = vumps_boundary_observables(boundary, beta; spectrum_num=8)
    Mraw = convert(Array, boundary.state.AL[1])
    M = reshape(Mraw, chi, 4, chi)

    @testset "Symmetry preserved by solve and dense export" begin
        @test boundary.diagnostics.converged
        @test symmetric.audits.symmetry == "bosonic_Z2"
        @test symmetric.audits.chi_total == chi
        @test symmetric.audits.chi_even == chi_even
        @test symmetric.audits.chi_odd == chi - chi_even
        @test !symmetric.audits.dense_reoptimized
        @test symmetric.audits.eigenvalue_basis_residual < 1e-10
        @test symmetric.audits.export_roundtrip_residual < 1e-12
        @test abs(observables.magnetization) < 1e-10
        @test observables.free_energy_rel_error < 2e-5
        @test isfinite(observables.xi) && observables.xi > 0
        @test all(isfinite, (observables.free_energy, observables.entropy_vn,
                            observables.entropy_renyi2))
        @test sum(observables.schmidt_probabilities) ≈ 1 atol=1e-13

        # TK arrays order the even and odd degeneracy spaces consecutively.
        # Physical ket/bra factors are 0,1 in parity basis; export applies H
        # to each, so global spin reversal becomes X on both dense factors.
        charges = vcat(zeros(Int, chi_even), ones(Int, chi - chi_even))
        graded_AL = convert(Array, symmetric.symmetric_state.AL[1])
        forbidden_norm2 = 0.0
        for l in 1:chi, k in 1:2, b in 1:2, r in 1:chi
            if isodd(charges[l] + (k - 1) + (b - 1) + charges[r])
                forbidden_norm2 += abs2(graded_AL[l, k, b, r])
            end
        end
        @test sqrt(forbidden_norm2) / norm(graded_AL) < 1e-13
        virtual_parity = Diagonal((-1.0) .^ charges)
        symmetry_error = 0.0
        for k in 1:2, b in 1:2
            p = k + 2 * (b - 1)
            flip_p = (3 - k) + 2 * (2 - b)
            symmetry_error += norm(M[:, p, :] -
                virtual_parity * M[:, flip_p, :] * virtual_parity)^2
        end
        @test sqrt(symmetry_error) / norm(M) < 1e-12

        canonical = canonical_residuals(boundary.state)
        @test maximum(values(canonical)) < 1e-7
        exported_again = dense_original_spin_mps(symmetric.symmetric_state)
        for field in (:AL, :AR, :AC, :C)
            @test convert(Array, getproperty(exported_again, field)[1]) ≈
                  convert(Array, getproperty(boundary.state, field)[1]) atol=1e-13
        end
    end

    @testset "Both transfer charge sectors retained" begin
        spectra = symmetric.sector_spectra
        @test !isempty(spectra.even)
        @test !isempty(spectra.odd)
        # With a 2+2 bond space each sector has dimension 8.  Request all
        # eigenvalues here so a missing charged sector cannot pass silently.
        @test length(spectra.even) == 8
        @test length(spectra.odd) == 8
        merged = vcat(spectra.even, spectra.odd)
        explicit = dense_transfer_spectrum(M)
        @test spectrum_multiset_error(merged, explicit) < 1e-8
        @test spectrum_multiset_error(spectra.combined, explicit) < 1e-8
        @test maximum(abs.(explicit)) ≈ 1 atol=1e-8
        @test spectra.xi_all ≈ observables.xi rtol=1e-7 atol=1e-8
    end

    @testset "Unmodified dense solve remains available" begin
        Random.seed!(0x5a2)
        legacy = solve_vumps_boundary(rk_ising(beta), chi;
            tolerance=1e-9, diagnostic_tolerance=1e-7,
            max_iterations=1000, verbosity=1)
        legacy_obs = vumps_boundary_observables(legacy, beta; spectrum_num=8)
        @test legacy.diagnostics.converged
        @test legacy_obs.free_energy_rel_error < 2e-5
        @test abs(legacy_obs.magnetization) < 1e-6
        @test abs(legacy_obs.free_energy - observables.free_energy) /
              abs(observables.exact_free_energy) < 2e-5
        @test legacy_obs.xi ≈ observables.xi rtol=0.05
        @info "RK-Ising Z2 smoke observables" beta chi chi_even legacy_obs observables
        @info "RK-Ising Z2 adapter audits" audits=symmetric.audits spectra=symmetric.sector_spectra
        # This also catches accidental runtime source rewriting by a driver.
        @test Dict(path => file_sha(path) for path in LEGACY_FILES) == LEGACY_SHA_BEFORE
    end
end
