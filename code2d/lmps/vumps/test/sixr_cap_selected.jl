# Independent tests for the opt-in, far-cap-selected six-R seam solver.
# This is deliberately separate from the legacy/default regression suite.
get(ENV, "SLURM_JOB_PARTITION", "") == "preempt" ||
    error("Submit this numerical test with Slurm --partition=preempt")

using Test
using LinearAlgebra
using SHA
using LMPSVUMPS
import ITensors

const CS_ROOT = dirname(@__DIR__)
const CS_LEGACY_PATHS = [
    joinpath(CS_ROOT, "src", "algorithms", "sixr_closure.jl"),
    joinpath(CS_ROOT, "src", "algorithms", "direct_route.jl"),
    joinpath(CS_ROOT, "src", "algorithms", "boundary.jl"),
]
cs_sha(path) = bytes2hex(open(sha256, path))
const CS_LEGACY_HASHES = Dict(path => cs_sha(path) for path in CS_LEGACY_PATHS)

# During standalone development the helper may not yet be included by the
# package entry point.  This adds only its new names, never replacing the old
# m2_endpoint implementation.
if !isdefined(LMPSVUMPS, :cap_selected_power)
    Base.include(LMPSVUMPS,
        joinpath(CS_ROOT, "src", "algorithms", "sixr_cap_selected.jl"))
end

function cs_solve(E, cap; maxiter=4000, state_tolerance=1e-11,
                  eigen_tolerance=1e-11, amplitude_tolerance=1e-11,
                  retain_unconverged=false)
    LMPSVUMPS.cap_selected_power(x -> E * x, cap;
        maxiter, tol=1e-12, state_tolerance, eigen_tolerance,
        amplitude_tolerance, window=24, retain_unconverged)
end

# The reconstructed vector includes the cap's magnitude and complex phase;
# looking only at normalized R would miss precisely the normalization bug.
cs_amplitude(result) = result.far_prefactor .* result.R

"Explicit finite-depth contraction, independent of the tested iteration loop."
function cs_finite_power(E, cap, lambda, depth)
    x = ComplexF64.(cap)
    for _ in 1:depth
        x = (E * x) / lambda
    end
    x
end

"One ITensor seam with its right/input orientation explicitly checked below."
function cs_matrix_seam(E)
    n, m = size(E)
    n == m || throw(DimensionMismatch("square transfer required"))
    left = ITensors.Index(n, "Chi,L,A,1_1b")
    right = ITensors.Index(n, "Chi,R,A,1_1b")
    # Contraction sums the first (left/input) leg, so T[l,r] = E[r,l].
    Ts = ITensors.ITensor[
        ITensors.ITensor(Matrix(transpose(E)), left, right),
    ]
    Ts
end

"A GHZ norm seam: sum_s |s><s| X |s><s|, implemented with two tensors."
function cs_ghz_seam()
    AL = ITensors.Index(2, "Chi,L,A,1_1b")
    AR = ITensors.Index(2, "Chi,R,A,1_1b")
    BL = ITensors.Index(2, "Chi,L,B,1_1b")
    BR = ITensors.Index(2, "Chi,R,B,1_1b")
    physical = ITensors.Index(2, "Physical,GHZ")
    M = zeros(ComplexF64, 2, 2, 2)
    M[1, 1, 1] = 1
    M[2, 2, 2] = 1
    ITensors.ITensor[
        ITensors.ITensor(M, AL, physical, AR),
        ITensors.ITensor(conj.(M), BL, physical, BR),
    ]
end

@testset "Far-cap-selected seam: exact limits and legacy isolation" begin
    @testset "Semisimple degenerate leading eigenvalue retains the whole cap" begin
        E = Matrix(Diagonal(ComplexF64[1, 1, 0.25]))
        cap = ComplexF64[3, 4, 2]
        cap_before = copy(cap)
        expected = ComplexF64[3, 4, 0]
        result = cs_solve(E, cap)
        @test result.converged
        @test result.lambda ≈ 1 atol=1e-11
        @test cs_amplitude(result) ≈ expected atol=2e-9 rtol=2e-9
        @test norm(result.R) ≈ 1 atol=1e-12
        @test abs(result.far_prefactor) ≈ 5 atol=2e-9
        @test result.log_prefactor ≈ log(5) atol=2e-9
        @test result.residual <= 1e-11
        @test cap == cap_before
        @test cs_finite_power(E, cap, 1, 48) ≈ expected atol=1e-13

        # Different physical caps must not be collapsed to the same arbitrary
        # eigenvector in the two-dimensional leading eigenspace.
        cap2 = ComplexF64[4, -3, 2]
        second = cs_solve(E, cap2)
        @test cs_amplitude(second) ≈ ComplexF64[4, -3, 0] atol=2e-9
        @test abs(dot(result.R, second.R)) < 1e-9
    end

    @testset "Nonnormal transfer needs an oblique, not orthogonal, projection" begin
        S = ComplexF64[1 0 2; 0 1 -1; 0 0 1]
        D = Matrix(Diagonal(ComplexF64[1, 1, 0.2]))
        E = S * D / S
        cap = ComplexF64[4, 5, 3]
        expected = ComplexF64[-2, 8, 0]
        orthogonal_projection = ComplexF64[4, 5, 0]
        result = cs_solve(E, cap)
        @test norm(E' * E - E * E') > 1
        @test result.converged
        @test cs_amplitude(result) ≈ expected atol=2e-9 rtol=2e-9
        @test norm(cs_amplitude(result) - orthogonal_projection) > 1
        @test cs_finite_power(E, cap, 1, 48) ≈ expected atol=1e-12
    end

    @testset "Complex leading phase and far-cap scale are not discarded" begin
        lambda = 1.15 * cis(0.23)
        E = lambda .* Matrix(Diagonal(ComplexF64[1, 1, 0.3]))
        cap = ComplexF64[1 + 2im, -0.4 + 0.2im, 0.7 - 0.3im]
        expected = ComplexF64[cap[1], cap[2], 0]
        result = cs_solve(E, cap)
        @test result.converged
        @test result.lambda ≈ lambda atol=1e-10 rtol=1e-10
        @test cs_amplitude(result) ≈ expected atol=2e-8 rtol=2e-8
        @test cs_finite_power(E, cap, lambda, 48) ≈ expected atol=1e-12

        scale = -0.7 + 1.1im
        scaled = cs_solve(E, scale .* cap)
        @test cs_amplitude(scaled) ≈ scale .* expected atol=2e-8 rtol=2e-8
        @test scaled.log_prefactor - result.log_prefactor ≈ log(abs(scale)) atol=2e-8
    end

    @testset "A small but nonzero gap is not merged into exact degeneracy" begin
        E = Matrix(Diagonal(ComplexF64[1, 0.999, 0.2]))
        cap = ComplexF64[1, 2, 1]
        result = cs_solve(E, cap; maxiter=50000,
            state_tolerance=1e-13, eigen_tolerance=1e-13)
        @test result.converged
        @test cs_amplitude(result) ≈ ComplexF64[1, 0, 0] atol=2e-8
        @test abs(result.R[2]) < 2e-8
        @test result.lambda ≈ 1 atol=1e-11
        # A short budget must not silently label a finite-depth combination
        # as the thermodynamic fixed point.
        short = cs_solve(E, cap; maxiter=60, retain_unconverged=true)
        @test !short.converged
        @test !isempty(short.failure_reason)
    end

    @testset "Different peripheral phases have no single-vector limit" begin
        E = Matrix(Diagonal(ComplexF64[1, -1, 0.2]))
        cap = ComplexF64[1, 1, 1]
        @test_throws ErrorException cs_solve(E, cap; maxiter=120)
        diagnostic = cs_solve(E, cap; maxiter=120, retain_unconverged=true)
        @test !diagnostic.converged
        @test !isempty(diagnostic.failure_reason)
        # The failure is not simply a large eigenvalue or a global phase:
        # both are accepted in the preceding test.
        @test diagnostic.residual > 0.1
        @test_throws ArgumentError cs_solve(E, zeros(ComplexF64, 3))
    end

    @testset "ITensor simple-spectrum regression against unchanged old solver" begin
        E = ComplexF64[0.8 0.2; 0.4 0.6]
        cap = ComplexF64[2, 1]
        expected = fill(ComplexF64(5 / 3), 2)
        Ts = cs_matrix_seam(E)
        lefts, lmap, rmap = LMPSVUMPS.m2_left_right(Ts)
        mapped, _ = LMPSVUMPS.m2_apply_array(Ts, lefts, lmap, rmap, cap)
        @test mapped ≈ E * cap atol=1e-14
        old = LMPSVUMPS.m2_endpoint(Ts; nmax=2000, tol=1e-12,
            far_boundary=cap, far_boundary_tolerance=1e-10,
            far_boundary_tail_tolerance=1e-11)
        new = LMPSVUMPS.m2_endpoint_cap_selected(Ts;
            nmax=2000, tail_maxiter=2000, tol=1e-12,
            far_boundary=cap, far_boundary_tolerance=1e-10,
            far_boundary_tail_tolerance=1e-11)
        @test new.converged
        @test new.lambda ≈ old.lambda atol=1e-10
        @test cs_amplitude(new) ≈ expected atol=2e-8
        @test cs_amplitude(new) ≈ cs_amplitude(old) atol=2e-8
        @test new.log_prefactor ≈ old.log_prefactor atol=2e-8
        @test old.residual < 1e-10
        @test new.residual < 1e-10
    end

    @testset "GHZ/cat seam preserves cap weights and known finite-depth amplitude" begin
        Ts = cs_ghz_seam()
        lefts, lmap, rmap = LMPSVUMPS.m2_left_right(Ts)
        cap = ComplexF64[0.8 0.1; 0.1 0.2]
        expected = Matrix(Diagonal(ComplexF64[0.8, 0.2]))
        cap_before = copy(cap)
        # Independent finite-depth GHZ channel kills off-diagonal entries in
        # one row; every subsequent row leaves both diagonal sectors intact.
        finite = copy(cap)
        for _ in 1:16
            finite = Diagonal([1, 0]) * finite * Diagonal([1, 0]) +
                     Diagonal([0, 1]) * finite * Diagonal([0, 1])
        end
        mapped, _ = LMPSVUMPS.m2_apply_array(Ts, lefts, lmap, rmap, cap)
        @test mapped ≈ finite atol=1e-14
        result = LMPSVUMPS.m2_endpoint_cap_selected(Ts;
            nmax=120, tail_maxiter=120, tol=1e-12, far_boundary=cap,
            far_boundary_tolerance=1e-10, far_boundary_tail_tolerance=1e-11)
        rho = cs_amplitude(result)
        @test result.converged
        @test rho ≈ expected atol=1e-11
        @test rho ≈ finite atol=1e-11
        @test tr(rho) ≈ 1 atol=1e-11
        @test tr(rho * rho) ≈ 0.68 atol=1e-11
        @test result.log_prefactor ≈ 0.5 * log(0.68) atol=1e-11
        @test cap == cap_before

        # The default common-frame corner cap is identity, not an arbitrary
        # one-sector pure state. Both weights must survive this path too.
        symmetric = LMPSVUMPS.m2_endpoint_cap_selected(Ts;
            nmax=120, tail_maxiter=120, tol=1e-12,
            far_boundary_tolerance=1e-10, far_boundary_tail_tolerance=1e-11)
        @test cs_amplitude(symmetric) ≈ Matrix{ComplexF64}(I, 2, 2) atol=1e-11
        @test abs(symmetric.far_prefactor) ≈ sqrt(2) atol=1e-11
    end

    @testset "Full six-R product-state closure and optional solver plumbing" begin
        M = reshape(ComplexF64[1, 0, 0, 0], 1, 4, 1)
        original = (G=ones(ComplexF64, 1, 1), B=M)
        rails = LMPSVUMPS.direct_cardinal_regional_rails(original, M, M, M)
        observed = NamedTuple[]
        observer = (spec, far, endpoint) -> push!(observed,
            (; seam=spec.seam, copies=length(spec.cyc),
               supplied_cap=far !== nothing, converged=endpoint.converged))
        old = LMPSVUMPS.m2_contract_regional_six_R(rails.rails, 2;
            nmax=120, tol=1e-12, verbose=false)
        new = LMPSVUMPS.m2_contract_regional_six_R(rails.rails, 2;
            nmax=120, tol=1e-12, verbose=false,
            endpoint_solver=LMPSVUMPS.m2_endpoint_cap_selected,
            endpoint_options=(tail_maxiter=120,), seam_observer=observer)
        @test length(observed) == 9
        @test count(x -> x.copies == 1, observed) == 3
        @test count(x -> x.copies == 2, observed) == 6
        @test all(x -> x.converged, observed)
        @test all(x -> x.supplied_cap, filter(x -> x.copies == 2, observed))
        for field in (:tildeS, :S2A, :S2B, :S2C, :S3)
            @test getproperty(new, field) ≈ getproperty(old, field) atol=1e-10
            @test abs(getproperty(new, field)) < 1e-10
        end
    end

    @test Dict(path => cs_sha(path) for path in CS_LEGACY_PATHS) == CS_LEGACY_HASHES
end
