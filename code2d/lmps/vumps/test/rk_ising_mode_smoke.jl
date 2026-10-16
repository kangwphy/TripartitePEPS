# The two campaign switches are independent.  These deliberately small tests
# exercise routing and full product-state closures; the separate Z2 adapter
# suite tests actual VUMPS optimization and lossless boundary export.
get(ENV, "SLURM_JOB_PARTITION", "") == "preempt" ||
    error("Submit numerical tests through genuine Slurm --partition=preempt")
haskey(ENV, "SLURM_JOB_ID") || error("Missing Slurm allocation")

using Test
using LinearAlgebra
using SHA
using LMPSVUMPS
import ITensors

const MODE_PACKAGE = dirname(@__DIR__)
const MODE_LEGACY_FILES = [
    joinpath(MODE_PACKAGE, "src", "algorithms", "boundary.jl"),
    joinpath(MODE_PACKAGE, "src", "algorithms", "direct_route.jl"),
    joinpath(MODE_PACKAGE, "src", "algorithms", "sixr_closure.jl"),
    joinpath(MODE_PACKAGE, "src", "models", "rk_ising_observables.jl"),
]
mode_sha(path) = bytes2hex(open(sha256, path))
const MODE_LEGACY_HASHES = Dict(path => mode_sha(path) for path in MODE_LEGACY_FILES)

# Including the driver must not require campaign paths or start a calculation.
withenv("RK_CAMPAIGN_ROOT" => nothing, "SLURM_ARRAY_TASK_ID" => nothing) do
    include(joinpath(MODE_PACKAGE, "production", "rk_ising", "rk_ising_point.jl"))
end
const Point = RKIsingPoint

function mode_test_config(z2, fix)
    Dict{String,Any}(
        "use_z2" => z2, "use_cap_fix" => fix,
        "sixr_maxiter" => 120, "sixr_far_tail_maxiter" => 120,
        "sixr_tolerance" => 1e-12,
        "far_boundary_tolerance" => 1e-10,
        "far_boundary_tail_tolerance" => 1e-11,
        "cap_fix_tolerance" => 1e-12,
        "cap_fix_state_tolerance" => 1e-10,
        "cap_fix_amplitude_tolerance" => 1e-11,
        "scale_tolerance" => 1e-10,
    )
end

function mode_matrix_seam(E)
    n, m = size(E)
    n == m || throw(DimensionMismatch("square seam required"))
    left = ITensors.Index(n, "Chi,L,A,1_1b")
    right = ITensors.Index(n, "Chi,R,A,1_1b")
    ITensors.ITensor[ITensors.ITensor(Matrix(transpose(E)), left, right)]
end

@testset "Independent RK-Ising symmetry and six-R repair switches" begin
    @testset "Default and four-way dispatch" begin
        defaults = Point.mode_config(Dict{String,Any}())
        @test !defaults.use_z2
        @test !defaults.use_cap_fix
        @test defaults.series_label == "no_constraint"
        @test_throws ArgumentError Point.mode_config(Dict("use_z2" => "true"))
        @test_throws ArgumentError Point.mode_config(Dict("use_cap_fix" => 1))
        @test_throws ArgumentError Point.mode_config(Dict{String,Any}(
            "use_z2" => true, "use_cap_fix" => true, "series_label" => "no_constraint"))

        for z2 in (false, true), fix in (false, true)
            config = mode_test_config(z2, fix)
            modes = Point.mode_config(config)
            @test modes.use_z2 == z2
            @test modes.use_cap_fix == fix
            expected_label = z2 ? (fix ? "Z2_fix" : "Z2") :
                                   (fix ? "no_constraint_fix" : "no_constraint")
            @test modes.series_label == expected_label
            @test Point.boundary_solver(config) === (z2 ?
                Point.RKIsingZ2.solve_rk_ising_z2 : Point.dense_boundary)
            options = Point.sixr_options(config)
            @test options.endpoint_solver === (fix ?
                LMPSVUMPS.m2_endpoint_cap_selected :
                Point.SixRModes.m2_endpoint_legacy_retained)
            @test options.endpoint_options.tail_maxiter == 120
            @test options.endpoint_options.retain_unconverged
            # Changing only one switch must not change the other stage.
            @test Point.boundary_solver(config) ===
                Point.boundary_solver(mode_test_config(z2, !fix))
            @test options.endpoint_solver ===
                Point.sixr_options(mode_test_config(!z2, fix)).endpoint_solver
        end
    end

    @testset "All four settings use the same complete bare/direct product network" begin
        M = reshape(ComplexF64[1, 0, 0, 0], 1, 4, 1)
        original = (G=ones(ComplexF64, 1, 1), B=M)
        rails = LMPSVUMPS.direct_cardinal_regional_rails(original, M, M, M)
        reference = LMPSVUMPS.m2_contract_regional_six_R(rails.rails, 2;
            nmax=120, tol=1e-12, verbose=false)
        for z2 in (false, true), fix in (false, true)
            observed = NamedTuple[]
            observer = (spec, far, e) -> push!(observed,
                (; copies=length(spec.cyc), has_cap=far !== nothing,
                   converged=e.converged, selection=e.selection_method))
            raw_options = Point.sixr_options(mode_test_config(z2, fix))
            options = merge(raw_options, (; endpoint_options=
                merge(raw_options.endpoint_options, (; verbose=false))))
            result = LMPSVUMPS.m2_contract_regional_six_R(rails.rails, 2;
                options..., seam_observer=observer, verbose=false)
            @test length(observed) == 9
            @test count(e -> e.copies == 1, observed) == 3
            @test count(e -> e.copies == 2, observed) == 6
            @test all(e -> e.converged, observed)
            @test all(e -> e.has_cap, filter(e -> e.copies == 2, observed))
            @test all(e -> e.selection == (fix ? "normalized_far_cap_power" :
                "legacy_arnoldi_direction_with_far_cap_intercept"), observed)
            for field in (:tildeS, :S2A, :S2B, :S2C, :S3)
                @test getproperty(result, field) ≈ getproperty(reference, field) atol=1e-10
                @test abs(getproperty(result, field)) < 1e-10
            end
        end
    end

    @testset "Unfixed adapter preserves old values and retain-last semantics" begin
        E = ComplexF64[0.8 0.2; 0.4 0.6]
        cap = ComplexF64[2, 1]
        Ts = mode_matrix_seam(E)
        old = LMPSVUMPS.m2_endpoint(Ts; nmax=2000, tol=1e-12,
            far_boundary=cap, far_boundary_tolerance=1e-10,
            far_boundary_tail_tolerance=1e-11)
        legacy = Point.SixRModes.m2_endpoint_legacy_retained(Ts;
            nmax=2000, tail_maxiter=2000, tol=1e-12,
            far_boundary=cap, far_boundary_tolerance=1e-10,
            far_boundary_tail_tolerance=1e-11)
        @test legacy.converged
        @test legacy.arnoldi_used
        @test legacy.lambda ≈ old.lambda atol=1e-10
        @test legacy.far_prefactor .* legacy.R ≈ old.far_prefactor .* old.R atol=2e-8
        @test legacy.log_prefactor ≈ old.log_prefactor atol=2e-8

        # Exact eigenvectors are easy, but a deliberately short tail cannot
        # align the supplied cap. Retention must not call that convergence.
        slow = mode_matrix_seam(Matrix(Diagonal(ComplexF64[1, 0.995])))
        short = Point.SixRModes.m2_endpoint_legacy_retained(slow;
            nmax=120, tail_maxiter=24, tol=1e-12,
            far_boundary=ComplexF64[1, 1], far_boundary_tolerance=1e-10,
            far_boundary_tail_tolerance=1e-11, retain_unconverged=true)
        @test !short.converged
        @test !isempty(short.failure_reason)
        @test isfinite(short.log_prefactor)
        @test all(isfinite, short.R)
        @test_throws ErrorException Point.SixRModes.m2_endpoint_legacy_retained(slow;
            nmax=120, tail_maxiter=24, tol=1e-12,
            far_boundary=ComplexF64[1, 1], far_boundary_tolerance=1e-10,
            far_boundary_tail_tolerance=1e-11, retain_unconverged=false)
    end

    @testset "Z2 full transfer length distinguishes cat degeneracy" begin
        regular = Point.spectrum_lengths([1.0, 0.5, 0.2])
        @test regular.xi ≈ -1 / log(0.5)
        @test !regular.xi_full_degenerate
        @test regular.xi_subleading == regular.xi
        cat = Point.spectrum_lengths([1.0, 1.0, 0.5])
        @test isinf(cat.xi)
        @test cat.xi_full_peripheral_count == 2
        @test cat.xi_full_degenerate
        @test cat.xi_subleading ≈ -1 / log(0.5)
        @test cat.xi_subleading_rank == 3
        peripheral = Point.spectrum_lengths([1.0, 1.0])
        @test isinf(peripheral.xi)
        @test isnan(peripheral.xi_subleading)
        @test peripheral.xi_subleading_rank == 0
        @test Point.spectrum_lengths([1.0, 0.0]).xi == 0
        @test_throws ErrorException Point.spectrum_lengths([0.0, 0.0])
        @test_throws ErrorException Point.spectrum_lengths([1.0])
    end

    @test Dict(path => mode_sha(path) for path in MODE_LEGACY_FILES) == MODE_LEGACY_HASHES
end
