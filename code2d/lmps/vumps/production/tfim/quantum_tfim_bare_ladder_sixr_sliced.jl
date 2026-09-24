using LMPSVUMPS
using LinearAlgebra
using Serialization
using SHA
include(joinpath(@__DIR__, "sliced_closure_runtime.jl"))

# This is the bare spatial LR/LTR candidate, NOT direct_finite_depth_maps.
# No endpoint_north field or endpoint checkpoint is used in this calculation.
const INPUT = abspath(ENV["TFIM_BARE_CARDINAL_INPUT"])
const OUT = abspath(ENV["TFIM_BARE_SIXR_OUTPUT"])
const SOURCE_SHA = ENV["TFIM_BARE_SOURCE_SHA"]
const INPUT_SHA = ENV["TFIM_BARE_CARDINAL_SHA"]
const source_bytes = read(INPUT)
bytes2hex(sha256(source_bytes)) == INPUT_SHA || error("cardinal input hash changed")
const cardinal = deserialize(IOBuffer(source_bytes))
cardinal.row.passed || error("bare free-energy/fidelity prerequisite failed")
all(b.converged for b in cardinal.boundaries) || error("unconverged cardinal boundary")
cardinal.original_north.residual_G <= 1e-11 || error("bare G residual failed")
cardinal.original_north.residual_B <= 1e-11 || error("bare B residual failed")
const D = isqrt(size(cardinal.a, 1))
D^2 == size(cardinal.a, 1) || error("expected fused quantum virtual dimension D^2")
const CHI = size(cardinal.M.north, 1)

function boundary_spectrum(A)
    chi, dp, chir = size(A)
    chi == chir || error("unequal retained bonds")
    E = zeros(ComplexF64, chi^2, chi^2)
    for p in 1:dp
        Ap = Matrix(@view A[:, p, :])
        E .+= kron(conj(Ap), Ap)
    end
    magnitudes = sort(abs.(eigvals(E)); rev=true)
    magnitudes[1] > 0 || error("zero transfer radius")
    ratios = magnitudes ./ magnitudes[1]
    xi = length(ratios) < 2 || ratios[2] == 0 ? 0.0 :
         ratios[2] >= 1 ? Inf : -inv(log(ratios[2]))
    (; xi, ratios)
end

println("stage=bare G/B cardinal rails; no endpoint maps read"); flush(stdout)
const rails = direct_cardinal_regional_rails(cardinal.original_north,
    cardinal.M.west, cardinal.M.east, cardinal.M.south; rtol=1e-12)
const spectra = (A=boundary_spectrum(rails.HA), B=boundary_spectrum(rails.HB),
                 C=boundary_spectrum(cardinal.M.south))
const measured = @timed sliced_contract_regional_six_R(
    rails.rails, D; nmax=8000, tol=1e-10, scale_tolerance=5e-8,
    far_boundary_tolerance=5e-7, far_boundary_tail_tolerance=5e-7,
    verbose=true)
const result = measured.value
const seconds = measured.time
const records = (result.endpoint_audit..., result.mixed_corner_audit...)
# The core uses physical-cap propagation, not a separately computed left
# Ritz vector; left_residual is intentionally NaN in this schema. Its
# unavailable value must not contaminate the actual right-residual check.
const max_residual = maximum(r.residual for r in records)
const max_adjoint_gate = maximum(r.adjoint_gate for r in records)
const alignment = maximum(r.prefactor_alignment for r in records)
const drift = maximum(r.prefactor_drift for r in records)
const slope_match = maximum(r.prefactor_slope_match for r in records)
const code_paths = (@__FILE__,
    joinpath(@__DIR__, "sliced_closure_runtime.jl"),
    joinpath(@__DIR__, "chunked_octahedron.jl"),
    joinpath(@__DIR__, "../../src/algorithms/direct_route.jl"),
    joinpath(@__DIR__, "../../src/algorithms/sixr_closure.jl"),
    joinpath(@__DIR__, "../../Manifest.toml"))
const code_sha = join((basename(p)*":"*bytes2hex(open(sha256,p)) for p in code_paths), ';')
const passed = all(isfinite, (max_residual, max_adjoint_gate, alignment, drift, slope_match)) &&
    max_residual <= 1e-8 && max_adjoint_gate <= 1e-11 && alignment <= 5e-7 && drift <= 5e-7 &&
    slope_match <= 5e-7 && result.scale_gate <= 5e-8 &&
    result.endpoint_count == 6 && result.metric_count == 3 &&
    all(isfinite, (result.tildeS, result.S2A, result.S2B, result.S3))
const row = (; method="bare_two_and_three_column_fixedpoints",
    status=passed ? "PASS_NUMERICAL_DIAGNOSTIC" : "FAIL_NUMERICAL_DIAGNOSTIC",
    production_release=false, endpoint_maps_used=false, closure_contraction="exact_one_bond_sliced", D, chi=CHI,
    source_state_sha256=SOURCE_SHA, cardinal_input=INPUT,
    cardinal_input_sha256=INPUT_SHA, code_sha256=code_sha,
    tildeS=result.tildeS, S2A=result.S2A, S2B=result.S2B, S2C=result.S2C, S3=result.S3,
    xiA=spectra.A.xi, xiB=spectra.B.xi, xiC=spectra.C.xi,
    mirror=result.mirror, metric_dropped=rails.metric_dropped,
    max_residual, max_adjoint_gate, legacy_left_residual_available=false,
    alignment, drift, slope_match, scale_gate=result.scale_gate, seconds)
function atomic_save(writer, path)
    mkpath(dirname(path))
    tmp = path * ".tmp." * get(ENV, "SLURM_JOB_ID", "manual")
    open(writer, tmp, "w")
    mv(tmp, path; force=true)
end
atomic_save(replace(OUT, r"\.csv$" => ".jls")) do io
    serialize(io, (; row, spectra, rails, result))
end
atomic_save(OUT) do io
    println(io, join(string.(keys(row)), ','))
    println(io, join((replace(string(x), ','=>';') for x in values(row)), ','))
end
println(row)
passed || error("bare six-R numerical diagnostic failed; output is not accepted")
