using LMPSVUMPS, LinearAlgebra, Serialization, SHA
include("bare_strip_checks.jl")
using .BareStripChecks
const STATE=abspath(ENV["TFIM_SELECTED_STATE"])
const OUT=abspath(ENV["TFIM_BARE_POINT"])
const CHI=parse(Int,ENV["TFIM_ROUTE_CHI"])
const EXPECTED=ENV["TFIM_BARE_SOURCE_SHA"]
bytes2hex(open(sha256,STATE))==EXPECTED || error("GS hash mismatch")
const payload=open(deserialize,STATE)
const peps=payload.selected_groundstate.peps
function save_jls(path,data)
    tmp=path*".tmp."*ENV["SLURM_JOB_ID"]
    open(io->serialize(io,data),tmp,"w")
    mv(tmp,path;force=true)
end
function solve_side(side,offset)
    trials=Any[]
    for trial in 0:2
        println("stage=boundary side=$side trial=$trial chi=$CHI");flush(stdout)
        push!(trials,solve_quantum_vumps_boundary(peps,CHI;side,
            seed=2026370000+100*CHI+offset+trial,tolerance=1e-8,
            diagnostic_tolerance=1e-6,max_iterations=4000,verbosity=0,
            accept_unconverged=true))
        save_jls(joinpath(OUT,"boundary_$(side)_trials.jls"),
            (;trials,source_state_sha256=EXPECTED,side,chi=CHI))
    end
    accepted=filter(b->b.converged,trials)
    isempty(accepted) && error("no converged $side candidate; trials saved")
    accepted[argmax(abs.([b.row_eigenvalue for b in accepted]))]
end
const boundaries=(north=solve_side(:north,10_001),south=solve_side(:south,20_001),
    east=solve_side(:east,30_001),west=solve_side(:west,40_001))
const raw=map(b->LMPSVUMPS._quantum_boundary_tensor(b),boundaries)
const M=(north=raw.north,south=permutedims(raw.south,(3,2,1)),
    east=raw.east,west=permutedims(raw.west,(3,2,1)))
const a=LMPSVUMPS._quantum_transfer_site(boundaries.north)
save_jls(joinpath(OUT,"cardinal_boundaries.jls"),(;boundaries,raw,M,a,source_state_sha256=EXPECTED))
println("stage=original oriented LR/LTR fixed points");flush(stdout)
const original_north=direct_original_lmps_fixedpoints(M.west,M.east,a;
    nmax=24000,tol=1e-11,initial_G=Matrix{ComplexF64}(I,CHI,CHI),
    initial_B=M.north,solver=:synchronized_power)
save_jls(joinpath(OUT,"original_fixedpoints.jls"),(;original_north,source_state_sha256=EXPECTED))
println("stage=independent closed-strip checks");flush(stdout)
const ref=BareStripChecks.strip(M.north,M.south,a)
const direct=BareStripChecks.strip(original_north.H,M.south,a)
const fidelity=BareStripChecks.fidelity(original_north.H,M.north)
const spectral_error=maximum(abs.(BareStripChecks.spectrum(original_north.H)-BareStripChecks.spectrum(M.north)))
const passed=abs(direct.logz-ref.logz)<=1e-7 && abs(fidelity-1)<=1e-5 &&
    original_north.residual_G<=1e-11 && original_north.residual_B<=1e-11
const row=(;construction="axis_aligned_bare_LR_LTR",chi=CHI,
    source_state_sha256=EXPECTED,source=STATE,endpoint_maps_used=false,
    script_sha256=bytes2hex(open(sha256,@__FILE__)),
    strip_script_sha256=bytes2hex(open(sha256,joinpath(@__DIR__, "bare_strip_checks.jl"))),
    west_reflected=true,south_reflected=true,logz_reference=ref.logz,
    logz_direct=direct.logz,logz_error=direct.logz-ref.logz,
    native_logz=real(log(abs(original_north.lambda_B/original_north.lambda_G))),
    fidelity,spectral_error,metric_condition=original_north.metric_condition,
    metric_dropped=original_north.metric_dropped,
    residual_G=original_north.residual_G,residual_B=original_north.residual_B,
    passed,production_release=false)
save_jls(joinpath(OUT,"closure.jls"),(;M,a,original_north,boundaries,row,ref,direct))
open(joinpath(OUT,"closure.csv"),"w") do io
    println(io,join(string.(keys(row)),','))
    println(io,join((replace(string(x),','=>';') for x in values(row)),','))
end
println(row);flush(stdout)
passed || error("bare reconstruction failed; all inputs and failed result retained")
