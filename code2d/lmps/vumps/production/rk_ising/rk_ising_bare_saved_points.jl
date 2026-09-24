# Literal bare LR/LTR test on the exact boundary checkpoints of the old plot.
# No grown-tail or endpoint-map constructor is called.
using LMPSVUMPS, LinearAlgebra, Serialization, SHA, Printf
include("bare_strip_checks.jl")
using .BareStripChecks

const BETAS = [0.44040840454546, 0.44053399683176, 0.44058175000000,
               0.44062772000000, 0.44068089000000, 0.44081811000000,
               0.44110207000000,
               # Extension: existing checkpoints, both phases and crossover.
               0.43056967239509, 0.43511621746807, 0.43762412004727,
               0.43900431317144, 0.43976293190036, 0.44017961786831,
               0.44065358000000, 0.44069993000000, 0.44076064000000,
               0.44152953848769, 0.44348866924400, 0.45003193991263]
const TASK = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const BETA = BETAS[TASK+1]
const CHI = 16
const TAG = replace(@sprintf("%.14f", BETA), "."=>"p")
const PKG = normpath(joinpath(@__DIR__, "..", ".."))
const SOURCE = joinpath(PKG,"data","production_v1")
const ROOT = ENV["RK_BARE_ROOT"]
const OUT = joinpath(ROOT,"points","beta_"*TAG)
mkpath(OUT)
const CHECKPOINT = joinpath(SOURCE,"boundaries","beta_"*TAG,"chi_16.jls")

function save_atomic(path, payload)
    tmp=path*".tmp."*ENV["SLURM_JOB_ID"]
    open(io->serialize(io,payload),tmp,"w")
    mv(tmp,path;force=true)
end
function csv_atomic(path, row)
    tmp=path*".tmp."*ENV["SLURM_JOB_ID"]
    open(tmp,"w") do io
        println(io,join(string.(keys(row)),','))
        println(io,join((replace(string(x),','=>';','\n'=>' ') for x in values(row)),','))
    end
    mv(tmp,path;force=true)
end
function firstrow(path)
    ls=readlines(path)
    Dict(split(ls[1],',').=>split(ls[2],',';keepempty=true))
end

function main()
    source_sha=bytes2hex(open(sha256,CHECKPOINT))
    saved=open(deserialize,CHECKPOINT)
    isapprox(saved.beta,BETA;atol=1e-14,rtol=0) || error("beta mismatch")
    saved.chi==CHI || error("chi mismatch")
    boundary=saved.boundary
    boundary.diagnostics.converged || error("original boundary is not converged")
    obs=firstrow(joinpath(SOURCE,"points","beta_"*TAG,"chi_16","observables.csv"))
    old=firstrow(joinpath(SOURCE,"points","beta_"*TAG,"chi_16","sixr.csv"))
    raw=LMPSVUMPS._host_array(boundary.state.AL[1])
    M=reshape(raw,CHI,boundary.source.D^2,CHI)
    a=LMPSVUMPS._dense_norm_site(boundary.source)
    symmetry_error=maximum(norm(a-permutedims(a,p))/norm(a) for p in
        ((2,3,4,1),(4,3,2,1),(1,4,3,2)))
    symmetry_error<1e-12 || error("cannot reuse this boundary: tensor symmetry failed")
    # Same saved branch, rotated to each cardinal side. No extra conjugation.
    cardinal=(north=M,east=M,west=permutedims(M,(3,2,1)),
              south=permutedims(M,(3,2,1)))
    common=(;beta=BETA,chi=CHI,source_checkpoint=CHECKPOINT,source_sha256=source_sha,
        method="bare_two_and_three_column_fixedpoints",endpoint_maps_used=false,
        center_matrices_inserted=false,boundary_reoptimized=false,
        symmetry_error,magnetization=parse(Float64,obs["magnetization"]),
        original_tildeS=parse(Float64,old["tildeS"]),
        original_S2A=parse(Float64,old["S2A"]),original_S2B=parse(Float64,old["S2B"]),
        original_S3=parse(Float64,old["S3"]))
    csv_atomic(joinpath(OUT,"source.csv"),common)
    println("stage=bare_fixedpoints beta=$BETA");flush(stdout)
    original=direct_original_lmps_fixedpoints(cardinal.west,cardinal.east,a;
        nmax=30000,tol=1e-11,initial_G=Matrix{ComplexF64}(I,CHI,CHI),
        initial_B=M,solver=:synchronized_power)
    save_atomic(joinpath(OUT,"bare_fixedpoints.jls"),(;common,cardinal,a,original))
    println("stage=strip_checks beta=$BETA");flush(stdout)
    reference=BareStripChecks.strip(M,cardinal.south,a)
    reconstructed=BareStripChecks.strip(original.H,cardinal.south,a)
    fidelity=BareStripChecks.fidelity(original.H,M)
    logz_exact=-BETA*onsager_f(BETA)
    prereq=original.residual_G<=1e-11 && original.residual_B<=1e-11 &&
        abs(reconstructed.logz-reference.logz)<=1e-7 && abs(fidelity-1)<=1e-5
    pre=(;common...,residual_G=original.residual_G,residual_B=original.residual_B,
        metric_condition=original.metric_condition,metric_dropped=original.metric_dropped,
        logz_exact,logz_reference=reference.logz,logz_bare=reconstructed.logz,
        logz_bare_minus_reference=reconstructed.logz-reference.logz,
        logz_bare_minus_exact=reconstructed.logz-logz_exact,fidelity,
        G_identity_residual=norm(original.G-tr(original.G)/CHI*I)/norm(original.G),
        reconstruction_passed=prereq)
    csv_atomic(joinpath(OUT,"reconstruction.csv"),pre)
    original.residual_G<=1e-11 && original.residual_B<=1e-11 ||
        error("bare fixedpoints failed; diagnostic preserved")
    rails=LMPSVUMPS.direct_cardinal_regional_rails(original,
        cardinal.west,cardinal.east,cardinal.south;rtol=1e-12)
    println("stage=sixr beta=$BETA reconstruction_passed=$prereq");flush(stdout)
    measured=@timed m2_contract_regional_six_R(rails.rails,boundary.source.D;
        nmax=8000,tol=1e-10,scale_tolerance=5e-8,
        far_boundary_tolerance=5e-7,far_boundary_tail_tolerance=5e-7,verbose=true)
    r=measured.value
    audits=(r.endpoint_audit...,r.mixed_corner_audit...)
    residual=maximum(x.residual for x in audits)
    adjoint=maximum(x.adjoint_gate for x in audits)
    alignment=maximum(x.prefactor_alignment for x in audits)
    drift=maximum(x.prefactor_drift for x in audits)
    slope=maximum(x.prefactor_slope_match for x in audits)
    passed=prereq && all(isfinite,(residual,adjoint,alignment,drift,slope,
        r.tildeS,r.S2A,r.S2B,r.S3,r.scale_gate,r.map_residual)) &&
        residual<=1e-8 && adjoint<=1e-11 && alignment<=5e-7 && drift<=5e-7 &&
        slope<=5e-7 && r.scale_gate<=5e-8 && r.map_residual<=1e-8 &&
        r.endpoint_count==6 && r.metric_count==3
    row=(;pre...,status=passed ? "PASS_NUMERICAL_DIAGNOSTIC" : "FAIL_NUMERICAL_DIAGNOSTIC",
        production_release=false,tildeS=r.tildeS,S2A=r.S2A,S2B=r.S2B,S2C=r.S2C,S3=r.S3,
        delta_tildeS=r.tildeS-common.original_tildeS,max_residual=residual,
        max_adjoint_gate=adjoint,alignment,drift,slope,scale_gate=r.scale_gate,
        map_residual=r.map_residual,endpoint_count=r.endpoint_count,metric_count=r.metric_count,
        seconds=measured.time)
    save_atomic(joinpath(OUT,"sixr.jls"),(;row,result=r,rails))
    csv_atomic(joinpath(OUT,"comparison.csv"),row)
    println(row);flush(stdout)
    passed || error("numerical diagnostic failed; values retained but not accepted")
end
try
    main()
catch err
    csv_atomic(joinpath(OUT,"failure.csv"),(;beta=BETA,chi=CHI,
        status="FAILED",error=sprint(showerror,err)))
    rethrow()
end
