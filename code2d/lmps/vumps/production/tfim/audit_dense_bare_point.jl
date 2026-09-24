using LMPSVUMPS, Serialization, SHA, Test
include("bare_strip_checks.jl")
const D=parse(Int,ENV["TFIM_BARE_D"])
const CTM=parse(Int,ENV["TFIM_CTM_CHI"])
const ROOT=ENV["TFIM_BARE_POINT"]
const EXPECTED=ENV["TFIM_BARE_SOURCE_SHA"]
const CHI=parse(Int,ENV["TFIM_ROUTE_CHI"])
const H=parse(Float64,ENV["TFIM_BARE_H"])
const INPUT=joinpath(ROOT,"sixr","summary.jls")
function quality(r)
    records=(r.endpoint_audit...,r.mixed_corner_audit...)
    values=(maximum(x.residual for x in records),maximum(x.adjoint_gate for x in records),
        maximum(x.prefactor_alignment for x in records),maximum(x.prefactor_drift for x in records),
        maximum(x.prefactor_slope_match for x in records),r.scale_gate,r.map_residual,
        r.intercept_gate,r.local_identity_gate)
    limits=(1e-8,1e-11,5e-7,5e-7,5e-7,5e-8,1e-8,1e-10,1e-10)
    all(isfinite(v)&&0<=v<=lim for (v,lim) in zip(values,limits)) &&
        r.endpoint_count==6 && r.metric_count==3 &&
        all(isfinite,(r.tildeS,r.S2A,r.S2B,r.S3))
end
@testset "direct-only gates reject bad observables and closure" begin
    a=(residual=1e-12,adjoint_gate=1e-14,prefactor_alignment=1e-9,
        prefactor_drift=1e-9,prefactor_slope_match=1e-9)
    r=(endpoint_audit=(a,),mixed_corner_audit=(a,),scale_gate=1e-12,
        map_residual=1e-12,intercept_gate=1e-12,local_identity_gate=1e-12,
        endpoint_count=6,metric_count=3,tildeS=.01,S2A=.1,S2B=.1,S3=.1)
    @test quality(r)
    @test !quality(merge(r,(tildeS=NaN,)))
    @test !quality(merge(r,(map_residual=1e-4,)))
    @test !quality(merge(r,(endpoint_count=5,)))
    @test !quality(merge(r,(local_identity_gate=NaN,)))
end
p=open(deserialize,INPUT);r=p.result;old=p.row
old.source_state_sha256==EXPECTED && old.D==D && old.chi==CHI || error("source/dimension mismatch")
old.method=="bare_two_and_three_column_fixedpoints" && !old.endpoint_maps_used || error("not bare direct")
bytes2hex(open(sha256,old.cardinal_input))==old.cardinal_input_sha256 || error("changed cardinal input")
c=open(deserialize,old.cardinal_input)
c.row.passed && c.row.source_state_sha256==EXPECTED || error("failed reconstruction")
all(b.converged for b in c.boundaries) || error("boundary not converged")
c.original_north.residual_G<=1e-11 && c.original_north.residual_B<=1e-11 || error("fixedpoint residual")
ratios_M=BareStripChecks.spectrum(c.M.north)
xi_M=length(ratios_M)>1 && 0<ratios_M[2]<1 ? -inv(log(ratios_M[2])) : Inf
valid=quality(r) && isfinite(old.xiA) && old.xiA>0 && isfinite(xi_M) && xi_M>0
row=(status=valid ? "PASS_DIRECT_NUMERICAL" : "FAIL_DIRECT_NUMERICAL",D=D,h=H,chi=CHI,
    ctm_chi=CTM,method="bare_LR_LTR",endpoint_maps_used=false,source_state_sha256=EXPECTED,
    tildeS=r.tildeS,xi=old.xiA,xi_H=old.xiA,xi_M=xi_M,
    xi_kind="xi_H:reconstructed_LMPS_H=BG+;xi_M:VUMPS_north_MPS",
    cardinal_input=old.cardinal_input,cardinal_input_sha256=old.cardinal_input_sha256,S2A=r.S2A,S2B=r.S2B,S3=r.S3,
    checkpoint=INPUT,checkpoint_sha256=bytes2hex(open(sha256,INPUT)),
    auditor_sha256=bytes2hex(open(sha256,@__FILE__)),production_release=false)
open(joinpath(ROOT,"direct_audit.csv"),"w") do io
    println(io,join(string.(keys(row)),','))
    println(io,join(string.(values(row)),','))
end
println(row)
valid || error("direct numerical gates failed; failed row preserved")
