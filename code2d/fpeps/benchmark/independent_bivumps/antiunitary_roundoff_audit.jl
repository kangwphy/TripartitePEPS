# Frozen-state absolute symmetry defect near a stationary point.
include("antiunitary_newton_chart.jl")
haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
source,out=ARGS;ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
files=joinpath.(source,("boundary_1.jls","boundary_3.jls","independent_pair.jls","report.toml"))
n,s=deserialize.(files[1:2]);pair=deserialize(files[3]);maps=ba_pair_maps(n,s)
symmetry=at_tangent(pair.R,pair.L,maps);c=btm_chart(pair.R,pair.L,pair.transfer)
g=c.gradient;leak=norm(g-symmetry.Q*(symmetry.Q'*g))
states=[]
for (state,K) in ((pair.R,symmetry.sr),(pair.L,symmetry.sl))
    raw=(state.AL[1]+K.apply(state.AL[1]))/2
    raw=raw*bn_invsqrt(raw'*raw);candidate,_=bt_canonical(raw);push!(states,candidate)
end
rr,ll=states;cr=btm_chart(rr,ll,pair.transfer)
# Gauge-preserving canonicalization makes direct raw gradients comparable.
# Compare scalar quotients and full equations separately as invariants.
coefficient_change=max(norm(rr.AL[1]-pair.R.AL[1])/norm(pair.R.AL[1]),
    norm(ll.AL[1]-pair.L.AL[1])/norm(pair.L.AL[1]))
a=bv_audit(pair.R,pair.L,pair.transfer);b=bv_audit(rr,ll,pair.transfer)
check=Dict("complete"=>true,"diagnostic_only"=>true,"accepted_entropy"=>false,
    "gradient_norm"=>norm(g),"outside_absolute"=>leak,"outside_relative"=>leak/norm(g),
    "coefficient_change"=>coefficient_change,"quotient_relative_change"=>abs(cr.base.q/c.base.q-1),
    "original_full_residual"=>a.residual,"reprojected_full_residual"=>b.residual,
    "full_residual_change"=>abs(b.residual-a.residual),"tangent_symmetry"=>symmetry.report,
    "job_id"=>ENV["SLURM_JOB_ID"],"source_hashes"=>Dict(p=>bytes2hex(sha256(read(p))) for p in files),
    "script_sha256"=>bytes2hex(sha256(read(@__FILE__))))
# An absolute check near zero complements, rather than replaces, the relative
# nonstationary controls. This does not accept the still unconverged boundary.
check["passed"]=leak<1e-12 && coefficient_change<1e-12 && abs(b.residual-a.residual)<1e-12
bv_write(out,check);println(check);check["passed"] || error("absolute symmetry defect is not small")
