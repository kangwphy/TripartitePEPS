using LMPSVUMPS, CUDA, Adapt, ITensors, LinearAlgebra, Serialization, SHA
const BACKEND=Symbol(get(ENV,"TFIM_BACKEND","cpu"))
BACKEND in (:cpu,:cuda) || error("TFIM_BACKEND must be cpu or cuda")
if BACKEND===:cuda
    CUDA.functional() || error("CUDA unavailable; no CPU fallback")
    CUDA.allowscalar(false)
    isdefined(@__MODULE__,:BareCUDA) || include(joinpath(@__DIR__,"../../src/algorithms/bare_cuda/BareCUDA.jl"))
else
    include(joinpath(@__DIR__,"sliced_closure_runtime.jl"))
end
const C=BACKEND===:cuda ? BareCUDA : LMPSVUMPS
boundary_array(b)=BACKEND===:cuda ? C.boundary_array(b) : C._quantum_boundary_tensor(b)
transfer_array(b)=BACKEND===:cuda ? C.transfer_array(b) : C._quantum_transfer_site(b)
device(x)=BACKEND===:cuda ? CUDA.CuArray(x) : Array(x)
host(x::AbstractArray)=Array(x)
host(x::NamedTuple)=map(host,x)
host(x)=x
const source=ENV["TFIM_SELECTED_STATE"]
const out=ENV["TFIM_BARE_OUTPUT"]
const chi=parse(Int,ENV["TFIM_ROUTE_CHI"])
const payload=deserialize(source)
const gs=payload.selected_groundstate
const source_sha=bytes2hex(open(sha256,source))
haskey(ENV,"TFIM_BARE_SOURCE_SHA") && ENV["TFIM_BARE_SOURCE_SHA"]!=source_sha && error("source hash mismatch")
const phases=get(ENV,"TFIM_BARE_WARMUP","false")=="true" ? ("warmup","timed") : ("measurement",)
any(isfile(joinpath(out,p*ext)) for p in phases for ext in (".jls",".csv")) && error("output exists; select a fresh directory")
const code_paths=[@__FILE__,joinpath(@__DIR__,"../../Manifest.toml")]
append!(code_paths,[joinpath(@__DIR__,"../../src/algorithms",n) for n in
    ("direct_route.jl","sixr_closure.jl","quantum_boundary.jl","quantum_sixr.jl",
     "bare_cuda/BareCUDA.jl","bare_cuda/bridge_cuda.jl","bare_cuda/direct_cuda.jl","bare_cuda/sixr_cuda.jl")])
const code_sha256=join((basename(p)*":"*bytes2hex(open(sha256,p)) for p in code_paths),';')
mkpath(out)
function save(name,x)
    path=joinpath(out,name*".jls")
    tmp=path*".tmp"
    open(io->serialize(io,x),tmp,"w"); mv(tmp,path;force=false)
end
function timed(f,label)
    BACKEND===:cuda && CUDA.synchronize(); t=time(); x=f(); BACKEND===:cuda && CUDA.synchronize(); dt=time()-t
    println("STAGE $label seconds=$dt"); flush(stdout)
    x,dt
end
# Same three seeds and acceptance rules as solve_oriented_bare_cardinal.jl.
const boundary_records=NamedTuple[]
function boundaries()
    map(((:north,10001),(:south,20001),(:east,30001),(:west,40001))) do (side,offset)
        trials=map(0:2) do trial
            b=solve_quantum_vumps_boundary(gs.peps,chi;side,backend=BACKEND,
                seed=2026370000+100chi+offset+trial,tolerance=1e-8,
                diagnostic_tolerance=1e-6,max_iterations=4000,
                verbosity=0,accept_unconverged=true)
            (b.state.AL[1].data isa CuArray)==(BACKEND===:cuda) || error("boundary backend mismatch")
            println("BOUNDARY side=$side trial=$trial converged=$(b.converged) iterations=$(b.iterations)");flush(stdout)
            b
        end
        # Store diagnostics without retaining backend-dependent TensorMap objects.
        push!(boundary_records,(;side,trials=[(;converged=b.converged,iterations=b.iterations,
            row_eigenvalue=b.row_eigenvalue,galerkin_residual=b.galerkin_residual,
            left_environment_residual=b.left_environment_residual,
            right_environment_residual=b.right_environment_residual,
            center_residual=b.center_residual) for b in trials]))
        accepted=filter(b->b.converged,trials)
        isempty(accepted) && error("no converged $side boundary")
        accepted[argmax(abs.([b.row_eigenvalue for b in accepted]))]
    end |> x->NamedTuple{(:north,:south,:east,:west)}(x)
end
function run(phase)
    empty!(boundary_records)
    b,tboundary=timed(boundaries,"$phase boundary")
    raw=map(boundary_array,b)
    M=(north=raw.north,south=permutedims(raw.south,(3,2,1)),east=raw.east,west=permutedims(raw.west,(3,2,1)))
    a=transfer_array(b.north)
    orig,tdirect=timed("$phase direct") do
        C.direct_original_lmps_fixedpoints(M.west,M.east,a;nmax=24000,tol=1e-11,
            initial_G=device(Matrix{ComplexF64}(I,chi,chi)),initial_B=M.north)
    end
    save("$(phase)_direct",(;M=host(M),a=host(a),orig=host(orig),source_sha))
    orig.residual_G<=1e-11 && orig.residual_B<=1e-11 || error("direct residual gate")
    rails=C.direct_cardinal_regional_rails(orig,M.west,M.east,M.south;rtol=1e-12)
    result,tsixr=timed("$phase sixr") do
        if BACKEND===:cuda
            C.m2_contract_regional_six_R(rails.rails,isqrt(size(a,1));nmax=8000,tol=1e-10,
                scale_tolerance=5e-8,far_boundary_tolerance=5e-7,
                far_boundary_tail_tolerance=5e-7,verbose=true)
        else
            sliced_contract_regional_six_R(rails.rails,isqrt(size(a,1));nmax=8000,tol=1e-10,
                scale_tolerance=5e-8,far_boundary_tolerance=5e-7,
                far_boundary_tail_tolerance=5e-7,verbose=true)
        end
    end
    # This is a speed experiment. An independent closed-strip/fidelity audit
    # is still required for scientific release; never overwrite current data.
    records=(result.endpoint_audit...,result.mixed_corner_audit...)
    passed=all(isfinite,(result.tildeS,result.S2A,result.S2B,result.S2C,result.S3)) &&
        all(x->x.residual<=1e-8 && x.adjoint_gate<=1e-11 &&
            x.prefactor_alignment<=5e-7 && x.prefactor_drift<=5e-7 &&
            x.prefactor_slope_match<=5e-7,records) && result.scale_gate<=5e-8
    row=(;phase,D=isqrt(size(a,1)),chi,source,source_sha,source_converged=gs.converged,
        code_sha256,status=passed ? "PASS_NUMERICAL_DIAGNOSTIC" : "FAIL_NUMERICAL_DIAGNOSTIC",
        boundary_seconds=tboundary,direct_seconds=tdirect,sixr_seconds=tsixr,
        total_seconds=tboundary+tdirect+tsixr,tildeS=result.tildeS,
        residual_G=orig.residual_G,residual_B=orig.residual_B,
        route="bare_direct",backend=String(BACKEND),production_release=false)
    save(phase,(;row,result,boundary_trials=copy(boundary_records),source_config=hasproperty(payload,:config) ? payload.config : nothing))
    open(joinpath(out,"$phase.csv"),"w") do io
        println(io,join(string.(keys(row)),','));println(io,join(string.(values(row)),','))
    end
    println(row);flush(stdout)
    passed || error("numerical diagnostic failed; output saved without production release")
end
println("INPUT sha=$source_sha chi=$chi source_converged=$(gs.converged)");flush(stdout)
for phase in phases
    run(phase)
end
