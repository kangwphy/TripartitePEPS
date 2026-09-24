# Fixed-input bare/direct scan. The chi4 core and production contraction are
# unchanged. Full signed cycle sums are used at every depth; character is only
# an audit. The chi^8 unfactorized graph was separately checked at chi4.
include("bare_direct_core.jl")
include("character_lmps_measurement.jl")

function write_report(path,report)
    open(path*".tmp","w") do io; TOML.print(io,report); end
    mv(path*".tmp",path;force=true)
end

function scan_bare(source,output;maxdepth=1024)
    BLAS.set_num_threads(parse(Int,get(ENV,"SLURM_CPUS_PER_TASK","2")))
    mkpath(output)
    isfile(joinpath(output,"bare_result.toml")) && error("refusing to overwrite a completed point")
    diagnostic=isfile(joinpath(source,"diagnostic_north.jls"))
    names=diagnostic ? ("diagnostic_north.jls","diagnostic_south.jls","diagnostic_rotated.jls") :
                      ("boundary_1.jls","boundary_3.jls","rotated_boundary.jls")
    files=[joinpath(source,f) for f in names]
    hashes=Dict(f=>bytes2hex(sha256(read(f))) for f in files)
    n,s,rn=deserialize.(files); chi=sum(n.chi)
    chi in (2,4,6,8,10,12,16) || error("this bounded scan supports selected even chi<=16 only")
    all(b->b.chi==n.chi,(s,rn)) || error("inconsistent chi sectors")
    n.peps[1]≈ksvc_ipeps()[1] || error("scan requires native w=1 KSVC tensor")
    rn.peps[1]≈rotl90(n.peps)[1] && rn.direction==1 || error("incorrect perpendicular input")
    native=[Dict("direction"=>name,"converged"=>b.converged,"galerkin"=>b.galerkin,
                 "left_residual"=>b.left_residual,"right_residual"=>b.right_residual,
                 "center_residual"=>b.center_residual)
            for (name,b) in (("north",n),("south",s),("rotated_north",rn))]
    common=Dict{String,Any}("chi"=>chi,"chi_sectors"=>collect(n.chi),"w"=>1.0,
        "method"=>"bare_two_and_three_column_fixedpoints","source"=>source,"source_hashes"=>hashes,
        "native_boundaries"=>native,"input_native_converged"=>all(b->b.converged,(n,s,rn)),
        "grown_tensor_used"=>false,"endpoint_maps_used"=>false,"character_used_for_entropy"=>false,
        "reconstruction_checked"=>false,"physical_benchmark_certified"=>false,"accepted_entropy"=>false)
    write_report(joinpath(output,"input.toml"),common)
    println("bare chi=",chi," inputs ",native);flush(stdout)
    objects=bare_direct_objects(n,s)
    serialize(joinpath(output,"bare_objects.jls"),objects)
    write_report(joinpath(output,"bare_fixedpoints.toml"),objects.report)
    xi_self=boundary_even_correlation_length(n)
    write_report(joinpath(output,"xi_self.toml"),Dict("xi_boundary_even"=>xi_self,
        "recomputed"=>true,"source_hashes"=>hashes))
    println("bare chi=",chi," fixed points residual=",(objects.g.report["residual"],objects.b.report["residual"]),
        " condition=",objects.report["metric_condition"]," xi_self=",xi_self);flush(stdout)
    seams=bare_cardinal_seams(objects,rn)
    caps=solve_seam_caps(seams;dense_threshold=512)
    serialize(joinpath(output,"bare_geometry.jls"),(;seams,caps,objects))
    fingerprint=bytes2hex(sha256(read(joinpath(output,"bare_geometry.jls"))))
    common["geometry_sha256"]=fingerprint
    common["xi_boundary_even"]=xi_self
    cap_report=Dict(string(k)=>Dict("gap"=>v.gap,"residual"=>v.residual,"isolated"=>v.isolated)
                    for (k,v) in pairs(caps.diagnostics))
    write_report(joinpath(output,"seam_caps.toml"),cap_report)
    # Compare full signed expansion against exact character at shallow depths.
    # This is additional to, and does not claim to repeat, the chi4 full graph audit.
    audits=Dict{String,Any}[]
    for depth in (1,2)
        full=finite_lmps_sectors(seams,caps.caps;depth)
        short=character_lmps_sectors(seams,caps.caps;depth)
        for name in keys(full)
            err=FermionicPEPS._sector_relative_error(getproperty(full,name),getproperty(short,name))
            push!(audits,Dict("depth"=>depth,"sector"=>string(name),"relative_error"=>err))
        end
        write_report(joinpath(output,"signed_audit.toml"),Dict("complete"=>depth==2,
            "checks"=>audits,"character_passed"=>all(r->r["relative_error"]<1e-10,audits),
            "full_cycle_repeated_at_this_chi"=>false,"full_cycle_reference"=>"data/bare_direct_20260917/chi4/bare_signed_audit.toml"))
        println("bare chi=",chi," signed audit depth=",depth," max=",maximum(r["relative_error"] for r in audits));flush(stdout)
    end
    common["character_audit_passed"]=all(r->r["relative_error"]<1e-10,audits)
    samples=Dict{String,Any}[];cache=Dict{Any,Any}();sectors=Dict{Int,Any}()
    result=nothing;depth=4;last_failure="fewer than three samples"
    while depth<=maxdepth
        z=finite_lmps_sectors(seams,caps.caps;depth,propagation_cache=cache)
        serialize(joinpath(output,"bare_sectors_depth$depth.jls"),z)
        sectors[depth]=z
        all(v->!iszero(v.value)&&isfinite(abs(v.value))&&isfinite(v.logscale),values(z)) || error("zero/nonfinite replica sector")
        logs=[log(abs(v.value))+v.logscale for v in values(z)]
        phases=[angle(v.value) for v in values(z)]
        phase_error=maximum(abs.([exp.(im.*(phases[2:4].-2phases[1])).-1;exp(im*(phases[5]-4phases[1]))-1]))
        row=Dict{String,Any}("depth"=>depth,"stilde_real_diagnostic"=>sum(logs[2:4])-logs[5]-2logs[1],
            "log_magnitudes"=>logs,"raw_phases"=>phases,"phase_error"=>phase_error,
            "phase_passed"=>phase_error<=1e-8,
            "endpoint_residual"=>maximum(FermionicPEPS._active_endpoint_residual(v) for v in values(z)),
            "max_cancellation"=>maximum(v.cancellation_condition for v in values(z)),
            "full_signed"=>true,"accepted_entropy"=>false)
        push!(samples,row)
        write_report(joinpath(output,"bare_depth$depth.toml"),row)
        println("bare chi=",chi," depth=",depth," Sdiag=",row["stilde_real_diagnostic"],
            " phase=",phase_error," endpoint=",row["endpoint_residual"]);flush(stdout)
        if length(samples)>=3
            ds=Tuple(r["depth"] for r in samples[end-2:end])
            builder=(a,b;depth,kwargs...)->sectors[depth]
            # Use the unchanged production phase/cancellation/length gates.
            result=try
                measure_lmps_stilde(seams,caps.caps;depths=ds,phase_tolerance=1e-8,
                    accept_unconverged=true,sector_builder=builder)
            catch exception
                exception isa ErrorException || rethrow()
                last_failure=sprint(showerror,exception)
                println("bare gate: ",last_failure);flush(stdout)
                nothing
            end
            result===nothing || (last_failure=result.converged ? "" : "length/endpoint gate failed")
        end
        passed=result!==nothing && result.converged
        report=merge(common,Dict("complete"=>passed || depth==maxdepth,"contraction_passed"=>passed,
            "phase_tolerance"=>1e-8,"length_tolerance"=>1e-7,"maxdepth"=>maxdepth,
            "samples"=>samples,"last_gate_failure"=>last_failure,
            "stilde_real_diagnostic"=>row["stilde_real_diagnostic"]))
        if result!==nothing
            report["length_drift"]=result.diagnostics.drift
            report["linear_tail_residual"]=result.diagnostics.linear_residual
            report["endpoint_residual"]=result.diagnostics.endpoint_residual
            passed && (report["stilde"]=result.stilde)
        end
        write_report(joinpath(output,"progress.toml"),report)
        if passed || depth==maxdepth
            all(bytes2hex(sha256(read(f)))==h for (f,h) in hashes) || error("input changed")
            write_report(joinpath(output,"bare_result.toml"),report)
            println("bare chi=",chi," complete; contraction_passed=",passed);flush(stdout)
            return report
        end
        # Retain only the three depth results needed for the unchanged gate.
        length(sectors)>3 && delete!(sectors,minimum(keys(sectors)))
        depth*=2
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    try
        scan_bare(ARGS[1],ARGS[2])
    catch exception
        mkpath(ARGS[2])
        write_report(joinpath(ARGS[2],"failure.toml"),Dict("complete"=>false,"accepted_entropy"=>false,
            "error"=>sprint(showerror,exception),"script_sha256"=>bytes2hex(sha256(read(@__FILE__)))))
        rethrow()
    end
end
