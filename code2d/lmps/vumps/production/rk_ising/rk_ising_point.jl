"""Unified RK-Ising point driver. Import is side-effect free; main requires preempt.

`use_z2` selects boundary optimization, independently of `use_cap_fix`, which
selects the nine seam fixed-point solves. Both default to false. All four
combinations use literal bare/direct LR/LTR; there is no endpoint-map fallback.
"""
module RKIsingPoint

using LMPSVUMPS, LinearAlgebra, Random, Serialization, SHA, Printf, Dates, TOML
using ITensors
include("bare_strip_checks.jl")
include("../../src/models/rk_ising_z2.jl")
include("../../src/algorithms/sixr_modes.jl")
using .BareStripChecks
using .SixRModes: sixr_options

export mode_config, boundary_solver, sixr_options, spectrum_lengths, main

function mode_config(config)
    z2, fix = get(config,"use_z2",false), get(config,"use_cap_fix",false)
    z2 isa Bool && fix isa Bool || throw(ArgumentError("use_z2/use_cap_fix must be booleans"))
    expected = z2 ? (fix ? "Z2_fix" : "Z2") : (fix ? "no_constraint_fix" : "no_constraint")
    declared=String(get(config,"series_label",expected))
    declared==expected || throw(ArgumentError("series_label=$declared conflicts with use_z2=$z2/use_cap_fix=$fix; expected $expected"))
    (;use_z2=z2,use_cap_fix=fix,series_label=declared)
end
dense_boundary(beta,chi;kwargs...) = solve_vumps_boundary(rk_ising(beta),chi;backend=:cpu,kwargs...)
boundary_solver(config) = mode_config(config).use_z2 ? RKIsingZ2.solve_rk_ising_z2 : dense_boundary

"Full-spectrum and nonperipheral lengths are distinct; a cat mode stays Inf."
function spectrum_lengths(spectrum; peripheral_tolerance=1e-12)
    magnitudes = sort(abs.(spectrum);rev=true)
    length(magnitudes)>=2 || error("fewer than two transfer eigenvalues")
    isfinite(magnitudes[1]) && magnitudes[1]>0 || error("invalid leading transfer magnitude")
    ratios=magnitudes./magnitudes[1]
    ratio=ratios[2]
    xi = abs(ratio-1)<=peripheral_tolerance ? Inf : ratio==0 ? 0.0 : 0<ratio<1 ? -1/log(ratio) : NaN
    isnan(xi) && error("invalid transfer ratio $ratio")
    count_peripheral=count(r->isfinite(r)&&abs(r-1)<=peripheral_tolerance,ratios)
    sub=findfirst(r->isfinite(r)&&0<r<1-peripheral_tolerance,ratios)
    (;xi,xi_full=xi,xi_full_peripheral_count=count_peripheral,
      xi_full_degenerate=count_peripheral>=2,xi_peripheral_tolerance=peripheral_tolerance,
      xi_subleading=sub===nothing ? NaN : -1/log(ratios[sub]),
      xi_subleading_rank=sub===nothing ? 0 : sub,
      xi_subleading_ratio=sub===nothing ? NaN : ratios[sub],
      lambda1_abs=magnitudes[1],lambda2_abs=magnitudes[2],lambda2_over_lambda1=ratio)
end

function z2_observables(boundary,beta,z2; spectrum_num=16,spectrum_tol=1e-11)
    # Preserve the exported graded environments and their invariant-sector
    # selection; do not independently rebuild nonunique dense environments.
    spectrum=ComplexF64.(LMPSVUMPS.MPSKit.transfer_spectrum(boundary.state;
        num_vals=min(spectrum_num,size(LMPSVUMPS._host_array(boundary.state.C[1]),1)^2),tol=spectrum_tol))
    sort!(spectrum;by=abs,rev=true)
    lengths=spectrum_lengths(spectrum)
    singular_values=Float64.(abs.(svdvals(LMPSVUMPS._host_array(boundary.state.C[1]))))
    probabilities=abs2.(singular_values); probabilities./=sum(probabilities)
    entropy_vn=-sum(p>eps(Float64) ? p*log(p) : 0.0 for p in probabilities)
    renyi(alpha)=log(sum(probabilities.^alpha))/(1-alpha)
    m=vumps_ising_magnetization(boundary)
    exact_m=ising_exact_magnetization(beta)
    f=vumps_free_energy(boundary,beta); exact_f=onsager_f(beta)
    merge(lengths,(;transfer_spectrum=spectrum,schmidt_singular_values=singular_values,
        schmidt_probabilities=probabilities,entropy_vn,entropy_renyi2=renyi(2),
        entropy_renyi3=renyi(3),entropy_renyi4=renyi(4),entropy_infinity=-log(maximum(probabilities)),
        schmidt_gap=probabilities[1]-probabilities[2],effective_rank=exp(entropy_vn),
        magnetization=m,magnetization_abs=abs(m),exact_magnetization=exact_m,
        magnetization_abs_error=abs(abs(m)-exact_m),magnetization_symmetric_reference=0.0,
        exact_spin_xi=ising_exact_correlation_length(beta),row_logz=vumps_logz(boundary),
        free_energy=f,exact_free_energy=exact_f,free_energy_abs_error=abs(f-exact_f),
        free_energy_rel_error=abs(f-exact_f)/abs(exact_f),
        xi_even=z2.sector_spectra.xi_even,xi_odd=z2.sector_spectra.xi_odd,
        xi_all=z2.sector_spectra.xi_all))
end

sha(path)=bytes2hex(open(sha256,path))
timestamp()=Dates.format(Dates.now(Dates.UTC),dateformat"yyyy-mm-ddTHH:MM:SSZ")
cell(x::Bool)=x ? "true" : "false"
cell(x::Integer)=string(x)
cell(x::Real)=@sprintf("%.17g",Float64(x))
function cell(x)
    s=string(x)
    occursin(r"[,\"\n\r]",s) ? "\""*replace(s,"\""=>"\"\"")*"\"" : s
end
function atomic(writer,path)
    mkpath(dirname(path))
    tmp=path*".tmp."*get(ENV,"SLURM_JOB_ID",string(getpid()))
    open(writer,tmp,"w"); mv(tmp,path;force=true)
end
save(path,x)=atomic(io->serialize(io,x),path)
function write_rows(path,rows)
    isempty(rows) && return
    fields=sort!(unique(vcat([collect(keys(r)) for r in rows]...)))
    atomic(path) do io
        println(io,join(cell.(fields),','))
        for r in rows
            println(io,join((cell(get(r,k,"")) for k in fields),','))
        end
    end
end
"CSV with quoted commas and doubled quotes; manifests deliberately have no multiline fields."
function csv_cells(line)
    values=String[]; current=IOBuffer(); quoted=false; chars=collect(chomp(line)); i=1
    while i<=length(chars)
        c=chars[i]
        if c=='"'
            if quoted && i<length(chars) && chars[i+1]=='"'
                print(current,'"'); i+=1
            else
                quoted=!quoted
            end
        elseif c==',' && !quoted
            push!(values,String(take!(current)))
        else
            print(current,c)
        end
        i+=1
    end
    quoted && error("unclosed CSV quote")
    push!(values,String(take!(current))); values
end
function read_csv(path)
    lines=readlines(path); isempty(lines) && return Dict{String,String}[]
    header=csv_cells(lines[1])
    [Dict(header.=>csv_cells(line)) for line in lines[2:end] if !isempty(strip(line))]
end
function export_arrays(path,arrays)
    atomic(path) do io
        println(io,"tensor,rank,i1,i2,i3,i4,i5,real,imag")
        for (name,A) in arrays, I in CartesianIndices(A)
            inds=ntuple(j->j<=ndims(A) ? I[j] : 0,5); z=A[I]
            println(io,join((name,string(ndims(A)),string.(inds)...,cell(real(z)),cell(imag(z))),','))
        end
    end
end

function direct_symmetry(original,Q)
    covariance(A)=sqrt(sum(norm(A[:,p,:]-Q*A[:,5-p,:]*Q)^2 for p in 1:4))/max(norm(A),eps(Float64))
    (;G=norm(original.G-Q*original.G*Q)/max(norm(original.G),eps(Float64)),
      B=covariance(original.B),H=covariance(original.H))
end

"Recover finite diagnostic scalars if a post-solve closure gate rejects the nine saved seams."
function retained_closure(rails,D,records)
    length(records)==9 || error("all nine finite seams required for diagnostic closure")
    L=LMPSVUMPS
    specs1=L.m2_specs(rails,D,1,[1],[1],[1])
    specs4=L.m2_specs(rails,D,4,L.M2_SIGMA_A,L.M2_SIGMA_B,L.M2_SIGMA_C)
    info1=[r.solution for r in records[1:3]]; info4=[r.solution for r in records[4:9]]
    R1=[ITensor(e.R,s.lefts...) for (e,s) in zip(info1,specs1)]
    R4=[ITensor(e.R,s.lefts...) for (e,s) in zip(info4,specs4)]
    verts(specs)=[(s.Xn,s.Yn,s.cyc) for s in specs]
    v1,v4=verts(specs1),verts(specs4)
    L.m2_topology_audit(R4,v4)
    metrics=Dict(s.seam=>(;R=R1[j],spec=s,info=info1[j]) for (j,s) in enumerate(specs1))
    sources=Dict(name=>begin
        j=L.m2_vertex_index(v4,name); (;R=R4[j],spec=specs4[j],info=info4[j])
    end for name in propertynames(L.M2_DESCRIPTORS))
    O1=log(abs(L.m2_close_small(R1,v1)[1])); O4=log(abs(L.m2_close_octahedron(R4,v4)[1]))
    I1=O1+sum(e.log_prefactor for e in info1); I4=O4+sum(e.log_prefactor for e in info4)
    O2=Dict{String,Float64}(); I2=Dict{String,Float64}(); mapres=0.0
    for (name,k,pA,pB,pC) in L.SIXR_O2_SECTORS
        specs=L.m2_specs(rails,D,k,pA,pB,pC)
        z,res,_=L.m2_build_O2(name,specs,sources,metrics)
        O2[name]=log(abs(z)); I2[name]=O2[name]+L.m2_O2_asymptotic_data(name,specs,sources,metrics).intercept
        mapres=max(mapres,res)
    end
    S2A=-(I2["O2A"]-I2["O2C"]); S2B=-(I2["O2B"]-I2["O2C"]); S2C=0.0
    S3=-0.5*(I4+2I1-3I2["O2C"])
    tildeS=-(O4+2O1-O2["O2A"]-O2["O2B"]-O2["O2C"])
    (;S2A,S2B,S2C,S3,tildeS,mirror=abs(S2A-S2B),map_residual=mapres,
      scale_gate=NaN,endpoint_count=6,metric_count=3,diagnostic_closure_only=true)
end

function run_point(root,task)
    start=time(); job=ENV["SLURM_JOB_ID"]
    config=TOML.parsefile(joinpath(root,"config.toml")); modes=mode_config(config)
    config_sha=sha(joinpath(root,"config.toml")); code_sha=sha(joinpath(root,"source_hashes.json"))
    grid=read_csv(joinpath(root,"requested_grid.csv"))
    0<=task<length(grid) || error("task out of bounds")
    spec=grid[task+1]; id=spec["point_id"]
    beta=parse(Float64,spec["beta"]); chi=parse(Int,spec["chi"]); seed=parse(Int,spec["seed"])
    path(kind,suffix=".jls")=joinpath(root,"checkpoints",id*"_"*kind*suffix)
    pointpath(suffix=".csv")=joinpath(root,"points",id*suffix)
    wrap(;kwargs...)=(;point_id=id,beta,chi,seed,code_sha256=code_sha,config_sha256=config_sha,
        use_z2=modes.use_z2,use_cap_fix=modes.use_cap_fix,series_label=modes.series_label,
        julia_version=string(VERSION),job_id=job,kwargs...)
    row=Dict{String,Any}(spec)
    merge!(row,Dict(string(k)=>v for (k,v) in pairs(modes)))
    merge!(row,Dict("code_sha256"=>code_sha,"config_sha256"=>config_sha,"job_id"=>job,
        "julia_version"=>string(VERSION),"started_utc"=>timestamp(),"numerical_pass"=>false,
        "production_release"=>false,"endpoint_maps_used"=>false,"center_matrices_inserted"=>false,
        "method"=>"bare_two_and_three_column_fixedpoints",
        "symmetry"=>(modes.use_z2 ? "bosonic_Z2" : "none"),
        "far_cap_definition"=>"identity one-copy; synchronized one-copy products for replica caps"))
    source_hashes=Dict{String,String}()
    provenance=isfile(joinpath(root,"inputs.csv")) ? filter(r->r["point_id"]==id,read_csv(joinpath(root,"inputs.csv"))) : Dict{String,String}[]
    function persist(stage)
        row["stage"]=stage; row["updated_utc"]=timestamp(); row["seconds_this_attempt"]=time()-start
        write_rows(pointpath(),[row]); println("stage=",stage," point=",id," elapsed=",row["seconds_this_attempt"]); flush(stdout)
    end
    function checked(file;own=false)
        payload=open(deserialize,file)
        all(getproperty(payload,k)==v for (k,v) in ((:point_id,id),(:beta,beta),(:chi,chi),(:seed,seed))) ||
            error("checkpoint parameter mismatch: $file")
        if own
            payload.code_sha256==code_sha && payload.config_sha256==config_sha || error("own checkpoint provenance mismatch")
            payload.use_z2==modes.use_z2 && payload.use_cap_fix==modes.use_cap_fix || error("checkpoint switches mismatch")
        end
        payload
    end
    function imported(role)
        file=path("input_"*role)
        isfile(file) || return nothing
        rec=filter(r->get(r,"role","")==role && get(r,"relative_path","")==relpath(file,root),provenance)
        length(rec)==1 || error("missing/ambiguous import provenance for $role")
        digest=sha(file); rec[1]["sha256"]==digest || error("input checksum mismatch for $role")
        source_hashes[file]=digest
        row["input_"*role*"_sha256"]=digest
        row["input_"*role*"_source_path"]=get(rec[1],"source_path","")
        payload=checked(file)
        source_config=get(rec[1],"source_config_sha256","")
        isempty(source_config) || payload.config_sha256==source_config || error("source configuration checksum mismatch for $role")
        row["input_"*role*"_config_sha256"]=payload.config_sha256
        row["input_"*role*"_code_sha256"]=payload.code_sha256
        payload
    end
    records=Any[]
    function observer(s,far,e)
        cap=far===nothing ? LMPSVUMPS.m2_corner_boundary(s.lefts) : copy(far)
        push!(records,(;seam=s.seam,cycle=copy(s.cyc),Ts=s.Ts,lefts=s.lefts,far_boundary=cap,solution=e))
        save(path("seams"),wrap(records=records,source_hashes=copy(source_hashes)))
        audit=Dict{String,Any}[]
        for r in records
            rec=Dict{String,Any}("point_id"=>id,"seam"=>r.seam,"cycle"=>join(r.cycle,";"))
            for (k,v) in pairs(r.solution)
                v isa Union{Number,AbstractString,Symbol} && (rec[string(k)]=v)
            end
            push!(audit,rec)
        end
        write_rows(pointpath("_sixr_audit.csv"),audit)
        println("SEAM ",s.seam," ",s.cyc," residual=",e.residual," convergence=",e.converged); flush(stdout)
    end
    try
        if isfile(path("result"))
            final=checked(path("result");own=true)
            final.boundary_sha256==sha(path("boundary")) && final.direct_sha256==sha(path("direct")) || error("completed checkpoint hashes mismatch")
            write_rows(pointpath(),[final.row]); println("skip complete ",id," ",final.row["status"]); return
        end
        persist("initialization"); Random.seed!(seed)
        ib=imported("boundary"); iz=modes.use_z2 ? imported("z2_boundary") : nothing
        saved=isfile(path("boundary")) ? checked(path("boundary");own=true) : nothing
        z2=nothing
        # Own checkpoints resume the exact last iterate, even if diagnostic.
        # A requeue must not reoptimize B while leaving a saved direct tensor
        # tied to the old B. Foreign imports remain convergence-gated below.
        boundary=if saved!==nothing && (!modes.use_z2 || isfile(path("z2_boundary")))
            row["boundary_reused"]=true; row["boundary_origin"]="own_checkpoint"
            row["own_unconverged_boundary_resumed"]=!saved.boundary.diagnostics.converged
            modes.use_z2 && (z2=checked(path("z2_boundary");own=true).result)
            saved.boundary
        elseif ib!==nothing && ib.boundary.diagnostics.converged && (!modes.use_z2 || iz!==nothing)
            row["boundary_reused"]=true; row["boundary_origin"]="imported_checkpoint"
            if modes.use_z2
                z2=iz.result
                hasproperty(z2.audits,:converged) && z2.audits.converged || error("graded import did not converge")
                z2.audits.symmetry=="bosonic_Z2" && z2.audits.chi_total==chi || error("graded symmetry/chi import mismatch")
                for component in (:AL,:AR,:AC,:C)
                    a1=LMPSVUMPS._host_array(getproperty(z2.boundary.state,component)[1])
                    a2=LMPSVUMPS._host_array(getproperty(ib.boundary.state,component)[1])
                    norm(a1-a2)<=1e-12*max(norm(a1),norm(a2),1.0) || error("dense/graded imported MPS disagree: $component")
                end
                norm(z2.boundary.source.A-ib.boundary.source.A)<=1e-12*max(norm(ib.boundary.source.A),1.0) || error("dense/graded PEPS mismatch")
                save(path("z2_boundary"),wrap(result=z2,input_sha256=row["input_z2_boundary_sha256"]))
            end
            save(path("boundary"),wrap(boundary=ib.boundary,seconds_boundary=0.0,input_sha256=row["input_boundary_sha256"]))
            ib.boundary
        else
            row["boundary_reused"]=false; row["boundary_origin"]="fresh_seeded_vumps"
            common=(tolerance=config["vumps_tolerance"],diagnostic_tolerance=config["vumps_diagnostic_tolerance"],
                max_iterations=config["vumps_maxiter"],verbosity=0,accept_unconverged=true)
            measured=@timed if modes.use_z2
                z2=boundary_solver(config)(beta,chi;common...,chi_even=parse(Int,get(spec,"chi_even",string(cld(chi,2)))),
                    spectrum_num=min(16,chi^2),spectrum_tol=config["spectrum_tolerance"])
                save(path("z2_boundary"),wrap(result=z2)); z2.boundary
            else
                boundary_solver(config)(beta,chi;common...)
            end
            row["seconds_boundary"]=measured.time
            save(path("boundary"),wrap(boundary=measured.value,seconds_boundary=measured.time))
            measured.value
        end
        if modes.use_z2
            z2.audits.symmetry=="bosonic_Z2" && z2.audits.chi_total==chi || error("selected graded symmetry/chi mismatch")
            for component in (:AL,:AR,:AC,:C)
                dense=LMPSVUMPS._host_array(getproperty(boundary.state,component)[1])
                graded_export=LMPSVUMPS._host_array(getproperty(z2.boundary.state,component)[1])
                norm(dense-graded_export)<=1e-12*max(norm(dense),norm(graded_export),1.0) ||
                    error("selected dense/graded boundary mismatch: $component")
            end
            norm(boundary.source.A-z2.boundary.source.A)<=1e-12*max(norm(boundary.source.A),1.0) ||
                error("selected dense/graded PEPS mismatch")
        end
        row["boundary_sha256"]=sha(path("boundary")); row["boundary_checkpoint"]=relpath(path("boundary"),root)
        bd=boundary.diagnostics
        for k in (:converged,:galerkin_residual,:iterations,:left_environment_residual,:right_environment_residual,:center_residual)
            row[k==:converged ? "vumps_converged" : k==:iterations ? "vumps_iterations" : string(k)]=getproperty(bd,k)
        end
        M=reshape(LMPSVUMPS._host_array(boundary.state.AL[1]),chi,boundary.source.D^2,chi)
        export_arrays(path("mps",".csv"),[("AL",M),("AR",reshape(LMPSVUMPS._host_array(boundary.state.AR[1]),size(M))),
            ("AC",reshape(LMPSVUMPS._host_array(boundary.state.AC[1]),size(M))),("C",LMPSVUMPS._host_array(boundary.state.C[1])),("PEPS",boundary.source.A)])
        persist("boundary_complete")
        obs=modes.use_z2 ? z2_observables(boundary,beta,z2;spectrum_num=min(16,chi^2),spectrum_tol=config["spectrum_tolerance"]) :
            vumps_boundary_observables(boundary,beta;spectrum_num=min(16,chi^2),spectrum_tol=config["spectrum_tolerance"])
        for (k,v) in pairs(obs)
            v isa Real && (row[string(k)]=v)
        end
        row["xi_boundary"]=obs.xi; row["xi_spin_exact"]=obs.exact_spin_xi
        row["sector"]=abs(obs.magnetization)<1e-3 ? "small_m" : "broken"
        spectra=Dict{String,Any}[]
        for (name,values) in (("transfer",obs.transfer_spectrum),("schmidt",obs.schmidt_singular_values))
            for (i,v) in enumerate(values)
                push!(spectra,Dict("spectrum"=>name,"rank"=>i,"real"=>real(v),"imag"=>imag(v),"magnitude"=>abs(v),
                    "probability"=>(name=="schmidt" ? obs.schmidt_probabilities[i] : "")))
            end
        end
        if modes.use_z2
            for (name,values) in pairs(z2.sector_spectra)
                values isa AbstractVector || continue
                for (i,v) in enumerate(values)
                    push!(spectra,Dict("spectrum"=>"z2_"*string(name),"rank"=>i,"real"=>real(v),"imag"=>imag(v),"magnitude"=>abs(v)))
                end
            end
            row["z2_spectrum_abs_error"]=z2.audits.dense_spectrum_abs_error
            row["z2_observable_passed"]=abs(obs.magnetization)<=1e-8 && row["z2_spectrum_abs_error"]<=1e-7
            graded_xi=z2.sector_spectra.xi_all
            row["z2_xi_all_from_sectors"]=graded_xi
            row["z2_xi_matches_sector_all"]=(isinf(obs.xi) && obs.xi>0 && isinf(graded_xi) && graded_xi>0) ||
                (isfinite(obs.xi) && isfinite(graded_xi) && isapprox(obs.xi,graded_xi;rtol=1e-7,atol=1e-8))
            row["z2_xi_saved_minus_sector_all"]=isfinite(obs.xi) && isfinite(graded_xi) ? obs.xi-graded_xi : NaN
            row["xi_definition"]="full dense boundary-MPS transfer; Inf for peripheral degeneracy within 1e-12; sector-resolved full xi separately in xi_all"
            row["xi_subleading_definition"]="first nonperipheral full-spectrum mode; NOT identified as physical-spin xi"
            row["magnetization_reference_definition"]="symmetric one-point reference zero; exact_magnetization is spontaneous broken-phase reference"
            write_rows(pointpath("_z2_audit.csv"),[Dict(string(k)=>v for (k,v) in pairs(z2.audits))])
        end
        write_rows(pointpath("_spectra.csv"),spectra); persist("observables_complete")
        a=LMPSVUMPS._dense_norm_site(boundary.source)
        row["bulk_spatial_symmetry_residual"]=maximum(norm(a-permutedims(a,p))/norm(a) for p in ((2,3,4,1),(4,3,2,1),(1,4,3,2)))
        row["bulk_spatial_symmetry_residual"]<1e-12 || error("RK spatial symmetry check failed")
        west=permutedims(M,(3,2,1)); south=west
        idirect=imported("direct")
        original=if isfile(path("direct"))
            sd=checked(path("direct");own=true)
            sd.boundary_sha256==row["boundary_sha256"] || error("own direct boundary mismatch")
            row["direct_reused"]=true; sd.original
        elseif row["boundary_origin"]=="imported_checkpoint" && idirect!==nothing &&
            idirect.boundary_sha256==row["input_boundary_sha256"]
            row["direct_reused"]=true
            save(path("direct"),wrap(original=idirect.original,seconds_direct=0.0,boundary_sha256=row["boundary_sha256"],
                input_sha256=row["input_direct_sha256"]))
            idirect.original
        else
            row["direct_reused"]=false
            idirect!==nothing && (row["direct_import_rejected"]="boundary hash does not match selected MPS")
            measured=@timed direct_original_lmps_fixedpoints(west,M,a;nmax=config["direct_maxiter"],tol=config["direct_tolerance"],
                metric_rtol=config["metric_rtol"],initial_G=Matrix{ComplexF64}(I,chi,chi),initial_B=M,solver=:synchronized_power)
            row["seconds_direct"]=measured.time
            save(path("direct"),wrap(original=measured.value,seconds_direct=measured.time,boundary_sha256=row["boundary_sha256"]))
            measured.value
        end
        row["direct_sha256"]=sha(path("direct")); row["direct_checkpoint"]=relpath(path("direct"),root)
        for k in (:residual_G,:residual_B,:iterations_G,:iterations_B,:solver,:metric_condition,:metric_dropped)
            row[string(k)]=getproperty(original,k)
        end
        row["direct_converged"]=max(original.residual_G,original.residual_B)<=config["direct_tolerance"]
        export_arrays(path("direct",".csv"),[("G",original.G),("B",original.B),("H",original.H)])
        if modes.use_z2
            sym=direct_symmetry(original,z2.virtual_parity)
            for (k,v) in pairs(sym); row["z2_direct_"*string(k)*"_covariance"]=v; end
            row["z2_direct_covariance_passed"]=maximum(values(sym))<=1e-8
        end
        persist("direct_complete")
        reference=BareStripChecks.strip(M,south,a); reconstructed=BareStripChecks.strip(original.H,south,a)
        row["fidelity"]=BareStripChecks.fidelity(original.H,M)
        row["strip_logz_reference"]=reference.logz; row["strip_logz_bare"]=reconstructed.logz
        row["strip_logz_difference"]=reconstructed.logz-reference.logz
        row["reconstruction_passed"]=abs(row["strip_logz_difference"])<=1e-7 && abs(row["fidelity"]-1)<=1e-5
        persist("reconstruction_complete")
        rails=direct_cardinal_regional_rails(original,west,M,south;rtol=config["metric_rtol"])
        row["rail_metric_dropped"]=rails.metric_dropped
        options=sixr_options(config)
        row["sixr_selection_method"]=modes.use_cap_fix ? "normalized_far_cap_power" : "legacy_arnoldi_direction_with_far_cap_intercept"
        row["sixr_tail_maxiter"]=options.endpoint_options.tail_maxiter
        row["sixr_state_tolerance"]=options.far_boundary_tolerance
        row["sixr_amplitude_tolerance"]=options.far_boundary_tail_tolerance
        row["closure_gates_passed"]=true; persist("sixr")
        measured=@timed try
            m2_contract_regional_six_R(rails.rails,boundary.source.D;options...,seam_observer=observer,verbose=true)
        catch err
            length(records)==9 || rethrow()
            message=sprint(showerror,err)
            known_gate=any(prefix->startswith(message,prefix),(
                "method2 endpoint-axis map residual ","method2 endpoint scale-cancellation residual ",
                "method2 thermodynamic-intercept cancellation ","method2 local-component identity "))
            known_gate || rethrow()
            row["closure_gates_passed"]=false; row["closure_error"]=message
            retained_closure(rails.rails,boundary.source.D,records)
        end
        r=measured.value; row["seconds_sixr"]=measured.time
        for (k,v) in pairs(r); v isa Real && (row[string(k)]=v); end
        for (key,prop) in (("max_residual",:residual),("max_adjoint_gate",:adjoint_gate),("alignment",:prefactor_alignment),
            ("drift",:prefactor_drift),("slope",:prefactor_slope_match))
            row[key]=maximum(getproperty(x.solution,prop) for x in records)
        end
        row["all_seams_converged"]=all(x.solution.converged for x in records)
        row["component_identity_error"]=abs(r.tildeS-(2r.S3-r.S2A-r.S2B-r.S2C))
        finite=all(isfinite,(r.tildeS,r.S2A,r.S2B,r.S2C,r.S3,row["max_residual"],row["alignment"],row["drift"],row["slope"]))
        sympass=!modes.use_z2 || (row["z2_observable_passed"] && row["z2_direct_covariance_passed"])
        gate=bd.converged && row["direct_converged"] && row["reconstruction_passed"] && sympass && finite &&
            row["closure_gates_passed"] && row["all_seams_converged"] && row["max_residual"]<=1e-8 &&
            row["max_adjoint_gate"]<=1e-11 && row["alignment"]<=5e-7 && row["drift"]<=5e-7 && row["slope"]<=5e-7 &&
            r.scale_gate<=5e-8 && r.map_residual<=5e-8 && r.endpoint_count==6 && r.metric_count==3 &&
            row["component_identity_error"]<=1e-10 && (!modes.use_cap_fix || r.mirror<=1e-5)
        lowrank=original.metric_dropped>0 || rails.metric_dropped>0
        row["numerical_pass"]=gate && !lowrank
        row["status"]=!gate ? "diagnostic_failed" : lowrank ? "diagnostic_low_rank" : "numerical_pass"
        row["finite_entropy_retained"]=finite
        row["stage"]="complete"; row["updated_utc"]=timestamp(); row["seconds_this_attempt"]=time()-start
        save(path("result"),wrap(result=r,row=copy(row),records=records,source_hashes=copy(source_hashes),input_provenance=provenance,
            boundary_sha256=row["boundary_sha256"],direct_sha256=row["direct_sha256"]))
        persist("complete")
    catch err
        row["status"]="failed"; row["error"]=sprint(showerror,err); persist("failed"); rethrow()
    finally
        all(sha(file)==digest for (file,digest) in source_hashes) || error("imported inputs changed during run")
        sha(joinpath(root,"config.toml"))==config_sha && sha(joinpath(root,"source_hashes.json"))==code_sha || error("campaign provenance changed during run")
    end
end

function main(args=ARGS)
    get(ENV,"SLURM_JOB_PARTITION","")=="preempt" || error("Run computations in genuine Slurm preempt")
    haskey(ENV,"SLURM_JOB_ID") || error("Missing Slurm allocation")
    BLAS.set_num_threads(parse(Int,get(ENV,"OPENBLAS_NUM_THREADS","1")))
    root=abspath(ENV["RK_CAMPAIGN_ROOT"])
    task=isempty(args) ? parse(Int,ENV["SLURM_ARRAY_TASK_ID"]) : parse(Int,args[1])
    run_point(root,task)
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    RKIsingPoint.main()
end
