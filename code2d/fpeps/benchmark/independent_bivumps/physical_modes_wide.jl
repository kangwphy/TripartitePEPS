# New-schema adapter of the unchanged signed physical measurement.
# Upstream measure_joint_physical_modes.jl sha256=e081317ce04411c969188049679e148774a05d5746a7aa9ffdd0687fe5606912
# Operator-coupled physical decay lengths, preserving the existing signed
# correlator and mode reconstruction. Independent from no-column pair xi.
include("../stability_observable_modes.jl")
using SHA
BLAS.set_num_threads(parse(Int,get(ENV,"SLURM_CPUS_PER_TASK","2")))
function measure_joint_modes(path,output)
    isfile(output) && error("refusing overwrite")
    files=joinpath.(path,("boundary_1.jls","boundary_3.jls"))
    hashes=Dict(f=>bytes2hex(sha256(read(f))) for f in files)
    boundary=deserialize(files[1]);chi=sum(boundary.chi)
    iseven(chi) && chi<=32 || error("bounded complete physical-mode spectrum through chi32")
    peps,env,partial,boundary_converged=diagnostic_caps(path;full_spectrum=true);site=CartesianIndex(1,1);target=CartesianIndex(1,2)
    T=PEPSKit.edge_transfermatrix(1,1,peps,peps,env)
    rows=Dict{String,Any}[]
    for name in ("normal","anomalous","nn")
        operator=signed_operators(space(peps[1],1))[name];O=MPSKit.FiniteMPO(operator)
        vn,vo=PEPSKit.start_correlator_left(site,peps,O[1],peps,env)
        mu=dot(vn,vn*T)/dot(vn,vn)
        norm_res=norm(vn*T-mu*vn)/norm(vn*T)
        @test norm_res<1e-10
        denominator=PEPSKit.end_correlator_right_denominator(target,vn*T,env)
        finish(v)=PEPSKit.end_correlator_right_numerator(target,v,peps,O[2],peps,env)/denominator
        d=length(vo.data);M=zeros(ComplexF64,d,d)
        for j in 1:d
            v=zero(vo);v.data[j]=1;M[:,j]=(v*T).data
        end
        E=eigen(M);coefficients=E.vectors\vo.data
        amplitudes=ComplexF64[]
        for j in 1:d
            v=zero(vo);v.data.=E.vectors[:,j];push!(amplitudes,finish(v)*coefficients[j])
        end
        ratios=E.values./mu
        actual=MPSKit.correlator(peps,operator,site,[CartesianIndex(1,r+1) for r in 1:64],env)
        predicted=[sum(amplitudes.*ratios.^(r-1)) for r in 1:64]
        reconstruction=maximum(abs.(predicted.-actual));@test reconstruction<1e-10
        # Degenerate eigenvectors are not individually invariant; sum their residues.
        groups=Dict{String,Any}[];remaining=Set(1:d)
        while !isempty(remaining)
            j=argmax(k->abs(ratios[k]),collect(remaining))
            indices=[k for k in remaining if abs(ratios[k]-ratios[j])<1e-9]
            amplitude=sum(amplitudes[indices]);ratio=ratios[j]
            push!(groups,Dict("multiplicity"=>length(indices),"ratio_real"=>real(ratio),
                "ratio_imag"=>imag(ratio),"ratio_abs"=>abs(ratio),"amplitude_real"=>real(amplitude),
                "amplitude_imag"=>imag(amplitude),"amplitude_abs"=>abs(amplitude)))
            setdiff!(remaining,indices)
        end
        selected=filter(g->g["amplitude_abs"]>1e-10 && g["ratio_abs"]<1-1e-9,groups)
        isempty(selected) && error("no resolved decaying operator mode")
        lead=first(selected);xi=-1/log(lead["ratio_abs"])
        cutoff_checks=[]
        for cutoff in (1e-10,1e-8,1e-6)
            kept=filter(g->g["amplitude_abs"]>cutoff && g["ratio_abs"]<1-1e-9,groups)
            push!(cutoff_checks,Dict("residue_cutoff"=>cutoff,
                "xi"=>isempty(kept) ? 0.0 : -1/log(first(kept)["ratio_abs"]),
                "resolved"=>!isempty(kept)))
        end
        row=Dict("operator"=>name,"dimension"=>d,"norm_channel_residual"=>norm_res,
            "eigenvector_condition"=>cond(E.vectors),"reconstruction_error_r64"=>reconstruction,
            "residue_cutoff"=>1e-10,"leading_nonzero_residue_xi"=>xi,
            "residue_cutoff_checks"=>cutoff_checks,"mode_groups"=>groups)
        push!(rows,row);println(name," dimension=",d," condition=",cond(E.vectors),
            " reconstruction=",reconstruction," coupled xi=",xi," first modes=",groups[1:min(5,length(groups))]);flush(stdout)
    end
    all(bytes2hex(sha256(read(f)))==h for (f,h) in hashes) || error("source changed")
    joint=TOML.parsefile(joinpath(path,"report.toml"))
    joint["schema"]=="independent_bivumps_v1" || error("wrong solver schema")
    open(output,"w") do io
        TOML.print(io,Dict("complete"=>true,"chi"=>chi,"cases"=>rows,
            "source_hashes"=>hashes,"caps_from_partial_spectrum"=>partial,
            "bivumps_residual"=>joint["final"]["outer_residual"],"bivumps_converged"=>joint["converged"],
            "boundary_converged"=>boundary_converged,"accepted_entropy"=>false,
            "definition"=>"Decay modes of actual physical correlator transfer including the PEPS column, with nonzero operator residues; stationary groups are retained but excluded from decay xi.",
            "script_sha256"=>bytes2hex(sha256(read(@__FILE__)))))
    end
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    for path in ARGS
        measure_joint_modes(path,joinpath(path,"physical_modes.toml"))
    end
end
