# Exact candidate parity relabeling, implemented with an explicit odd auxiliary
# line. All reordering/fusion is graded TensorKit arithmetic.
include("core.jl")

function vps_tensor(A)
    numout(A)==3 && numin(A)==1 || error("expected boundary MPS tensor")
    q=Vect[FermionParity](FermionParity(1)=>1)
    vl=q⊗space(A,1);vr=q⊗domain(A)[1]
    UL=isomorphism(ComplexF64,fuse(vl)←vl)
    UR=isomorphism(ComplexF64,fuse(vr)←vr)
    P=id(ComplexF64,space(A,2))⊗id(ComplexF64,space(A,3))
    (UL⊗P)*(id(ComplexF64,q)⊗A)*UR'
end

function vps_center(C)
    q=Vect[FermionParity](FermionParity(1)=>1)
    vl=q⊗codomain(C)[1];vr=q⊗domain(C)[1]
    UL=isomorphism(ComplexF64,fuse(vl)←vl)
    UR=isomorphism(ComplexF64,fuse(vr)←vr)
    UL*(id(ComplexF64,q)⊗C)*UR'
end

function vps_state(state)
    length(state)==1 || error("one-site control only")
    shifted=MPSKit.InfiniteMPS([vps_tensor(state.AL[1])],
        [vps_tensor(state.AR[1])],[vps_center(state.C[1])],[vps_tensor(state.AC[1])])
    err=max(norm(shifted.AC[1]-shifted.AL[1]*shifted.C[1]),
        norm(shifted.AC[1]-MPSKit._mul_front(shifted.C[0],shifted.AR[1])))
    err<1e-10 || error("parity shift broke canonical center equations: $err")
    shifted,err
end

function vps_ring(A,n)
    n in 1:4 || error("bounded finite ring control")
    labels=[[i,-(2i-1),-2i,mod1(i+1,n)] for i in 1:n]
    ncon(fill(A,n),labels;order=collect(1:n))
end

function vps_ring_checks(A,B)
    rows=[]
    for n in 1:4
        a,b=vps_ring(A,n),vps_ring(B,n)
        norm(a)>1e-12 && norm(b)>1e-12 || error("null finite-ring control")
        phase=dot(a,b)/dot(a,a)
        err=norm(b-phase*a)/norm(a)
        row=Dict("length"=>n,"relative_error_up_to_global_phase"=>err,
            "phase_real"=>real(phase),"phase_imag"=>imag(phase),
            "phase_modulus_error"=>abs(abs(phase)-1),
            "minus_supertrace_loop_error"=>norm(b+a)/norm(a))
        push!(rows,row)
        err<1e-10 && abs(abs(phase)-1)<1e-10 || error("finite ring changed: $row")
    end
    rows
end

function vps_prepare(source,out)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite")
    mkpath(out);BLAS.set_num_threads(2)
    files=joinpath.(source,("boundary_1.jls","boundary_3.jls","report.toml"))
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in files)
    n,s=deserialize.(files[1:2]);oldreport=TOML.parsefile(files[3])
    oldreport["final"]["outer_residual"]<1e-9 || error("input pair not accepted")
    meta=Dict{String,Any}("complete"=>false,"passed"=>false,"diagnostic_only"=>true,
        "accepted_entropy"=>false,"zero_optimization_steps"=>true,"source"=>source,
        "source_hashes"=>hashes,"job_id"=>ENV["SLURM_JOB_ID"],
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))))
    cp(@__FILE__,joinpath(out,"source_virtual_parity_shift.jl"))
    checks=[];states=[]
    try
        for (name,b) in (("north",n),("south",s))
            state,err=vps_state(b.state)
            rings=vps_ring_checks(b.state.AL[1],state.AL[1])
            twice,err2=vps_state(state)
            double_rings=vps_ring_checks(b.state.AL[1],twice.AL[1])
            push!(states,state)
            push!(checks,Dict("direction"=>name,"canonical_center_error"=>err,
                "double_shift_center_error"=>err2,"finite_rings"=>rings,
                "double_shift_finite_rings"=>double_rings))
            meta["checks"]=checks;bv_write(joinpath(out,"checks.toml"),meta)
        end
        R,S=states
        bra=FermionicPEPS._spatial_boundary_bra
        L=MPSKit.InfiniteMPS([bra(S.AR[1])];tol=1e-13,maxiter=1000)
        audit=bv_audit(R,L,n.transfer)
        meta["shifted_audit"]=audit.report
        meta["xi_mps_right"]=bv_xi(R,R);meta["xi_mps_left"]=bv_xi(L,L)
        meta["xi_pair"]=bv_xi(R,L)
        audit.residual<1e-9 || error("shifted equations failed: $(audit.residual)")
        for (file,old,state) in (("boundary_1.jls",n,R),("boundary_3.jls",s,S))
            b=save_boundary(joinpath(out,file),old,state,old.transfer,0)
            serialize(joinpath(out,file),merge(b,(;source=:graded_virtual_parity_shift,
                bivumps_converged=false,bivumps_residual=audit.residual)))
        end
        serialize(joinpath(out,"independent_pair.jls"),(;R,L,transfer=n.transfer))
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("input changed")
        meta["complete"]=true;meta["passed"]=true
        bv_write(joinpath(out,"checks.toml"),meta)
        # Exact transform remains separate from a solver run. Its report is
        # intentionally unsupported by the existing entropy adapters.
        report=merge(meta,Dict("schema"=>BIVUMPS_SCHEMA,
            "independent_left_right"=>true,"direction_constraint"=>false,
            "converged"=>audit.residual<1e-9,"tolerance"=>1e-9,
            "chi"=>sum(n.chi),"chi_sectors"=>reverse(collect(n.chi)),
            "update_strategy"=>"graded_virtual_parity_shift",
            "run_sha256"=>meta["script_sha256"],"final"=>audit.report))
        bv_write(joinpath(out,"report.toml"),report)
        println("graded shift control passed; residual=",audit.residual);flush(stdout)
    catch e
        meta["error"]=sprint(showerror,e);meta["checks"]=checks
        bv_write(joinpath(out,"checks.toml"),meta)
        rethrow()
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    vps_prepare(ARGS...)
end
