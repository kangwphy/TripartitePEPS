# Diagnostic normal/anomalous/connected-density mode residues at the specified boundary.
# Stationary modes are retained in mode_groups; reported decay xi excludes |ratio| >= 1-1e-9.
include("mortier_physical_channel_xi.jl")
function diagnostic_caps(path;full_spectrum=false)
    diagnostic=isfile(joinpath(path,"diagnostic_north.jls"))
    n=deserialize(joinpath(path,diagnostic ? "diagnostic_north.jls" : "boundary_1.jls"))
    s=deserialize(joinpath(path,diagnostic ? "diagnostic_south.jls" : "boundary_3.jls"))
    # Deliberately reproduce the old single-vector caps for a mode audit.
    # Full spectra below diagnose the multiplicity this partial solve misses.
    if diagnostic || full_spectrum
        # Construct caps directly, preserving the actual unconverged flags.
        # This branch requires the COMPLETE spectrum and a unique leading mode.
        Et,Eb=n.state.AL[1],s.state.AL[1];O=n.transfer[1]
        A,B=PEPSKit.ket(O),PEPSKit.bra(O)
        rng=MersenneTwister(514)
        l0=randn(rng,ComplexF64,dual(space(Eb,4))⊗dual(space(A,5))⊗space(B,5)←space(Et,1))
        r0=randn(rng,ComplexF64,dual(space(Et,4))⊗dual(space(A,3))⊗space(B,3)←space(Eb,1))
        fl=l->PEPSKit.edge_transfer_left(l,O,Et,Eb)
        fr=r->PEPSKit.edge_transfer_right(r,O,Et,Eb)
        full_threshold=max(2048,2sum(n.chi)^2)
        vl,ls,ml=FermionicPEPS._paired_channel_eigenpairs(fl,l0;tolerance=1e-12,maxiter=1000,dense_threshold=full_threshold)
        vr,rs,mr=FermionicPEPS._paired_channel_eigenpairs(fr,r0;tolerance=1e-12,maxiter=1000,dense_threshold=full_threshold)
        @test ml==mr==:dense
        @test min(1-abs(vl[2]/vl[1]),1-abs(vr[2]/vr[1]))>1e-8
        @test maximum(norm(fl(v)-z*v)/norm(fl(v)) for (z,v) in zip(vl,ls))<1e-10
        @test maximum(norm(fr(v)-z*v)/norm(fr(v)) for (z,v) in zip(vr,rs))<1e-10
        @test abs(vl[1]-vr[1])/abs(vl[1])<1e-10
        L,R=ls[1],rs[1]
    else
        p=paired_boundary_observable(n,s,id(ComplexF64,space(n.peps[1],1));dense_threshold=0)
        @test !p.spectral_multiplicity_checked
        L,R=p.left,p.right
    end
    cs=reshape([id(ComplexF64,space(L,4)'),id(ComplexF64,space(R,1)),
        id(ComplexF64,space(R,4)'),id(ComplexF64,space(L,1))],4,1,1)
    es=reshape([n.state.AL[1],R,s.state.AL[1],L],4,1,1)
    n.peps,PEPSKit.CTMRGEnv(cs,es),!(diagnostic || full_spectrum),n.converged&&s.converged
end
function main_modes(path,output)
    peps,env,partial,boundary_converged=diagnostic_caps(path);site=CartesianIndex(1,1);target=CartesianIndex(1,2)
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
        lead=first(selected);xi=-1/log(lead["ratio_abs"])
        row=Dict("operator"=>name,"dimension"=>d,"norm_channel_residual"=>norm_res,
            "eigenvector_condition"=>cond(E.vectors),"reconstruction_error_r64"=>reconstruction,
            "residue_cutoff"=>1e-10,"leading_nonzero_residue_xi"=>xi,"mode_groups"=>groups)
        push!(rows,row);println(name," dimension=",d," condition=",cond(E.vectors),
            " reconstruction=",reconstruction," coupled xi=",xi," first modes=",groups[1:min(5,length(groups))]);flush(stdout)
    end
    open(output,"w") do io;TOML.print(io,Dict("cases"=>rows,
        "caps_from_partial_spectrum"=>partial,"boundary_converged"=>boundary_converged,
        "physical_replica_benchmark_certified"=>false));end
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    main_modes(ARGS[1],ARGS[2])
end
