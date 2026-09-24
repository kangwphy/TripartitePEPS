# Experimental contraction of Gaussian-fidelity MPSs. Production gates and
# caches are untouched: these inputs explicitly remain non-VUMPS boundaries.
include("direct_grown_turn_probe.jl")
include("character_lmps_measurement.jl")
using DelimitedFiles
include("gaussian_boundary_oriented_chart.jl")

function import_gaussian_boundary(root,name,w,chi)
    p=InfinitePEPS(PEPSKit.flip_virtualspace(ksvc_tensor(;bond_weight=(w,w),bond_gauge=:balanced),(2,4)))
    peps=name in ("b","w") ? rotl90(p) : p
    direction=name in ("s","w") ? 3 : 1
    chart=gaussian_oriented_chart(peps,direction);op=chart.original
    O=chart.fused;V=Vect[FermionParity](0=>1,1=>1)
    X=FermionicPEPS._environment_space((chi÷2,chi÷2))
    template=PEPSKit.initialize_mps(randn,ComplexF64,chart.canonical,[X])
    raw=readdlm(joinpath(root,"$(name)_w$(w)","chi$(chi).csv"),',',Float64)
    array=reshape(complex.(raw[:,1],raw[:,2]),chi,2,2,chi)
    F=isomorphism(ComplexF64,space(O,2),V⊗V')
    G=isomorphism(ComplexF64,space(O,2),space(template.AL[1],2)⊗space(template.AL[1],3))
    T0=TensorMap(array,X⊗V⊗V'←X)
    @tensor Tf[l p;r] := T0[l a b;r]*F[p;a b]
    @tensor T[l a b;r] := Tf[l p;r]*conj(G[p;a b])
    # Exact reversible fusion, including physical dual metrics.
    @tensor Tf2[l p;r] := T[l a b;r]*G[p;a b]
    @tensor back[l a b;r] := Tf2[l p;r]*conj(F[p;a b])
    norm(back-T0)<1e-12*norm(T0) || error("physical fusion roundtrip failed")
    T=gaussian_flip_boundary(T,chart;inverse=true)
    native_template=PEPSKit.initialize_mps(randn,ComplexF64,op,[X])
    space(T)==space(native_template.AL[1]) || error("native arrow transport failed")
    state=MPSKit.InfiniteMPS([T]);env=MPSKit.environments(state,op)
    galerkin=MPSKit.calc_galerkin(state,op,state,env)
    lambda=FermionicPEPS._boundary_expectation(state,op,env)
    boundary=(;state,transfer=op,peps,direction,chi=(chi÷2,chi÷2),galerkin,lambda,
               converged=false,source=:gaussian_fidelity_compression)
    println("import ",name," chi=",chi," galerkin=",galerkin);flush(stdout)
    boundary
end

function compressed_observables(n,s;distances=1:8,dense_threshold=512)
    n.direction==1 && s.direction==3 && n.peps[1]≈s.peps[1] || error("orientation mismatch")
    Et,Eb=n.state.AL[1],s.state.AL[1];O=n.transfer[1]
    A,B=PEPSKit.ket(O),PEPSKit.bra(O)
    rng=MersenneTwister(514)
    l0=randn(rng,ComplexF64,dual(space(Eb,4))⊗dual(space(A,5))⊗space(B,5)←space(Et,1))
    r0=randn(rng,ComplexF64,dual(space(Et,4))⊗dual(space(A,3))⊗space(B,3)←space(Eb,1))
    fl=l->PEPSKit.edge_transfer_left(l,O,Et,Eb)
    fr=r->PEPSKit.edge_transfer_right(r,O,Et,Eb)
    # Use the same multiplicity-aware channel solver as the native physical
    # probe. This diagnostic may accept non-VUMPS inputs, but an accidental
    # single-vector omission of a degenerate eigenvalue is never a gap.
    vl,ls,left_method=FermionicPEPS._paired_channel_eigenpairs(fl,l0;
        tolerance=1e-12,maxiter=1000,dense_threshold)
    vr,rs,right_method=FermionicPEPS._paired_channel_eigenpairs(fr,r0;
        tolerance=1e-12,maxiter=1000,dense_threshold)
    l,r=ls[1],rs[1];pair(x,y)=@tensor x[a k b;c]*y[c k b;a]
    residuals=[maximum(norm(fl(v)-z*v)/norm(fl(v)) for (z,v) in zip(vl,ls)),
               maximum(norm(fr(v)-z*v)/norm(fr(v)) for (z,v) in zip(vr,rs)),
               abs(vl[1]-vr[1])/max(abs(vl[1]),abs(vr[1]))]
    maximum(residuals)<1e-10 || error("physical channel residuals: $residuals")
    gap=min(1-abs(vl[2]/vl[1]),1-abs(vr[2]/vr[1]))
    gap>1e-8 || error("physical leading sector unresolved: $gap")
    condition=norm(l)*norm(r)/abs(pair(l,r))
    condition<1e10 || error("physical overlap ill conditioned: $condition")
    number=FermionicPEPS._number_operator(space(A,1))
    @tensor An[p;n e s w] := number[p;q]*A[q;n e s w]
    advance(x,insert)=PEPSKit.edge_transfer_left(x,insert ? (An,B) : O,Et,Eb)
    density=pair(advance(l,true),r)/pair(fl(l),r)
    values=ComplexF64[]
    for distance in distances
        numerator=advance(l,true);denominator=advance(l,false)
        for step in 1:distance
            numerator=advance(numerator,step==distance);denominator=advance(denominator,false)
        end
        push!(values,pair(numerator,r)/pair(denominator,r))
    end
    Dict("density_real"=>real(density),"density_imag"=>imag(density),
         "distances"=>collect(distances),"nn_real"=>real.(values),"nn_imag"=>imag.(values),
         "phase_pass"=>maximum(abs,imag.([density;values]))<=1e-8,
         "channel_residuals"=>residuals,"gap"=>gap,"overlap_condition"=>condition,
         "spectral_multiplicity_checked"=>left_method==right_method==:dense,
         "spectrum_methods"=>[string(left_method),string(right_method)],
         "physical_even_channel_xi"=>-1/log(1-gap))
end

function compressed_direct_rails(boundary;full_spectrum=false)
    AL,AR=boundary.state.AL[1],boundary.state.AR[1];O=boundary.transfer[1]
    raw=FermionicPEPS._grow_boundary(AL,PEPSKit.ket(O),PEPSKit.bra(O))
    Q=permute(PEPSKit.twistdual(raw,(2,3)),((1,2,3,4,5),(6,7,8)))
    Vl=space(Q,1)⊗space(Q,2)⊗space(Q,3);Vr=domain(Q)
    UL=isomorphism(ComplexF64,fuse(Vl)←Vl);UR=isomorphism(ComplexF64,fuse(Vr)←Vr)
    physical=id(ComplexF64,space(Q,4))⊗id(ComplexF64,space(Q,5))
    D=(UL⊗physical)*Q*UR'
    rng=MersenneTwister(610)
    l0=randn(rng,ComplexF64,space(AL,1)←space(D,1))
    r0=randn(rng,ComplexF64,dual(space(D,4))←dual(space(AR,4)))
    fl=x->MPSKit.transfer_left(x,D,AL);fr=x->MPSKit.transfer_right(x,D,AR)
    vl,ls,vr,rs=if full_spectrum
        vl,ls,ml=FermionicPEPS._paired_channel_eigenpairs(fl,l0;tolerance=1e-12,maxiter=1000,dense_threshold=length(l0.data))
        vr,rs,mr=FermionicPEPS._paired_channel_eigenpairs(fr,r0;tolerance=1e-12,maxiter=1000,dense_threshold=length(r0.data))
        ml==mr==:dense || error("complete direct tail spectra required")
        abs(vl[1]-vr[1])<1e-10*max(abs(vl[1]),abs(vr[1])) || error("direct full-tail eigenvalue mismatch")
        (vl,ls,vr,rs)
    else
        vl,ls,il=eigsolve(fl,l0,2,:LM;tol=1e-12,maxiter=1000,krylovdim=40)
        vr,rs,ir=eigsolve(fr,r0,2,:LM;tol=1e-12,maxiter=1000,krylovdim=40)
        il.converged>=2 && ir.converged>=2 || error("direct tail eigensolve failed")
        (vl,ls,vr,rs)
    end
    L,R=ls[1]/norm(ls[1]),rs[1]/norm(rs[1])
    residuals=[FermionicPEPS._scaled_residual(fl(L),L),FermionicPEPS._scaled_residual(fr(R),R)]
    maximum(residuals)<1e-10 || error("direct tail residual failed")
    gaps=[1-abs(vl[2]/vl[1]),1-abs(vr[2]/vr[1])]
    minimum(gaps)>1e-8 || error("direct tails unresolved: $gaps")
    G=L*R;B=(L⊗physical)*D*R;target=AL*G
    kappa=dot(target,B)/dot(target,target)
    bg=norm(B-kappa*target)/norm(B)
    singular=reduce(vcat,[svdvals(b) for (_,b) in blocks(G)])
    support=minimum(singular)/maximum(singular)
    support>1e-12 || error("unresolved direct metric: $support")
    H=B*inv(G)
    vals,_,info=eigsolve(x->MPSKit.transfer_right(x,H,H),id(ComplexF64,space(H,1)),1,:LM;tol=1e-12,maxiter=1000)
    info.converged>=1 || error("rail normalization failed");H/=sqrt(abs(vals[1]))
    turn=permute(L*UR,((1,3,4),(2,)))
    report=Dict("galerkin"=>boundary.galerkin,"tail_residuals"=>residuals,
                "tail_spectral_multiplicity_checked"=>full_spectrum,
                "tail_gaps"=>gaps,"bg_residual"=>bg,"metric_relative_singular"=>support,
                "rail_vs_boundary_residual"=>FermionicPEPS._scaled_residual(H,AL),
                "boundary_self_even_xi"=>boundary_even_correlation_length(boundary),
                "vumps_converged"=>boundary.converged)
    println("compressed direct rail ",report);flush(stdout)
    (;H,turn,report)
end

function compressed_mixed_rails(boundary,opposite)
    Mn=boundary.state.AL[1];Mb=FermionicPEPS._spatial_boundary_bra(opposite.state.AR[1])
    O=boundary.transfer[1];A,Ab=PEPSKit.ket(O),PEPSKit.bra(O)
    function grow(Aket)
        raw=FermionicPEPS._grow_boundary(Mn,Aket,Ab)
        Q=permute(PEPSKit.twistdual(raw,(2,3)),((1,2,3,4,5),(6,7,8)))
        Vl=space(Q,1)⊗space(Q,2)⊗space(Q,3);Vr=domain(Q)
        UL=isomorphism(ComplexF64,fuse(Vl)←Vl);UR=isomorphism(ComplexF64,fuse(Vr)←Vr)
        physical=id(ComplexF64,space(Q,4))⊗id(ComplexF64,space(Q,5))
        (;D=(UL⊗physical)*Q*UR',UR,physical)
    end
    function fixed(M,side,initial)
        f=side==:left ? x->MPSKit.transfer_left(x,M,Mb) : x->MPSKit.transfer_right(x,M,Mb)
        vals,vecs,info=eigsolve(f,initial,2,:LM;tol=1e-12,maxiter=1000,krylovdim=40)
        info.converged>=2 || error("physical mixed tail failed")
        v=vecs[1]/norm(vecs[1]);residual=norm(f(v)-vals[1]*v)/norm(f(v))
        gap=1-abs(vals[2]/vals[1])
        residual<1e-10 && gap>1e-8 || error("unresolved mixed tail: residual=$residual gap=$gap")
        (;v,lambda=vals[1],residual,gap,f)
    end
    rng=MersenneTwister(907)
    base=fixed(Mn,:left,randn(rng,ComplexF64,space(Mb,1)←space(Mn,1)))
    x=grow(A)
    left=fixed(x.D,:left,randn(rng,ComplexF64,space(Mb,1)←space(x.D,1)))
    right=fixed(x.D,:right,randn(rng,ComplexF64,dual(space(x.D,4))←dual(space(Mb,4))))
    abs(left.lambda-right.lambda)<1e-10*abs(left.lambda) || error("mixed eigenvalue mismatch")
    number=FermionicPEPS._number_operator(space(A,1))
    @tensor An[p;n e s w] := number[p;q]*A[q;n e s w]
    density=dot(MPSKit.transfer_left(left.v,grow(An).D,Mb)',right.v)/dot(left.f(left.v)',right.v)
    # Check this projection against the bare edge contraction, before replicas.
    independent=boundary.direction==1 ? compressed_observables(boundary,opposite;distances=1:1) :
                                       compressed_observables(opposite,boundary;distances=1:1)
    ref=complex(independent["density_real"],independent["density_imag"])
    abs(density-ref)<1e-8 || error("physical mixed density chart mismatch: $density vs $ref")
    K=base.v;singular=reduce(vcat,[svdvals(b) for (_,b) in blocks(K)])
    support=minimum(singular)/maximum(singular);support>1e-12 || error("mixed metric rank deficient")
    Ki=inv(K);turn=permute(left.v*x.UR,((1,3,4),(2,)))*Ki
    H=(K⊗x.physical)*Mn*Ki
    report=Dict("galerkin"=>boundary.galerkin,"bg_residual"=>NaN,"tail_residuals"=>[base.residual,left.residual,right.residual],
        "tail_gaps"=>[base.gap,left.gap,right.gap],"metric_relative_singular"=>support,
        "boundary_self_even_xi"=>boundary_even_correlation_length(boundary),
        "density_real"=>real(density),"density_imag"=>imag(density),"density_chart_error"=>abs(density-ref),
        "vumps_converged"=>false)
    println("compressed physical mixed rail ",report);flush(stdout)
    (;H,turn,report)
end

function measure_compressed_stilde(boundaries,out;maxdepth=256,construction=:direct,
        boundary_label="Gaussian fidelity boundary")
    regional=construction==:direct ? map(compressed_direct_rails,(boundaries.n,boundaries.b,boundaries.s)) :
        (compressed_mixed_rails(boundaries.n,boundaries.s),compressed_mixed_rails(boundaries.b,boundaries.w),
         compressed_mixed_rails(boundaries.s,boundaries.n))
    a,b,c=regional;fliprail(M)=permute(M,((4,2,3),(1,)))
    seams=(;AB=(a.turn,fliprail(b.H)),AC=(fliprail(a.H),c.H),BC=(b.turn,fliprail(c.H)))
    serialize(joinpath(out,"uncapped_geometry.jls"),(;regional,seams))
    caps=solve_seam_caps(seams;dense_threshold=512)
    println("seam gaps ",[x.gap for x in values(caps.diagnostics)]);flush(stdout)
    serialize(joinpath(out,"geometry.jls"),(;regional,seams,caps))
    # Verify the exact parity-character shortcut on THIS approximate geometry.
    errors=Float64[]
    for depth in (1,2)
        full=finite_lmps_sectors(seams,caps.caps;depth)
        shortcut=character_lmps_sectors(seams,caps.caps;depth)
        append!(errors,[FermionicPEPS._sector_relative_error(x,y) for (x,y) in zip(values(full),values(shortcut))])
    end
    use_character=maximum(errors)<1e-10
    builder=use_character ? character_lmps_sectors : finite_lmps_sectors
    println("character audit errors=",errors," using shortcut=",use_character);flush(stdout)
    cache=Dict{Any,Any}();rows=Dict{String,Any}[]
    for depth in (4,8,16,32,64,128,256,512,1024,2048)
        depth<=maxdepth || break
        sectors=builder(seams,caps.caps;depth,propagation_cache=cache)
        serialize(joinpath(out,"sectors_depth$(depth).jls"),sectors)
        logs=[log(abs(z.value))+z.logscale for z in values(sectors)]
        phases=[angle(z.value) for z in values(sectors)]
        phase2=[exp(im*(phases[i]-2phases[1])) for i in 2:4]
        phase4=exp(im*(phases[5]-4phases[1]))
        phase_error=maximum(abs.([phase2 .- 1;phase4-1]))
        row=Dict{String,Any}("depth"=>depth,"log_magnitudes"=>logs,"raw_phases"=>phases,
            "physical_phase_error"=>phase_error,"phase_pass"=>phase_error<=1e-8,
            "log_ratio_real_diagnostic"=>sum(logs[2:4])-logs[5]-2logs[1],
            "log_ratio_phase_diagnostic"=>angle(prod(phase2)/phase4),
            "endpoint_residual"=>maximum(FermionicPEPS._active_endpoint_residual(z) for z in values(sectors)))
        if row["phase_pass"]
            row["stilde"]=lmps_sector_entropies(sectors;phase_tolerance=1e-8).stilde
        end
        push!(rows,row)
        report=Dict{String,Any}("construction"=>string(construction," ",boundary_label," LMPS diagnostic"),
            "physical_replica_benchmark_certified"=>false,"vumps_converged"=>all(x->x.converged,values(boundaries)),
            "character_audit_errors"=>errors,"character_used"=>use_character,
            "regional"=>[x.report for x in regional],"samples"=>rows,"length_converged"=>false)
        if length(rows)>=3 && all(r->r["phase_pass"],rows[end-2:end])
            tail=rows[end-2:end]
            drift=maximum(abs(tail[i]["stilde"]-tail[i-1]["stilde"]) for i in 2:3)
            linear=maximum(FermionicPEPS._line_fit([r["depth"] for r in tail],
                [r["log_magnitudes"][i] for r in tail]).residual for i in 1:5)
            report["length_drift"]=drift;report["linear_tail_residual"]=linear
            report["length_converged"]=drift<=1e-7 && linear<=1e-7 && row["endpoint_residual"]<=1e-6
            report["stilde"]=row["stilde"]
        end
        open(joinpath(out,"entropy.toml"),"w") do io;TOML.print(io,report);end
        println("entropy ",row);flush(stdout)
        report["length_converged"] && break
    end
end

function gaussian_measurement_main()
    root=ARGS[1];w=parse(Float64,ARGS[2]);chi=parse(Int,ARGS[3])
    mode=length(ARGS)>=4 ? ARGS[4] : "all"
    cache=joinpath(root,"measure_w$(w)_chi$(chi)","experimental_boundaries.jls")
    out=joinpath(mode=="mixed" ? joinpath(root,"physical_mixed") : root,"measure_w$(w)_chi$(chi)");mkpath(out)
    boundaries=if length(ARGS)>=5 && ARGS[5]=="reuse"
        saved=deserialize(cache)
        all(b->b.source==:gaussian_fidelity_compression && sum(b.chi)==chi && !b.converged,values(saved)) || error("invalid experimental cache")
        expected=InfinitePEPS(PEPSKit.flip_virtualspace(ksvc_tensor(;bond_weight=(w,w),bond_gauge=:balanced),(2,4)))
        saved.n.peps[1]≈expected[1] || error("cached PEPS mismatch")
        saved
    else
        NamedTuple{(:n,:s,:b,:w)}(Tuple(import_gaussian_boundary(root,name,w,chi) for name in ("n","s","b","w")))
    end
    serialize(joinpath(out,"experimental_boundaries.jls"),boundaries)
    if mode in ("all","observables")
        report=Dict{String,Any}("w"=>w,"chi"=>chi,"vumps_converged"=>false)
        csvrows=Any["w" "chi" "axis" "r" "density_real" "density_imag" "nn_real" "nn_imag" "physical_even_channel_xi"]
        for (axis,n,s) in (("x",boundaries.n,boundaries.s),("y",boundaries.b,boundaries.w))
            obs=compressed_observables(n,s);report[axis]=obs
            for (i,r) in enumerate(obs["distances"])
                csvrows=vcat(csvrows,reshape(Any[w,chi,axis,r,obs["density_real"],obs["density_imag"],
                    obs["nn_real"][i],obs["nn_imag"][i],obs["physical_even_channel_xi"]],1,:))
            end
            open(joinpath(out,"observables.toml"),"w") do io;TOML.print(io,report);end
            writedlm(joinpath(out,"observables.csv"),csvrows,',')
        end
        println("observables ",report);flush(stdout)
    end
    if mode in ("all","entropy","mixed")
        try
            measure_compressed_stilde(boundaries,out;construction=mode=="mixed" ? :physical_mixed : :direct)
        catch err
            open(joinpath(out,"entropy_failure.toml"),"w") do io
                TOML.print(io,Dict("error"=>sprint(showerror,err),"physical_replica_benchmark_certified"=>false))
            end
            rethrow()
        end
    end
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    gaussian_measurement_main()
end
