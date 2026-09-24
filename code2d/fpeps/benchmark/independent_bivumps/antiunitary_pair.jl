# Independent-pair antiunitary audit. No legacy joint solver is included.
# The physical J is a hypothesis imported from the finite Gaussian reference;
# mixed-equation covariance is checked here before any constrained update.
include("core.jl")
include("../gaussian_boundary_oriented_chart.jl")

function ba_bridge(old)
    chart=gaussian_oriented_chart(old.peps,old.direction)
    prototype=gaussian_flip_boundary(old.state.AL[1],chart)
    V=Vect[FermionParity](0=>1,1=>1)
    F=isomorphism(ComplexF64,space(chart.fused,2),V⊗V')
    G=isomorphism(ComplexF64,space(chart.fused,2),space(prototype,2)⊗space(prototype,3))
    function to_modes(X)
        T=gaussian_flip_boundary(X,chart)
        @tensor Tf[l p;r] := T[l a b;r]*G[p;a b]
        @tensor Y[l a b;r] := Tf[l p;r]*conj(F[p;a b])
        Y
    end
    function from_modes(T)
        @tensor Tf[l p;r] := T[l a b;r]*F[p;a b]
        @tensor Y[l a b;r] := Tf[l p;r]*conj(G[p;a b])
        gaussian_flip_boundary(Y,chart;inverse=true)
    end
    norm(from_modes(to_modes(old.state.AL[1]))-old.state.AL[1])<1e-12 || error("mode chart roundtrip failed")
    (;to_modes,from_modes,chart)
end

function ba_act(T,J)
    raw=convert(Array,T);out=zeros(ComplexF64,size(raw))
    for a in 0:1,b in 0:1,c in 0:1,d in 0:1
        out[:,a+1,b+1,:] .+= J[2a+b+1,2c+d+1].*raw[:,c+1,d+1,:]
    end
    TensorMap(out,space(T))
end
ba_conj(T)=TensorMap(conj.(T.data),space(T))

function ba_native_map(old)
    bridge=ba_bridge(old);J=Matrix{ComplexF64}(I,4,4)
    J[2:3,2:3]=ComplexF64[0 im;im 0]
    change=T->bridge.from_modes(ba_act(ba_conj(bridge.to_modes(T)),J))
    (;change,bridge,J)
end

function ba_pair_maps(n,s)
    kr=ba_native_map(n);ks=ba_native_map(s)
    bra=FermionicPEPS._spatial_boundary_bra
    prototype=s.state.AR[1];chart=bv_matrix(bra,prototype)
    norm(chart'*chart-I)<1e-12 || error("nonunitary spatial-bra chart")
    unbra=y->TensorMap(conj.(chart'*y.data),space(prototype))
    kl=y->bra(ks.change(unbra(y)))
    (;right=kr.change,left=kl,south=ks.change,unbra,bra,native=(kr,ks))
end

function ba_state(state,change)
    result=MPSKit.InfiniteMPS(change.(state.AL),change.(state.AR),ba_conj.(state.C),change.(state.AC))
    err=max(norm(result.AC[1]-result.AL[1]*result.C[1]),
        norm(result.AC[1]-MPSKit._mul_front(result.C[0],result.AR[1])))
    err<1e-10 || error("antiunitary canonical identities failed: $err")
    result
end

function ba_covariance(R,L,op,maps)
    a=bv_audit(R,L,op)
    r=ba_state(R,maps.right);l=ba_state(L,maps.left);b=bv_audit(r,l,op)
    jr=bv_matrix(maps.right,R.AC[1]);jl=bv_matrix(maps.left,L.AC[1])
    unitary=max(norm(jr'*jr-I),norm(jl'*jl-I))
    involution=max(norm(jr*conj(jr)-I),norm(jl*conj(jl)-I))
    pencils=[]
    for (name,P,Q,ur,ul) in (("AC",a.env.ac,b.env.ac,jr,jl),
        ("C",a.env.c,b.env.c,Matrix{ComplexF64}(I,length(R.C[1].data),length(R.C[1].data)),
         Matrix{ComplexF64}(I,length(L.C[1].data),length(L.C[1].data))))
        for field in (:H,:N)
            expected=ul*conj.(getproperty(P,field))*ur'
            actual=getproperty(Q,field);scale=dot(expected,actual)/dot(expected,expected)
            error=norm(actual-scale*expected)/norm(actual)
            push!(pencils,Dict("center"=>name,"operator"=>string(field),"relative_error"=>error,
                "scale_real"=>real(scale),"scale_imag"=>imag(scale)))
        end
    end
    rowerror=abs(b.env.q/conj(a.env.q)-1)
    residual_error=abs(b.residual-a.residual)/max(a.residual,b.residual,1e-10)
    report=Dict("unitarity_error"=>unitary,"involution_error"=>involution,
        "row_conjugacy_error"=>rowerror,"residual_relative_difference"=>residual_error,
        "original_residual"=>a.residual,"partner_residual"=>b.residual,"pencils"=>pencils)
    (;report,partner=(R=r,L=l),original=a,transformed=b)
end

function ba_audit(source,out;perturb=true)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");BLAS.set_num_threads(2)
    paths=joinpath.(source,("boundary_1.jls","boundary_3.jls","independent_pair.jls"))
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in paths)
    for f in (@__FILE__,joinpath(@__DIR__,"core.jl"),joinpath(@__DIR__,"../gaussian_boundary_oriented_chart.jl"))
        hashes[f]=bytes2hex(sha256(read(f)))
    end
    n,s=deserialize.(paths[1:2]);pair=deserialize(paths[3]);maps=ba_pair_maps(n,s)
    R,L=deepcopy(pair.R),deepcopy(pair.L)
    if perturb
        # Independent nonstationary variations prevent a stationary-zero test.
        states=[]
        for (i,state) in enumerate((R,L))
            eta=randn(MersenneTwister(28103+i),ComplexF64,space(state.AL[1]))
            raw=state.AL[1]+0.01norm(state.AL[1])*eta/norm(eta)
            push!(states,MPSKit.InfiniteMPS([raw];tol=1e-13,maxiter=2000))
        end
        R,L=states
    end
    c=ba_covariance(R,L,n.transfer,maps)
    relevant=[c.report["unitarity_error"],c.report["involution_error"],c.report["row_conjugacy_error"]]
    append!(relevant,[p["relative_error"] for p in c.report["pencils"]])
    perturb && push!(relevant,c.report["residual_relative_difference"])
    perturb && c.original.residual<1e-5 && error("nonstationary control accidentally stationary")
    all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("source changed")
    report=Dict("complete"=>true,"passed"=>maximum(relevant)<1e-9,"nonstationary"=>perturb,
        "independent_left_right"=>true,"direction_constraint"=>false,"checks"=>c.report,
        "source_hashes"=>hashes,"job_id"=>ENV["SLURM_JOB_ID"],"accepted_entropy"=>false)
    bv_write(out,report);println(report);flush(stdout)
    report["passed"] || error("independent antiunitary covariance not established")
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    ba_audit(ARGS[1],ARGS[2];perturb=length(ARGS)>2 ? parse(Bool,ARGS[3]) : true)
end
