# Isolated antiunitary tangent audit for the independent Newton equations.
# The ba_* action definitions below are copied verbatim from antiunitary_pair.jl
# (79aaaa4652722ed54c80b923b71760d9b1abea72cb1b6cca38147063170f9dc5).
# The original is not included because it reloads core.jl and its tensor module.
# No established solver or source used by live jobs is modified.
include("branch_response.jl")
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

# Each state determines its own intertwining gauge. Neither is obtained from
# the other boundary. These are the same maps used by projected Anderson.
function at_symmetry(state,change)
    A=ba_state(state,change).AL[1];B=state.AL[1]
    cap=bv_cap(x->MPSKit.transfer_left(x,A,B),
        randn(MersenneTwister(2851),ComplexF64,space(B,1)←space(A,1)))
    fidelity=abs(abs(cap.z)-1)
    fidelity<1e-10 || error("state outside antiunitary manifold: $fidelity")
    U,_,Vh=svd_compact(cap.v);G=U*Vh;phase=cap.z/abs(cap.z)
    P=id(ComplexF64,space(A,2))⊗id(ComplexF64,space(A,3))
    apply=T->(G⊗P)*change(T)*G'/phase
    J=bv_matrix(apply,B)
    report=Dict("state_fidelity_error"=>fidelity,
        "unitarity_error"=>norm(J'*J-I)/sqrt(size(J,1)),
        "involution_error"=>norm(J*conj.(J)-I)/sqrt(size(J,1)),
        "coefficient_defect"=>norm(apply(B)-B)/norm(B))
    (;apply,report)
end

function at_tangent(R,L,maps)
    sr=at_symmetry(R,maps.right);sl=at_symmetry(L,maps.left)
    NR=TensorKit.left_null(R.AL[1]);NL=TensorKit.left_null(L.AL[1])
    tr=zero(NR'*R.AL[1]);tl=zero(NL'*L.AL[1])
    nr=length(tr.data);nl=length(tl.data);count=nr+nl
    function apply(v)
        z=complex.(v[1:count],v[count+1:end])
        vr=TensorMap(copy(z[1:nr]),space(tr))
        vl=TensorMap(copy(z[nr+1:end]),space(tl))
        bn_real(vcat((NR'*sr.apply(NR*vr)).data,(NL'*sl.apply(NL*vl)).data))
    end
    S=zeros(2count,2count)
    for j in 1:2count
        v=zeros(2count);v[j]=1;S[:,j]=apply(v)
    end
    E=eigen(Symmetric((S+S')/2));positive=findall(>(0),E.values)
    length(positive)==count || error("unexpected symmetry-fixed tangent dimension")
    Q=E.vectors[:,positive]
    report=Dict("right"=>sr.report,"left"=>sl.report,
        "full_dimension"=>2count,"fixed_dimension"=>count,
        "symmetry_error"=>norm(S-S')/sqrt(2count),
        "involution_error"=>norm(S*S-I)/sqrt(2count),
        "eigenvalue_error"=>maximum(abs.(abs.(E.values).-1)),
        "basis_isometry_error"=>norm(Q'*Q-I)/sqrt(count),
        "basis_fixed_error"=>norm(S*Q-Q)/sqrt(count))
    (;S,Q,apply,sr,sl,report)
end

function at_audit(source,out;perturb=false)
    haskey(ENV,"SLURM_JOB_ID") || error("submit through Slurm")
    ispath(out) && error("refusing overwrite");mkpath(out);BLAS.set_num_threads(2)
    paths=joinpath.(source,("boundary_1.jls","boundary_3.jls","independent_pair.jls"))
    files=[paths...]
    for f in ("antiunitary_newton_chart.jl","antiunitary_pair.jl","branch_response.jl",
        "newton_metric_response.jl","newton_response_fast.jl","core.jl",
        "../gaussian_boundary_oriented_chart.jl")
        push!(files,joinpath(@__DIR__,f))
    end
    hashes=Dict(p=>bytes2hex(sha256(read(p))) for p in files)
    original=read(joinpath(@__DIR__,"antiunitary_pair.jl"),String)
    copied=original[findfirst("function ba_bridge",original).start:findfirst("function ba_covariance",original).start-1]
    occursin(copied,read(@__FILE__,String)) || error("antiunitary actions differ from audited originals")
    meta=Dict{String,Any}("complete"=>false,"passed"=>false,"diagnostic_only"=>true,
        "independent_left_right"=>true,"direction_constraint"=>false,"accepted_entropy"=>false,
        "source_hashes"=>hashes,"perturbed"=>perturb,"job_id"=>ENV["SLURM_JOB_ID"])
    bv_write(joinpath(out,"checks.toml"),meta)
    try
        n,s=deserialize.(paths[1:2]);pair=deserialize(paths[3]);maps=ba_pair_maps(n,s)
        R,L=deepcopy(pair.R),deepcopy(pair.L)
        if perturb
            states=[]
            for (i,state,change) in ((1,R,maps.right),(2,L,maps.left))
                symmetry=at_symmetry(state,change)
                raw=state.AL[1]+0.002randn(MersenneTwister(3010+i),ComplexF64,space(state.AL[1]))
                raw=(raw+symmetry.apply(raw))/2
                raw=raw*bn_invsqrt(raw'*raw)
                candidate,_=bt_canonical(raw);push!(states,candidate)
            end
            R,L=states
        end
        meta["initial_audit"]=bv_audit(R,L,pair.transfer).report
        symmetry=at_tangent(R,L,maps);meta["tangent_symmetry"]=symmetry.report
        bv_write(joinpath(out,"checks.toml"),meta)
        c=btm_chart(R,L,pair.transfer);Q=symmetry.Q;S=symmetry.S
        c.count==size(S,1) || error("tangent dimension mismatch")
        g=c.gradient;meta["gradient_norm"]=norm(g)
        norm(g)>1e-7 || error("stationary zero does not test gradient symmetry")
        meta["gradient_outside_fixed_fraction"]=norm(g-Q*(Q'*g))/norm(g)
        meta["quotient_relative_imaginary"]=abs(imag(c.base.q))/abs(c.base.q)
        rng=MersenneTwister(3013);checks=[]
        for side in ("right","left"),quadrature in ("real","imag")
            raw=zeros(c.count);offset=quadrature=="real" ? 0 : div(c.count,2)
            ids=side=="right" ? (1:c.nr) : (c.nr+1:c.nr+c.nl)
            raw[offset .+ ids].=randn(rng,length(ids))
            v=(raw+S*raw)/2;v/=norm(c.pushforward(v))
            scale=max(norm(c.pushforward(raw)),eps())
            hv=c.action(v)
            commutator=norm(S*c.pushforward(v)-c.pushforward(S*v))/norm(c.pushforward(v))
            leakage=norm(hv-Q*(Q'*hv))/max(norm(hv),eps())
            covariance=norm(c.action(S*(raw/scale))-S*c.action(raw/scale))/max(norm(c.action(raw/scale)),eps())
            stencils=[]
            for h in (1e-4,3e-5,1e-5)
                p=c.evaluate(h*v);m=c.evaluate(-h*v)
                fd=(p.gradient-m.gradient)/(2h)
                defect=max(norm(symmetry.sr.apply(p.R)-p.R)/norm(p.R),
                    norm(symmetry.sl.apply(p.L)-p.L)/norm(p.L))
                push!(stencils,Dict("step"=>h,
                    "hessian_relative_error"=>norm(fd-hv)/max(norm(hv),eps()),
                    "scalar_error"=>abs(real((p.q-m.q)/(2h*c.base.q))-dot(g,v)),
                    "retracted_tensor_symmetry_defect"=>defect,
                    "quotient_relative_imaginary"=>abs(imag(p.q))/abs(p.q)))
            end
            push!(checks,Dict("side"=>side,"quadrature"=>quadrature,
                "metric_commutator"=>commutator,"hessian_outside_fixed_fraction"=>leakage,
                "hessian_covariance_error"=>covariance,"stencils"=>stencils))
            meta["checks"]=checks;bv_write(joinpath(out,"checks.toml"),meta)
        end
        errors=[meta["gradient_outside_fixed_fraction"],meta["quotient_relative_imaginary"]]
        for (k,v) in symmetry.report
            v isa AbstractDict ? append!(errors,values(v)) : occursin("error",k) && push!(errors,v)
        end
        for r in checks
            append!(errors,[r["metric_commutator"],r["hessian_outside_fixed_fraction"],r["hessian_covariance_error"]])
            append!(errors,[t["retracted_tensor_symmetry_defect"] for t in r["stencils"]])
            minimum(t["hessian_relative_error"] for t in r["stencils"])<1e-4 || error("reduced Hessian finite difference failed")
            minimum(t["scalar_error"] for t in r["stencils"])<1e-7 || error("reduced scalar derivative failed")
        end
        meta["maximum_symmetry_error"]=maximum(errors)
        meta["passed"]=maximum(errors)<1e-8
        meta["complete"]=true
        all(bytes2hex(sha256(read(p)))==h for (p,h) in hashes) || error("audit source changed")
        bv_write(joinpath(out,"checks.toml"),meta)
        meta["passed"] || error("antiunitary tangent is not verified: $(maximum(errors))")
        println("antiunitary Newton tangent verified: ",symmetry.report["full_dimension"],
            " -> ",symmetry.report["fixed_dimension"]," real dimensions");flush(stdout)
    catch err
        meta["error"]=sprint(showerror,err);bv_write(joinpath(out,"checks.toml"),meta);rethrow()
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    at_audit(ARGS[1],ARGS[2];perturb=length(ARGS)>2 ? parse(Bool,ARGS[3]) : false)
end
