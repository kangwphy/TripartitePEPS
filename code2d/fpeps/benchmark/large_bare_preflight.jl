# Complete bare G/B caps up to chi=32, isolated from the unchanged small-chi core.
# This is a preflight only: it does not measure or certify a replica entropy.
include("bare_direct_core.jl")

function large_bare_fixedpoint(action,seed)
    d=length(seed.data);d<=2048 || error("bounded complete bare spectrum exceeded")
    matrix=zeros(ComplexF64,d,d)
    for j in 1:d
        x=zero(seed);x.data[j]=1;y=action(x)
        space(y)==space(seed) || error("bare channel changes space")
        matrix[:,j]=y.data
    end
    E=eigen(matrix);order=sortperm(abs.(E.values);rev=true);values=E.values[order]
    scale=norm(matrix)
    errors=[norm(matrix*v-z*v)/max(scale*norm(v),eps()) for (z,v) in zip(E.values,eachcol(E.vectors))]
    maximum(errors)<1e-11 || error("complete eigenpair backward error")
    x=copy(seed);x.data.=E.vectors[:,order[1]];x/=norm(x)
    overlap=dot(seed,x);abs(overlap)>1e-14 && (x*=conj(overlap)/abs(overlap))
    lambda=values[1];gap=1-abs(values[2]/lambda)
    residual=norm(action(x)-lambda*x)/norm(action(x))
    gap>1e-8 && residual<1e-11 || error("bare leading mode unresolved")
    report=Dict("dimension"=>d,"relative_modulus_gap"=>gap,"residual"=>residual,
        "max_eigenpair_residual"=>maximum(errors),"isolated"=>true,
        "eigenvalues_real"=>real.(values),"eigenvalues_imag"=>imag.(values),
        "complete_seed_space_spectrum"=>true,"odd_insertion_spectrum_measured"=>false)
    (;tensor=x,lambda,report,matrix)
end

function large_bare_preflight(source,out)
    mkpath(out);isfile(joinpath(out,"report.toml")) && error("refusing overwrite")
    files=joinpath.(source,("boundary_1.jls","boundary_3.jls"))
    hashes=Dict(f=>bytes2hex(sha256(read(f))) for f in files)
    n,s=deserialize.(files);chi=sum(n.chi)
    n.direction==1 && s.direction==3 && n.peps[1]≈s.peps[1] && n.chi==s.chi || error("invalid pair")
    Et,Eb=n.state.AL[1],s.state.AL[1];O=n.transfer[1];A,Ab=PEPSKit.ket(O),PEPSKit.bra(O)
    G0=zeros(ComplexF64,dual(space(Eb,4))←space(Et,1))
    for (_,b) in blocks(G0);b.=Matrix{ComplexF64}(I,size(b)...);end
    B0=randn(MersenneTwister(20260917),ComplexF64,
        dual(space(Eb,4))⊗dual(space(A,5))⊗space(Ab,5)←space(Et,1))
    g=large_bare_fixedpoint(x->PEPSKit.edge_transfer_left(x,Et,Eb),G0)
    b=large_bare_fixedpoint(x->PEPSKit.edge_transfer_left(x,O,Et,Eb),B0)
    G,B=g.tensor,b.tensor
    singular=reduce(vcat,[svdvals(block) for (_,block) in blocks(G)])
    minimum(singular)>1e-12*maximum(singular) || error("unresolved bare metric support")
    Gi=inv(G);inverse_error=norm(G*Gi-id(ComplexF64,codomain(G)))
    inverse_error<1e-9 || error("bare metric inverse failed")
    physical=id(ComplexF64,space(B,2))⊗id(ComplexF64,space(B,3))
    HA=B*Gi;HB=(Gi⊗physical)*B
    joint=TOML.parsefile(joinpath(source,"report.toml"));last=joint["rows"][end]
    side=last["sides"][1];expected=complex(side["row_lambda_real"],side["row_lambda_imag"])
    ratio=b.lambda/g.lambda;row_error=abs(ratio/expected-1)
    row_error<1e-9 || error("bare and joint row weights disagree: $row_error")
    control=Dict{String,Any}("performed"=>chi<=16)
    if chi<=16
        old=bare_direct_objects(n,s)
        error=max(norm(old.G-G),norm(old.B-B),norm(old.HA-HA),norm(old.HB-HB))
        error<1e-10 || error("original bare core comparison failed")
        control["max_tensor_difference"]=error
    end
    report=Dict("complete"=>true,"chi"=>chi,"method"=>"bare_two_and_three_column_fixedpoints",
        "grown_tensor_used"=>false,"endpoint_maps_used"=>false,"G_channel"=>g.report,
        "B_channel"=>b.report,"metric_condition"=>maximum(singular)/minimum(singular),
        "inverse_residual"=>inverse_error,"logz"=>log(abs(ratio)),"row_weight_relative_error"=>row_error,
        "original_core_control"=>control,"source_hashes"=>hashes,"accepted_entropy"=>false,
        "joint_converged"=>joint["joint_converged"],"joint_residual"=>joint["joint_residual"],
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "original_core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"bare_direct_core.jl")))))
    serialize(joinpath(out,"bare_objects.jls"),(;G,B,HA,HB,g,b,report,Et,Eb,O))
    all(bytes2hex(sha256(read(f)))==h for (f,h) in hashes) || error("source changed")
    open(joinpath(out,"report.toml"),"w") do io;TOML.print(io,report);end
    println("large bare preflight chi=",chi," dimensions=",(g.report["dimension"],b.report["dimension"]),
        " row weight relative error=",row_error," inverse error=",inverse_error);flush(stdout)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    BLAS.set_num_threads(parse(Int,get(ENV,"SLURM_CPUS_PER_TASK","2")))
    for source in ARGS[2:end]
        n=deserialize(joinpath(source,"boundary_1.jls"))
        large_bare_preflight(source,joinpath(ARGS[1],"chi$(sum(n.chi))"))
    end
end
