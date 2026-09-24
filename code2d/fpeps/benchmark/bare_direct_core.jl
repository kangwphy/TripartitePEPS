# Bare spatial two-/three-column fixed points. No grown tensor or endmap.
include("../src/FermionicPEPS.jl")
using .FermionicPEPS, PEPSKit, MPSKit, TensorKit, LinearAlgebra, Random, Serialization, TOML, SHA

function bare_dense_fixedpoint(action,seed;gap_tolerance=1e-8)
    d=length(seed.data);d<=512 || error("small-chi complete spectrum only")
    matrix=zeros(ComplexF64,d,d)
    for j in 1:d
        x=zero(seed);x.data[j]=1
        y=action(x);space(y)==space(seed) || error("bare channel changes spaces")
        matrix[:,j]=y.data
    end
    E=eigen(matrix);order=sortperm(abs.(E.values);rev=true)
    values=E.values[order];lambda=values[1]
    gap=1-abs(values[2]/lambda)
    pairerrors=[norm(matrix*v-z*v)/max(norm(matrix)*norm(v),eps()) for (z,v) in zip(E.values,eachcol(E.vectors))]
    maximum(pairerrors)<1e-11 || error("bare full spectrum residual failed")
    x=copy(seed);x.data.=E.vectors[:,order[1]];x/=norm(x)
    z=dot(seed,x);abs(z)>1e-14 && (x*=conj(z)/abs(z))
    residual=norm(action(x)-lambda*x)/norm(action(x))
    report=Dict("dimension"=>d,"relative_modulus_gap"=>gap,"residual"=>residual,
        "eigenvalues_real"=>real.(values),"eigenvalues_imag"=>imag.(values),
        "max_eigenpair_residual"=>maximum(pairerrors),"isolated"=>gap>gap_tolerance)
    (;tensor=x,lambda,report,matrix)
end

"Bare G=fp(top|bottom), B=fp(top|PEPS|bottom), in PEPSKit's spatial chart."
function bare_direct_objects(north,south)
    north.direction==1 && south.direction==3 && north.peps[1]≈south.peps[1] || error("bare pair directions")
    Et,Eb=north.state.AL[1],south.state.AL[1]
    O=north.transfer[1];A,Ab=PEPSKit.ket(O),PEPSKit.bra(O)
    G0=zeros(ComplexF64,dual(space(Eb,4))←space(Et,1))
    for (_,b) in blocks(G0); b.=Matrix{ComplexF64}(I,size(b)...);end
    B0=randn(MersenneTwister(20260917),ComplexF64,
        dual(space(Eb,4))⊗dual(space(A,5))⊗space(Ab,5)←space(Et,1))
    fG=x->PEPSKit.edge_transfer_left(x,Et,Eb)
    fB=x->PEPSKit.edge_transfer_left(x,O,Et,Eb)
    g=bare_dense_fixedpoint(fG,G0);b=bare_dense_fixedpoint(fB,B0)
    G,B=g.tensor,b.tensor
    all(x->x.report["isolated"] && x.report["residual"]<1e-11,(g,b)) ||
        error("bare leading spaces unresolved")
    singular=reduce(vcat,[svdvals(block) for (_,block) in blocks(G)])
    minimum(singular)>1e-12*maximum(singular) || error("bare metric needs resolved support")
    Gi=inv(G)
    HA=B*Gi
    physical=id(ComplexF64,space(B,2))⊗id(ComplexF64,space(B,3))
    HB=(Gi⊗physical)*B
    report=Dict("method"=>"bare_two_and_three_column_fixedpoints",
        "grown_tensor_used"=>false,"endpoint_maps_used"=>false,
        "G_channel"=>g.report,"B_channel"=>b.report,
        "metric_condition"=>maximum(singular)/minimum(singular),
        "inverse_residual"=>norm(G*Gi-id(ComplexF64,codomain(G))),
        "logz"=>log(abs(b.lambda/g.lambda)),"accepted_entropy"=>false)
    (;G,B,HA,HB,g,b,report,Et,Eb,O)
end

"Global bare geometry, rotated as a whole: no fitted right-turn/endmap tensor."
function bare_cardinal_seams(objects,opposite)
    fliprail(T)=permute(T,((4,2,3),(1,)))
    C=opposite.state.AL[1]
    seams=(;AB=(objects.Eb,fliprail(objects.Et)),
             AC=(fliprail(objects.HA),C),BC=(objects.HB,fliprail(C)))
    for (name,(X,Y)) in pairs(seams)
        all(i->space(X,i)==dual(space(Y,i)),(2,3)) || error("bare seam physical mismatch $name")
    end
    for (X,Y) in ((seams.AB[1],seams.AC[1]),(seams.AB[2],seams.BC[1]),(seams.AC[2],seams.BC[2]))
        space(X,1)==dual(space(Y,1)) || error("bare regional junction arrows")
    end
    seams
end
