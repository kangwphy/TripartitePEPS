# Wide bare G/B construction using the previously checked complete-spectrum helper.
# The original small-chi core and seam geometry are unchanged.
include("large_bare_preflight.jl")

function wide_bare_direct_objects(north,south)
    north.direction==1 && south.direction==3 && north.peps[1]≈south.peps[1] || error("bare pair directions")
    Et,Eb=north.state.AL[1],south.state.AL[1]
    O=north.transfer[1];A,Ab=PEPSKit.ket(O),PEPSKit.bra(O)
    G0=zeros(ComplexF64,dual(space(Eb,4))←space(Et,1))
    for (_,b) in blocks(G0); b.=Matrix{ComplexF64}(I,size(b)...);end
    B0=randn(MersenneTwister(20260917),ComplexF64,
        dual(space(Eb,4))⊗dual(space(A,5))⊗space(Ab,5)←space(Et,1))
    fG=x->PEPSKit.edge_transfer_left(x,Et,Eb)
    fB=x->PEPSKit.edge_transfer_left(x,O,Et,Eb)
    g=large_bare_fixedpoint(fG,G0);b=large_bare_fixedpoint(fB,B0)
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

