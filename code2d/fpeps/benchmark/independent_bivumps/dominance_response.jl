# Independent trial: linearized spectral-dominance constraints for Newton steps.
# No stationary equation, fermionic contraction or final acceptance gate changes.
include("branch_response.jl")

function bd_pair_spectrum(A,B;ranks=2,targets=nothing)
    rng=MersenneTwister(2507)
    xl=randn(rng,ComplexF64,space(B,1)←space(A,1))
    xr=randn(rng,ComplexF64,dual(space(A,4))←dual(space(B,4)))
    fl=x->MPSKit.transfer_left(x,A,B);fr=x->MPSKit.transfer_right(x,A,B)
    EL=eigen(bv_matrix(fl,xl));ER=eigen(bv_matrix(fr,xr))
    order=sortperm(abs.(EL.values);rev=true);modes=[]
    indices=targets===nothing ? order[1:min(ranks,length(order))] :
        [argmin(abs.(EL.values.-z)) for z in targets]
    length(unique(indices))==length(indices) || error("ambiguous competing-root labels")
    for j in indices
        z=EL.values[j];k=argmin(abs.(ER.values.-z))
        abs(ER.values[k]-z)<1e-9*abs(z) || error("dominance channel spectra disagree")
        left=TensorMap(copy(EL.vectors[:,j]),space(xl));left/=norm(left)
        right=TensorMap(copy(ER.vectors[:,k]),space(xr));right/=norm(right)
        res=max(norm(fl(left)-z*left)/norm(fl(left)),norm(fr(right)-z*right)/norm(fr(right)))
        res<1e-10 || error("dominance eigenvector residual")
        sep=minimum(abs.(EL.values[setdiff(eachindex(EL.values),[j])].-z))/abs(z)
        sep>1e-9 || error("nonsimple root in dominance derivative")
        push!(modes,(;z,left,right,res,sep))
    end
    (;modes,gap=log(abs(modes[1].z/modes[2].z)))
end

function bd_constraints(R,L,O;vectors=true,refs=nothing)
    grown=bv_grow(R,O[1]);D=grown.D;P=grown.physical
    row=bd_pair_spectrum(D,L;targets=refs===nothing ? nothing : refs.row)
    overlap=bd_pair_spectrum(R,L;targets=refs===nothing ? nothing : refs.overlap)
    nextrefs=(row=[m.z for m in row.modes],overlap=[m.z for m in overlap.modes])
    gaps=[row.gap,overlap.gap]
    reports=[Dict("channel"=>name,"log_modulus_gap"=>p.gap,
        "roots"=>[Dict("real"=>real(m.z),"imag"=>imag(m.z),
            "residual"=>m.res,"complex_separation"=>m.sep) for m in p.modes])
        for (name,p) in (("row",row),("overlap",overlap))]
    vectors || return (;gaps,reports,refs=nextrefs)
    G=bv_matrix(x->bv_grow(x,O[1]).D,R)
    growadj=Z->TensorMap(G'*Z.data,space(R))
    function loggradient(A,m;grow=false)
        h=(m.left⊗P)*A*m.right;den=dot(L,h)
        abs(den)>1e-14*norm(L)*norm(h) || error("ill-conditioned dominance pairing")
        hback=(m.left⊗P)'*L*m.right'
        gr=(grow ? growadj(hback) : hback)/conj(den)
        gl=h/den
        (;gr,gl)
    end
    normals=[]
    for (A,p,grow) in ((D,row,true),(R,overlap,false))
        a=loggradient(A,p.modes[1];grow);b=loggradient(A,p.modes[2];grow)
        push!(normals,(;R=a.gr-b.gr,L=a.gl-b.gl))
    end
    (;gaps,reports,normals,refs=nextrefs)
end

function bd_chart_constraints(R,L,O,c)
    s=bd_constraints(R.AL[1],L.AL[1],O)
    NR=TensorKit.left_null(R.AL[1]);NL=TensorKit.left_null(L.AL[1])
    gradients=[c.pushforward(bn_real(vcat((NR'*v.R).data,(NL'*v.L).data))) for v in s.normals]
    # Leave a margin above the unchanged original 1e-9 modulus-gap guard.
    floors=max.(1.5e-9,0.1s.gaps)
    minimum(s.gaps-floors)>0 || error("starting cap too close to dominance boundary")
    (;s.gaps,s.reports,s.refs,gradients,floors)
end
