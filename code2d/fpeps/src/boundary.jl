"Parity-resolved environment space; both sectors are explicitly requested."
function _environment_space(chi::Tuple{Int,Int})
    all(>=(0),chi) && sum(chi)>0 || throw(ArgumentError("nonnegative parity dimensions required"))
    Vect[FermionParity](0=>chi[1],1=>chi[2])
end

"""Construct a length-two row transfer in the selected direction.

The PEPS is only repeated; its local tensor and infinite physical state are unchanged.
North/south rows run along columns, whereas east/west rows run along rows.
"""
function _period2_transfer(peps,direction)
    direction in 1:4 || throw(ArgumentError("direction must be in 1:4"))
    size(peps)==(1,1) || throw(ArgumentError("period2 solver expects a one-site PEPS"))
    p2 = PEPSKit.InfinitePEPS(peps[1]; unitcell=isodd(direction) ? (1,2) : (2,1))
    transfer = PEPSKit.InfiniteTransferPEPS(p2,direction,1)
    length(transfer)==2 || error("period2 transfer must contain two sites")
    p2,transfer
end

"""Experimental VUMPS solve with two independently optimized boundary sites.

This changes the boundary ansatz, not the local PEPS. It does not by itself
certify a physical sector or make one-site entropy routines applicable.
`lambda` is the expectation over the complete two-site cell.
"""
function solve_boundary_period2(peps=ksvc_ipeps(); chi=(4,4), direction=1, seed=1729,
                                tolerance=1e-9, maxiter=500, verbosity=0,
                                diagnostic_tolerance=max(100tolerance,1e-8),
                                accept_unconverged=false,initial_state=nothing,
                                iteration_callback=nothing,inner_tolerance=nothing,which=:LM,
                                algorithm=:vumps)
    which in (:LM,:LR) || throw(ArgumentError("supported local eigenvalue selectors: :LM, :LR"))
    algorithm in (:vumps,:vomps) || throw(ArgumentError("algorithm must be :vumps or :vomps"))
    algorithm==:vumps || which==:LM || throw(ArgumentError("VOMPS supports only :LM"))
    p2,transfer = _period2_transfer(peps,direction)
    Vchi = _environment_space(chi)
    rng = MersenneTwister(seed)
    draw(T,spaces...) = randn(rng,T,spaces...)
    initial = if initial_state===nothing
        PEPSKit.initialize_mps(draw,ComplexF64,transfer,[Vchi,Vchi])
    else
        length(initial_state)==2 || throw(DimensionMismatch("two-site initial MPS required"))
        for i in 1:2
            MPSKit.left_virtualspace(initial_state,i)==Vchi ||
                throw(DimensionMismatch("initial MPS must match requested chi at both cuts"))
        end
        copy(initial_state)
    end
    length(initial)==2 || error("period2 initial MPS must contain two sites")
    iterations=Ref(0)
    finalize=function(i,state,operator,env)
        iterations[]=max(iterations[],Int(i)+1)
        iteration_callback===nothing || iteration_callback(i,state,operator,env)
        state,env
    end
    inner_tolerance===nothing || (isfinite(inner_tolerance) && inner_tolerance>0) ||
        throw(ArgumentError("positive finite inner tolerance required"))
    inner_options=inner_tolerance===nothing ? (;) : (;tol=inner_tolerance,dynamic_tols=false)
    common=(;tol=tolerance,maxiter,verbosity,
        alg_gauge=MPSKit.Defaults.alg_gauge(;inner_options...),
        alg_environments=MPSKit.Defaults.alg_environments(;inner_options...),finalize)
    alg = algorithm==:vumps ? MPSKit.VUMPS(;common...,
        alg_eigsolve=MPSKit.Defaults.alg_eigsolve(;ishermitian=false,inner_options...)) :
        MPSKit.VOMPS(;common...)
    state,env,galerkin = if which==:LM && algorithm==:vumps
        MPSKit.leading_boundary(initial,transfer,alg)
    else
        env0=MPSKit.environments(initial,transfer)
        ms,ev,res=MPSKit.dominant_eigsolve(convert(MPSKit.MultilineMPO,transfer),
            convert(MPSKit.MultilineMPS,initial),alg,MPSKit.Multiline([env0]);which)
        (convert(MPSKit.InfiniteMPS,ms),ev[1],res)
    end
    length(state)==2 || error("VUMPS changed the boundary unit-cell length")
    lambda = ComplexF64(_boundary_expectation(state,transfer,env))
    left,right = env.GLs[1],env.GRs[end]
    left_residual = _scaled_residual(left*MPSKit.TransferMatrix(state.AL,transfer,state.AL),left)
    right_residual = _scaled_residual(MPSKit.TransferMatrix(state.AR,transfer,state.AR)*right,right)
    center_residuals=[max(norm(state.AC[i]-state.AL[i]*state.C[i]),
        norm(state.AC[i]-MPSKit._mul_front(state.C[i-1],state.AR[i])))/
        max(norm(state.AC[i]),eps()) for i in 1:2]
    center_residual=maximum(center_residuals)
    converged = isfinite(abs(lambda)) && abs(lambda)>0 && galerkin<=tolerance &&
        max(left_residual,right_residual,center_residual)<=diagnostic_tolerance
    !converged && !accept_unconverged && error("period2 boundary did not converge: " *
        "Galerkin=$galerkin, left=$left_residual, right=$right_residual, center=$center_residual")
    (;peps=p2,transfer,state,environments=env,chi,direction,lambda,galerkin,converged,
       left_residual,right_residual,center_residual,center_residuals,iterations=iterations[],
       unitcell_length=2,inner_tolerance,which,algorithm)
end

"""Represent an oppositely oriented physical edge as an MPS bra.

The auxiliary legs reverse direction and complex conjugation converts the
spatial edge to a bra tensor. Regrouping the two physical cups into the MPS
overlap convention additionally twists the dual physical strand. In the
north chart that is the bra leg; after a half-turn it is the ket leg.
The map is checked against complete bare and PEPS-grown edge contractions,
not against an entropy or fitted observable.
"""
function _spatial_boundary_bra(edge)
    numout(edge)==3 && numin(edge)==1 && sectortype(edge)==FermionParity ||
        throw(ArgumentError("expected a fermionic edge (chi,ket,bra;chi)"))
    @tensor result[l k b;r] := conj(edge[r k b;l])
    PEPSKit.twistdual(result,(2,3))
end

function _scaled_residual(y,x)
    xx = real(dot(x,x))
    xx > 0 || return Inf
    lambda = dot(x,y)/xx
    norm(y-lambda*x)/max(norm(y),abs(lambda)*norm(x),eps())
end

function _boundary_expectation(state,transfer,env)
    result = MPSKit.expectation_value(convert(MPSKit.MultilineMPS,state),
        convert(MPSKit.MultilineMPO,transfer),MPSKit.Multiline([env]))
    result isa Number ? result : prod(result)
end

"""Solve the norm boundary of a fixed fPEPS; never optimize its local tensor.

chi=(even,odd) specifies retained degeneracies, not a dense total dimension.
All four directions are solved by typed lattice rotations.
"""
function solve_boundary(peps=ksvc_ipeps(); chi=(4,4), direction=1, seed=1729,
                        tolerance=1e-9, maxiter=500, verbosity=0,
                        diagnostic_tolerance=max(100tolerance,1e-8),
                        accept_unconverged=false,initial_state=nothing,iteration_callback=nothing)
    direction in 1:4 || throw(ArgumentError("direction must be north=1,east=2,south=3,west=4"))
    size(peps)==(1,1) || throw(ArgumentError("initial boundary implementation requires a one-site cell"))
    transfer = PEPSKit.InfiniteTransferPEPS(peps,direction,1)
    Vchi = _environment_space(chi)
    rng = MersenneTwister(seed)
    draw(T,spaces...) = randn(rng,T,spaces...)
    initial = PEPSKit.initialize_mps(draw,ComplexF64,transfer,[Vchi])
    if initial_state !== nothing
        length(initial_state)==1 && space(initial_state.AL[1])==space(initial.AL[1]) ||
            throw(DimensionMismatch("initial boundary must match target chi and oriented physical spaces"))
        initial=copy(initial_state)
    end
    iterations = Ref(0)
    finalize = function (i,state,operator,environments)
        iterations[] = max(iterations[],Int(i)+1)
        iteration_callback === nothing || iteration_callback(i,state,operator,environments)
        state,environments
    end
    alg = MPSKit.VUMPS(;tol=tolerance,maxiter,verbosity,
        alg_eigsolve=MPSKit.Defaults.alg_eigsolve(;ishermitian=false),finalize)
    state,env,galerkin = MPSKit.leading_boundary(initial,transfer,alg)
    lambda = ComplexF64(_boundary_expectation(state,transfer,env))
    left,right = env.GLs[1],env.GRs[end]
    left_residual = _scaled_residual(left*MPSKit.TransferMatrix(state.AL,transfer,state.AL),left)
    right_residual = _scaled_residual(MPSKit.TransferMatrix(state.AR,transfer,state.AR)*right,right)
    center = state.AC[1]
    center_residual = max(norm(center-state.AL[1]*state.C[1]),
        norm(center-MPSKit._mul_front(state.C[0],state.AR[1])))/max(norm(center),eps())
    converged = isfinite(abs(lambda)) && abs(lambda)>0 && galerkin<=tolerance &&
        max(left_residual,right_residual,center_residual)<=diagnostic_tolerance
    !converged && !accept_unconverged && error("fermionic boundary did not converge: " *
        "Galerkin=$galerkin, left=$left_residual, right=$right_residual, center=$center_residual")
    (;peps,transfer,state,environments=env,chi,direction,lambda,galerkin,
       left_residual,right_residual,center_residual,iterations=iterations[],converged)
end

"CTMRG cross-check for the same fixed fPEPS, in parity-resolved spaces."
function solve_ctm(peps=ksvc_ipeps();chi=(4,4),tolerance=1e-9,maxiter=500,verbosity=0,seed=1730,
                   accept_unconverged=false,algorithm=PEPSKit.Defaults.ctmrg_alg,
                   projector=PEPSKit.Defaults.projector_alg)
    algorithm in (:SimultaneousCTMRG,:SequentialCTMRG) ||
        throw(ArgumentError("use generic CTMRG; C4v fermionic contractions are not assumed"))
    rng = MersenneTwister(seed)
    draw(T,spaces...) = randn(rng,T,spaces...)
    initial = PEPSKit.CTMRGEnv(draw,ComplexF64,peps,_environment_space(chi))
    environment,info = PEPSKit.leading_boundary(initial,peps;tol=tolerance,maxiter,verbosity,
                                                alg=algorithm,projector_alg=projector)
    lambda=norm(peps,environment)
    converged=info.converged && isfinite(abs(lambda)) && abs(lambda)>0
    !converged && !accept_unconverged && error("fermionic CTMRG did not converge: $info")
    (;peps,environment,info,chi,lambda,converged,algorithm,projector)
end

function _number_operator(V)
    n = zeros(ComplexF64,V←V)
    for (fo,fi) in fusiontrees(n)
        n[fo,fi] .= fo.uncoupled[1].isodd ? 1 : 0
    end
    n
end

"""Same-boundary variational sandwich (not a general non-Hermitian PEPS density).

Use `boundary_number(north, south)` for the physical two-sided contraction.
This one-argument method is retained for existing diagnostic scripts.
"""
function boundary_number(boundary)
    O = boundary.transfer[1]
    A,B = PEPSKit.ket(O),PEPSKit.bra(O)
    number = _number_operator(space(A,1))
    @tensor An[p;n e s w] := number[p;q]*A[q;n e s w]
    # The MPO sandwich contains a ket/bra pair, preserving its virtual factors.
    impurity = (An,B)
    center = boundary.state.AC[1]
    L,R = boundary.environments.GLs[1],boundary.environments.GRs[1]
    numerator = MPSKit.contract_mpo_expval(center,L,impurity,R)
    denominator = MPSKit.contract_mpo_expval(center,L,O,R)
    numerator/denominator
end

function _paired_channel_eigenpairs(action,initial;tolerance,maxiter,dense_threshold)
    d=length(initial.data);required=min(2,d)
    if d<=dense_threshold
        # One-vector Krylov iteration can return only one vector from an
        # exactly degenerate eigenspace and falsely report a positive gap.
        matrix=zeros(ComplexF64,d,d);v=copy(initial)
        for j in 1:d
            fill!(v.data,0);v.data[j]=1
            matrix[:,j]=action(v).data
        end
        dec=eigen(matrix);order=sortperm(abs.(dec.values);rev=true)[1:required]
        vectors=map(order) do j
            v=copy(initial);v.data.=dec.vectors[:,j];v
        end
        return dec.values[order],vectors,:dense
    end
    values,vectors,info=eigsolve(action,initial,required,:LM;
        tol=tolerance,maxiter,krylovdim=40)
    info.converged>=required || error("two-sided channel eigenpairs did not converge")
    values[1:required],vectors[1:required],:krylov
end

"""Physical single-site observable between opposite, independently solved boundaries.

Both MPSs are used as directed double-layer tensors, without replacing the
south boundary by a conjugate of the north state. PEPSKit's typed edge
transfers retain the ket/bra twists. The left/right mixed horizontal channel
is solved explicitly; its arbitrary per-cell phase cancels in the ratio.
Residual, spectral isolation, overlap conditioning and Hermiticity gates are
independent of any Gaussian reference or entropy value.
Channels of dimension at most `dense_threshold` use the complete spectrum to
detect exact multiplicity. Larger Krylov solves check both returned residuals,
but report `spectral_multiplicity_checked=false` since one starting vector
cannot certify the multiplicity of an exactly degenerate eigenspace.
"""
function paired_boundary_observable(north,south,operator;
        tolerance=1e-12,residual_tolerance=1e-10,gap_tolerance=1e-8,
        phase_tolerance=1e-8,max_overlap_condition=1e10,seed=514,maxiter=1000,
        dense_threshold=512)
    north.converged && south.converged || throw(ArgumentError("unconverged opposite boundary"))
    north.direction==1 && south.direction==3 || throw(ArgumentError("expected north=1, south=3"))
    size(north.peps)==size(south.peps)==(1,1) && north.peps[1]≈south.peps[1] ||
        throw(ArgumentError("opposite boundaries must belong to the same one-site PEPS"))
    Et,Eb=north.state.AL[1],south.state.AL[1]
    O=north.transfer[1]; A,B=PEPSKit.ket(O),PEPSKit.bra(O)
    space(operator)==space(id(space(A,1))) || throw(DimensionMismatch("physical operator space"))
    rng=MersenneTwister(seed)
    l0=randn(rng,ComplexF64,dual(space(Eb,4))⊗dual(space(A,5))⊗space(B,5)←space(Et,1))
    r0=randn(rng,ComplexF64,dual(space(Et,4))⊗dual(space(A,3))⊗space(B,3)←space(Eb,1))
    fl=l->PEPSKit.edge_transfer_left(l,O,Et,Eb)
    fr=r->PEPSKit.edge_transfer_right(r,O,Et,Eb)
    vl,ls,left_method=_paired_channel_eigenpairs(fl,l0;tolerance,maxiter,dense_threshold)
    vr,rs,right_method=_paired_channel_eigenpairs(fr,r0;tolerance,maxiter,dense_threshold)
    l,r=ls[1],rs[1]
    pair(x,y)=@tensor x[a k b;c]*y[c k b;a]
    left_residual=maximum(norm(fl(v)-z*v)/norm(fl(v)) for (z,v) in zip(vl,ls))
    right_residual=maximum(norm(fr(v)-z*v)/norm(fr(v)) for (z,v) in zip(vr,rs))
    eigenvalue_mismatch=abs(vl[1]-vr[1])/max(abs(vl[1]),abs(vr[1]))
    gap=min(length(vl)>1 ? 1-abs(vl[2]/vl[1]) : 1.0,
            length(vr)>1 ? 1-abs(vr[2]/vr[1]) : 1.0)
    overlap_condition=norm(l)*norm(r)/abs(pair(l,r))
    maximum((left_residual,right_residual,eigenvalue_mismatch))<=residual_tolerance ||
        error("two-sided channel residual failed: $left_residual, $right_residual, $eigenvalue_mismatch")
    gap>gap_tolerance || error("two-sided channel has unresolved leading sector: gap=$gap")
    isfinite(overlap_condition) && overlap_condition<=max_overlap_condition ||
        error("two-sided overlap is ill conditioned: $overlap_condition")
    @tensor An[p;n e s w] := operator[p;q]*A[q;n e s w]
    value=pair(PEPSKit.edge_transfer_left(l,(An,B),Et,Eb),r)/pair(fl(l),r)
    if isapprox(operator,operator';atol=1e-12,rtol=1e-12)
        abs(imag(value))<=phase_tolerance*max(1,abs(value)) ||
            error("two-sided Hermitian observable is complex: $value; inspect boundary sector/support")
    end
    (;value,left_residual,right_residual,eigenvalue_mismatch,gap,overlap_condition,
      spectral_multiplicity_checked=(left_method==right_method==:dense),
      spectrum_methods=(left_method,right_method),
      lambda_left=vl[1],lambda_right=vr[1],left=l,right=r,converged=true)
end

"Physical density using a north/south pair; the one-argument diagnostic is distinct."
function boundary_number(north,south;kwargs...)
    A=PEPSKit.ket(north.transfer[1])
    result=paired_boundary_observable(north,south,_number_operator(space(A,1));kwargs...)
    -1e-8<=real(result.value)<=1+1e-8 || error("density outside physical interval: $(result.value)")
    result.value
end

function ctm_number(boundary)
    V = PEPSKit.physicalspace(boundary.peps)
    n = _number_operator(V[1,1])
    op = PEPSKit.LocalOperator(V,((1,1),)=>n)
    PEPSKit.expectation_value(boundary.peps,op,boundary.environment)
end

"Density pair separated along the transfer row; orientation follows direction."
function boundary_number_correlation(boundary,distance::Int=1)
    distance>=1 || throw(ArgumentError("positive separation required"))
    O=boundary.transfer[1]
    A,B=PEPSKit.ket(O),PEPSKit.bra(O)
    number=_number_operator(space(A,1))
    @tensor An[p;n e s w] := number[p;q]*A[q;n e s w]
    L,R=boundary.environments.GLs[1],boundary.environments.GRs[1]
    function contraction(insertions)
        current=L
        for i in 0:distance
            M=i==distance ? boundary.state.AC[1] : boundary.state.AL[1]
            op=insertions && i in (0,distance) ? (An,B) : O
            current=MPSKit.transfer_left(current,op,M,M)
        end
        PEPSKit.environment_overlap(current,R)
    end
    contraction(true)/contraction(false)
end

"Finite-chi even boundary correlation length in original lattice spacings (full cell transfer)."
function boundary_even_correlation_length(boundary)
    V=MPSKit.left_virtualspace(boundary.state,1)
    template=similar(boundary.state.AL[1],V,V)
    d=length(template.data)
    d<=4096 || error("dense correlation-length diagnostic exceeds limit")
    action=flip(MPSKit.TransferMatrix(boundary.state.AL,boundary.state.AL))
    M=zeros(ComplexF64,d,d)
    for j in 1:d
        fill!(template.data,0); template.data[j]=1
        M[:,j]=action(template).data
    end
    vals=eigvals(M)
    ord=sortperm(abs.(vals);rev=true)
    ratio=abs(vals[ord[2]]/vals[ord[1]])
    ratio<1 || return Inf
    -length(boundary.state)/log(ratio)
end
