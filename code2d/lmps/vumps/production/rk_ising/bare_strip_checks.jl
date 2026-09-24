module BareStripChecks
using LinearAlgebra
using TensorKit: @tensor
using KrylovKit

function apply2(X,north,south)
    @tensor Y[ip,jp] := north[i,p,ip]*south[j,p,jp]*X[i,j]
    Y
end
function apply3(X,north,south,a)
    @tensor Y[ip,xp,jp] := north[i,n,ip]*a[n,x,s,xp]*south[j,s,jp]*X[i,x,j]
    Y
end
function leading(f,initial)
    vals,vecs,info=eigsolve(f,initial/norm(initial),1,:LM;
        tol=1e-12,krylovdim=min(length(initial),40),maxiter=500)
    info.converged>=1 || error("matrix-free strip solve failed")
    v=vecs[1];lambda=vals[1];y=f(v)
    residual=norm(y-lambda*v)/max(norm(y),abs(lambda)*norm(v),eps())
    residual<=1e-10 || error("strip residual $residual")
    (;lambda,residual)
end
function strip(north,south,a)
    chi,dp,_=size(north)
    size(north)==size(south) || error("boundary shape mismatch")
    size(a)==(dp,dp,dp,dp) || error("site shape mismatch")
    e2=leading(X->apply2(X,north,south),Matrix{ComplexF64}(I,chi,chi))
    initial=ones(ComplexF64,chi,dp,chi)
    e3=leading(X->apply3(X,north,south,a),initial)
    (;logz=log(abs(e3.lambda/e2.lambda)),residual2=e2.residual,
      residual3=e3.residual,lambda2=e2.lambda,lambda3=e3.lambda)
end
function transfer(A,B)
    chi,dp,_=size(A)
    E=zeros(ComplexF64,chi^2,chi^2)
    for p in 1:dp
        E .+= kron(conj(Matrix(@view B[:,p,:])),Matrix(@view A[:,p,:]))
    end
    E
end
radius(A,B)=maximum(abs,eigvals(transfer(A,B)))
fidelity(A,B)=radius(A,B)/sqrt(radius(A,A)*radius(B,B))
function spectrum(A)
    s=sort(abs.(eigvals(transfer(A,A)));rev=true)
    s/s[1]
end
end
