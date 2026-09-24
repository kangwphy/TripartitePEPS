"Return `⟨X⟩` and `⟨Z⟩` from one saved boundary (useful for TFIM smoke tests)." 
function tfim_one_site_observables(boundary::QuantumVUMPSBoundary)
    top, = boundary.transfer[1]
    physical = codomain(top)
    X = TensorKit.TensorMap(ComplexF64[0 1; 1 0], physical ← physical)
    Z = TensorKit.TensorMap(ComplexF64[1 0; 0 -1], physical ← physical)
    (; x=quantum_one_site_expectation(boundary, X), z=quantum_one_site_expectation(boundary, Z))
end
