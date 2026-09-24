"""
An isolated PEPSKit/MPSKit VUMPS frontend for the existing ITensor six-R LMPS
closure.  The historical density-matrix projector is not included here and is
not modified by loading this module.
"""
module LMPSVUMPS

using Adapt
import CUDA
using ITensors
using KrylovKit
using LinearAlgebra
using MPSKit
using PEPSKit
using Random
using TensorKit
import cuTENSOR

include("algorithms/state.jl")
include("models/rk_ising.jl")
include("algorithms/sixr_closure.jl")
include("algorithms/direct_route.jl")
include("algorithms/boundary.jl")
include("models/rk_ising_observables.jl")
include("algorithms/bridge.jl")
include("models/tfim.jl")
include("algorithms/quantum_boundary.jl")
include("models/tfim_observables.jl")
include("algorithms/quantum_sixr.jl")
include("models/rvb_d3.jl")

const rvb_d3_peps = RVBD3Model.rvb_d3_peps
const rvb_d3_dense_tensor = RVBD3Model.rvb_d3_dense_tensor

export PEPS, rk_ising, onsager_f,
       VUMPSBoundaryDiagnostics, VUMPSBoundary,
       pepskit_from_lmps, solve_vumps_boundary, vumps_storage_backend,
       vumps_logz, vumps_free_energy, vumps_boundary_observables,
       vumps_ising_magnetization, ising_exact_magnetization,
       ising_exact_correlation_length,
       VUMPSLTRDiagnostics, vumps_ltr_objects,
       VUMPSRectangleDiagnostics, vumps_projector_objects,
       vumps_sixr_rails, solve_vumps_sixr,
       m2_contract_shared_six_R, m2_contract_regional_six_R,
       TransverseIsing2D, tfim_hamiltonian, tfim_initial_peps,
       SimpleUpdateGroundState, VariationalGroundState,
       solve_simple_update_groundstate,
       solve_tfim_groundstate, materialize_simple_update_state,
       solve_tfim_variational_groundstate,
       tfim_groundstate_observables, tfim_peps_observables,
       quantum_one_site_representative,
       direct_left_map, direct_right_map, direct_right_map_plain,
       direct_ltr_block,
       direct_right_metric, direct_finite_depth_maps,
       direct_mps_transfer_radius, direct_unit_transfer_lmps,
       direct_balanced_support_endpoints,
       direct_original_lmps_fixedpoints,
       direct_original_oriented_rails,
       direct_cardinal_regional_rails,
       direct_orthonormal_endpoints,
       QuantumVUMPSBoundary,
       QuantumDirectionalBoundaries, solve_quantum_vumps_boundary,
       solve_quantum_vumps_boundaries, quantum_one_site_expectation,
       tfim_one_site_observables, quantum_vumps_ltr_objects,
       quantum_vumps_projector_objects, RVBD3Model, rvb_d3_peps,
       rvb_d3_dense_tensor,
       quantum_vumps_sixr_rails, solve_quantum_vumps_sixr

end
