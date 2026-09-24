"""Exact KSVC fermionic PEPS and independent contraction checks.

This module is isolated from the RK/TFIM LMPS modules. All network tensors
retain TensorKit FermionParity spaces, including at environment boundaries.
"""
module FermionicPEPS

using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using MPSKit
using KrylovKit

include("state.jl")
include("fock.jl")
include("gaussian.jl")
include("replicas.jl")
include("parity_sectors.jl")
include("boundary.jl")
include("lmps.jl")
include("seam.jl")
include("rails.jl")
include("sixr.jl")
include("measurement.jl")
include("finite_regions.jl")
include("infinite_regions.jl")

export ksvc_coefficients, ksvc_projector, ksvc_tensor, ksvc_ipeps,
       finite_fock_state, finite_tensor_state, reorder_fock,
       occupation_sectors, majorana_covariance, solve_boundary, solve_boundary_period2, solve_ctm,
       boundary_number, ctm_number, gaussian_sectors, occupation_permute,
       paired_boundary_observable,
       tensor_replica_sectors, finite_tensor, particle_hole_fock,
       parent_gaussian_reference, lmps_objects, boundary_number_correlation,
       boundary_even_correlation_length,
       ksvc_density_reference, replica_parity_expansion, tensor_replica_parity_overlap,
       graded_seam_apply, fermionic_rails, regional_seam_rails,
       finite_graded_sixr, finite_occupation_sixr, seam_identity_cap,
       leading_seam_endpoint, finite_lmps_sectors, lmps_sector_entropies,
       ksvc_norm_reference, ksvc_infinite_norm_reference,
       finite_region_mps, mps_junction_geometry, finite_mps_sectors, measure_finite_fpeps_stilde,
       infinite_region_rdm, infinite_local_stilde,
       solve_seam_caps, measure_lmps_stilde, prepare_fpeps_stilde, solve_fpeps_stilde

end
