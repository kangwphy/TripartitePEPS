# Fresh bidirectional endpoint seeds

`make_bidirectional_seed.jl` constructs both D3 endpoint PEPS from scratch, with no checkpoint input and no deserialization. The old GS campaigns are not read or modified. Numerical construction runs only inside a Slurm allocation. `--output` must be beneath a `qr_bidirectional_fresh_20261006` path component.

Controller commands inside the allocation are:

```sh
julia --project=/path/to/lmps/vumps make_bidirectional_seed.jl \
  --D 3 --branch ordered --h 3.038 --seed 2026100601 --epsilon .05 \
  --output /path/to/qr_bidirectional_fresh_20261006/seeds/increasing_h.jls
julia --project=/path/to/lmps/vumps make_bidirectional_seed.jl \
  --D 3 --branch disordered --h 3.056 --seed 2026100602 --epsilon .05 \
  --output /path/to/qr_bidirectional_fresh_20261006/seeds/decreasing_h.jls
```

In the canonical physical basis, Z=diag(1,-1) and X has off-diagonal entries 1. The low-field product tensor has physical vector (1,0); the high-field tensor has vector (1,1)/sqrt(2). Only virtual indices (north,east,south,west)=(1,1,1,1) are populated in these product cores, inside full virtual dimension D3.

The canonical `tfim_initial_peps` creates a seeded ComplexF64 Gaussian tensor carrier with the correct TensorKit spaces. Its full-D tensor is projected with `PEPSKit.RotateReflect()`. This projector includes conjugation and has a real-linear fixed manifold. The generator removes `real(dot(P,N))/norm(P)^2` times the real product core, applies C4v again, then scales the noise by a positive real scalar so `norm(noise)/norm(product)=0.05`. Finally it normalizes `P+noise`. No physical spin-flip/Z2 constraint is imposed. The same relative norm recipe is used for both branches; their independent RNG seeds are fixed above.

The JLS payload is raw PEPS metadata, not a solved GS: `peps`, `D`, `h`, `config`, `initialization_info`, `source="generated_from_scratch"`, and `uses_old_checkpoint=false`. It contains no CTM environment. The GPU solver creates a fresh chi48 environment and measures the initial observables before optimization.

The adjacent `<output>.json` has the flat recipe fields `type="product_plus_c4v_noise"`, branch, h, D, epsilon, seed, source, uses_old_checkpoint, checkpoint path/SHA and dense-tensor SHA. It records achieved noise norm, product overlap, C4v residual, and each virtual-leg unfolding's minimum/maximum singular value and condition number. Recorded mx/mz=(0,1) or (1,0) are explicitly **analytic unperturbed product-core labels**; no claim is made about the perturbed infinite PEPS magnetization before CTM contraction.

Re-running regenerates the deterministic tensor from scratch, then checks existing JSON recipe/tensor SHA and the JLS file SHA without deserializing it. A mismatch or incomplete JLS/JSON pair is rejected, and an existing artifact is never overwritten. Julia syntax parsing passed locally; actual package constructor and intended-phase checks remain part of the Slurm pilot.
