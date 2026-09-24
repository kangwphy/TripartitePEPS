# Clean-tree entry points

The historical `README.md` and `VALIDATION.md` preserve the scientific status
and old job history. For this cleaned copy:

* Code is still graded fermionic code, independently of `lmps/vumps`.
* `environments/default/{Project,Manifest}.toml` is now a standalone, pinned
  environment. Versions were inherited from the previously used environment;
  the LMPSVUMPS package itself is not a dependency of the fermion module.
* Submit `jobs/run_preempt.sh` with `CLEAN_ROOT` set. Jobs use a private writable
  depot before the existing dependency depots, not a shared writable cache.
* `bin/run_boundary.jl`, `bin/run_finite_stilde.jl`, `bin/run_stilde.jl` record the
  **fermionic environment's** manifest hash. Choose new output paths explicitly.
* `benchmark/independent_bivumps/` retains the active research entry points and
  their dependency closure. The historical name is not a claim of equivalence
  to standard biVUMPS; its README describes limitations and failed trials.
* Consolidated data is `../../data/fermion/`; the local `data` link points to
  its physically relocated complete `store/`. Old paths are compatibility
  aliases into this tree. Finite-window acceptance and provisional
  infinite-boundary diagnostics remain separate.

No fermionic signs, parity conventions, numerical gates or formulas were
changed in the cleanup.

The missing bare/direct backend and Python adapter dependency closure has now
been copied byte-for-byte. See [BARE_ENTRYPOINTS.md](BARE_ENTRYPOINTS.md) for the
current entry points and preempt-only submission. The `run_cpu.sh` compatibility
wrapper is included because preserved adapters invoke that filename internally.
