# LMPS/VUMPS tests

`runtests.jl` is the small default regression suite. The remaining seven files
are separate CPU/GPU integration, symmetry, six-R, optimization and pullback
checks. They are retained because they exercise distinct supported paths,
not because they are historical scan outputs. GPU tests require a GPU allocation.

All numerical checks on Pitt CRC must use `--partition=preempt`. Example:

```bash
sbatch -M smp --partition=preempt --export=ALL,CLEAN_SMOKE=1 \
  "$CLEAN_ROOT/code2d/lmps/vumps/jobs/run_preempt.sh" test/runtests.jl
```

Local endpoint/projector identity tests do not establish equivalence of every
endpoint network to the bare/direct construction. Preserve that distinction
when reporting test coverage.
