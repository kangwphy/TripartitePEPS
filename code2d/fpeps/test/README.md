# Fermionic tests

This directory now has one regression entry point and two installation smokes:

* `runtests.jl`: default self-contained Bell/occupation-replica/gauge tests.
* `regression/`: additional self-contained checks of CAR signs, physical sewing,
  six-R, parity, centers, finite regions and contraction ordering.
* `bare_load_smoke.jl`: loads the bare/direct backend without measuring entropy.
* `bare_python_import_smoke.py`: Python adapter syntax/dependency check only.

Pass `all` to the regression entry to select all cases, or pass case basenames
without `.jl`. The full set includes costly physical replica contractions;
passing the default subset does not imply all cases were rerun.

On Pitt CRC, export `CLEAN_ROOT` to the repository root and submit:

```bash
sbatch -M smp --partition=preempt --export=ALL,CLEAN_SMOKE=0 \
  "$CLEAN_ROOT/code2d/fpeps/jobs/run_cpu.sh" test/runtests.jl
```

One-off solver probes, fixed-checkpoint diagnostics and historical scans were
archived locally under `trash/test_cleanup_20260923/`, not deleted or published.
Their older README references are historical, not default regression commands.
Original regression files are backed up there before include-path relocation.
Bare/direct production and research-backend dependencies were not removed.

Do not use the reduced `CLEAN_SMOKE=1` (no compiled modules) configuration for
the full tensor-contraction suite: in the current Julia 1.12.6 / TensorKit
installation it raises `_repartition_body` binding errors in replica tests.
That configuration was only verified for small Bell and backend-load smokes.
Keep environment errors distinct from failed physical assertions.
