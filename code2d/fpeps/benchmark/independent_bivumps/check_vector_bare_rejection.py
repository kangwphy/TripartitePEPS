"""Check that the actual failed chi16 inputs cannot reach bare measurement.

This is a rejection test, not a successful entropy/control measurement.
Run through preempt; arguments: native_failed rotated_failed small_report out.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "preempt"
native, rotated, small, out = map(Path, sys.argv[1:])
assert not out.exists()
out.mkdir(parents=True)
adapter = Path(__file__).with_name("bare_mixed_vector_v9.py").resolve()
compile(adapter.read_text(), str(adapter), "exec")
measurement = out / "rejected_attempt"
# These paths must never be read: the real completed but nonconverged
# native chi16 report must fail before any physical or replica invocation.
command = [sys.executable, str(adapter), str(native), str(rotated),
           "UNREACHED_equation_audit", str(measurement), "UNREACHED_native_rdm",
           "UNREACHED_rotated_rdm", str(small)]
with (out / "preflight.log").open("w") as log:
    run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
assert run.returncode != 0
status = json.loads((measurement / "status.json").read_text())
assert status["error"] == "Solver did not converge.", status
assert not status["complete"] and not status["accepted_entropy"]
assert "command" not in status and not (measurement / "measurement").exists()
sources = [adapter, Path(__file__).resolve(), native / "report.toml", rotated / "report.toml", small]
report = dict(complete=True, passed=True, rejection_only=True,
              accepted_entropy=False, zero_optimization_steps=True,
              job_id=os.environ["SLURM_JOB_ID"], partition="preempt",
              expected_rejection=status["error"],
              source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
