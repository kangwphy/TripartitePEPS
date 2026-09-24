"""Recheck saved occupation RDMs; keep physical and equation acceptance separate."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.plot_deps'))
import numpy as np
from pip._vendor import tomli

assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm.'
source, out = map(Path, sys.argv[1:3])
assert not out.exists(), 'Refusing overwrite.'
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report_file = source / 'correlations.toml'
record = tomli.loads(report_file.read_text())
assert record['complete'] and len(record['cases']) == 1
case = record['cases'][0]
assert case['complete_unique_physical_caps']
hashes = {str(report_file): digest(report_file)}
for path, expected in case['source_hashes'].items():
    assert digest(path) == expected, path
    hashes[path] = expected

# Same physical RDM tolerance as src/infinite_regions.jl. This is a
# consistency check, not a bound on finite-chi Gaussian approximation error.
tolerance = 1e-8
rows = []
for saved in case['rdms']:
    sites = saved['sites']
    paths = [source / f'rho{sites}_{part}.csv' for part in ('real', 'imag')]
    real, imag = [np.loadtxt(p, delimiter=',') for p in paths]
    for p in paths:
        hashes[str(p)] = digest(p)
    rho = real + 1j * imag
    assert rho.shape == (2**sites, 2**sites) and np.isfinite(rho).all()
    measured = dict(trace_error=float(abs(np.trace(rho) - 1)),
                    hermiticity_error=float(np.linalg.norm(rho - rho.conj().T)),
                    min_eigenvalue=float(np.linalg.eigvalsh((rho + rho.conj().T)/2).min()))
    for key, value in measured.items():
        assert abs(value - saved[key]) < 1e-12, (sites, key, value, saved[key])
    passed = (measured['trace_error'] <= tolerance and
              measured['hermiticity_error'] <= tolerance and
              measured['min_eigenvalue'] >= -tolerance)
    rows.append(dict(sites=sites, passed=passed, **measured))

for path, expected in hashes.items():
    assert digest(path) == expected, path
result = dict(complete=True, passed=all(row['passed'] for row in rows),
              tolerance=tolerance, chi=case['chi'], checks=rows,
              bivumps_residual=case['bivumps_residual'],
              source_hashes=hashes, script_sha256=digest(__file__),
              job_id=os.environ['SLURM_JOB_ID'], accepted_entropy=False,
              gaussian_accuracy_certified=False,
              interpretation='Raw occupation RDMs are checked without Hermitian projection. '
                             'The Hermitian part is used only for an eigenvalue diagnostic. '
                             'Equation convergence does not imply physical consistency.')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, indent=2))
