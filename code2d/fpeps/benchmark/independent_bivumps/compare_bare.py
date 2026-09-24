"""Crosscheck two independently solved boundaries without mixing their inputs."""
import hashlib
import json
import os
from pathlib import Path
import sys
from pip._vendor import tomli

assert os.environ.get('SLURM_JOB_ID'), 'Run comparisons through Slurm.'
first, second, out = map(Path, sys.argv[1:4])
assert not out.exists(), 'Refusing overwrite.'
hashes = {}

def read(p, fmt='toml'):
    raw = p.read_bytes()
    hashes[str(p)] = hashlib.sha256(raw).hexdigest()
    return (json.loads if fmt == 'json' else tomli.loads)(raw.decode())

results = []
for source in (first, second):
    s = read(source / 'status.json', 'json')
    assert s['complete'] and s['environment_gate_passed'] and s['contraction_passed']
    assert not s['grown_tensor_used'] and not s['endpoint_maps_used']
    for file, expected in s['source_hashes'].items():
        actual = hashlib.sha256(Path(file).read_bytes()).hexdigest()
        assert expected == actual, f'Changed measurement source: {file}'
        hashes[file] = actual
    b = read(source / 'bare/bare_result.toml')
    assert b['character_audit_passed'] and not b['character_used_for_entropy']
    results.append((s, b))
(sa, ba), (sb, bb) = results
for key in ('chi', 'chi_sectors', 'w', 'method'):
    assert ba[key] == bb[key], f'Incompatible {key}'
difference = abs(sa['stilde'] - sb['stilde'])
depths = []
for p in (first / 'bare').glob('bare_depth*.toml'):
    other = second / 'bare' / p.name
    if not other.exists():
        continue
    a, b = read(p), read(other)
    depths.append(dict(depth=a['depth'],
        stilde_difference=abs(a['stilde_real_diagnostic'] - b['stilde_real_diagnostic']),
        phase_error_first=a['phase_error'], phase_error_second=b['phase_error']))
depths.sort(key=lambda d: d['depth'])
assert depths
record = dict(complete=True, passed=difference < 1e-8 and
    max(d['stilde_difference'] for d in depths) < 1e-7,
    tolerance=1e-8, diagnostic_curve_tolerance=1e-7,
    first=str(first), second=str(second), chi=ba['chi'],
    stilde_first=sa['stilde'], stilde_second=sb['stilde'],
    stilde_difference=difference, depths=depths, source_hashes=hashes,
    physical_accuracy_certified=False,
    interpretation='Agreement of two converged finite-chi bare contractions; '
        'not a Gaussian accuracy certificate.')
out.write_text(json.dumps(record, indent=2)+'\n')
assert record['passed'], 'Independent bare measurements disagree.'
print(json.dumps({k: v for k, v in record.items() if k not in ('depths', 'source_hashes')}, indent=2))
