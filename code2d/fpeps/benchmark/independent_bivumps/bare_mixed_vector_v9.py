"""Validate v9 provenance, then run the unchanged signed bare/direct adapter.

Arguments: native rotated equation_audit out native_rdm rotated_rdm small_report
Both directions must be completed, converged v9 chi16 solves, with their own
actual-input derivative control and full-step replay. The original v8 chi4
control is inherited evidence only. No reference-side shortcut is offered.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from pip._vendor import tomli

assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm.'
assert os.environ.get('SLURM_JOB_PARTITION') == 'preempt', 'preempt required.'
native, rotated, audit, out, native_rdm, rotated_rdm, small_path = map(Path, sys.argv[1:8])
assert not sys.argv[8:], 'Unknown options.'
assert not out.exists(), 'Refusing overwrite.'
out.mkdir(parents=True)
here = Path(__file__).resolve().parent
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes = {str(Path(__file__)): digest(__file__)}
status = dict(complete=False, accepted_entropy=False, partition='preempt',
              job_id=os.environ['SLURM_JOB_ID'], source_hashes=hashes,
              inherited_v8_small_control=True,
              grown_tensor_used=False, endpoint_maps_used=False)


def read(path):
    hashes[str(path)] = digest(path)
    return tomli.loads(path.read_text())


def verify(record):
    for p, h in record['source_hashes'].items():
        assert digest(p) == h, f'Stale source: {p}'
        hashes[p] = h


def save():
    temporary = out / 'status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(out / 'status.json')


def solver_checks(r, *, vector=False):
    assert r['complete'] and r['converged'], 'Solver did not converge.'
    assert r['independent_left_right'] and not r['direction_constraint']
    assert r['method'] == 'dense_real_Gauss_Newton_trust'
    assert r['field'] == 'raw_RMS_and_exact_mixed_white'
    filename = 'newton_mixed_vector_v9.jl' if vector else 'newton_mixed_center_v8.jl'
    assert r['run_sha256'] == digest(here / filename)
    assert r['final']['outer_residual'] < min(r['tolerance'], 1e-9)
    assert r['final_branch_dominant'] and r['final_center_dominant']
    assert {row['center'] for row in r['final_center_roots']} == {'AC', 'C'}
    assert all(row['modulus_rank'] == 1 and row['eigenvalue_relative_error'] < 1e-8
               for row in r['final_center_roots'])
    assert r['trust_control']['passed'] and r['trust_control']['rows']
    assert all(max(row['relative_step_error'], row['KKT_error']) < 1e-8
               for row in r['trust_control']['rows'])
    assert all(row[key]['coefficient_change'] <= 1e-10 for row in r['rows']
               for key in ('reprojection_right', 'reprojection_left'))
    verify(r)
    assert r['derivative_controls'], 'Derivative controls missing.'
    for path in r['derivative_controls']:
        c = read(Path(path))
        assert c['complete'] and c['passed'] and c['nonstationary']
        assert c['independent_left_right'] and not c['direction_constraint']
        assert c['white_identity_error'] < 1e-6 and c['real_linearity_error'] < 1e-8
        assert {(row['side'], row['quadrature']) for row in c['derivative_checks']} == {
            ('right', 'real'), ('right', 'imag'), ('left', 'real'), ('left', 'imag')}
        assert all(min(s['relative_error'] for s in row['stencils']) < 1e-4
                   for row in c['derivative_checks'])
        verify(c)
    if vector:
        assert r['chi'] == 16 and r['response_backend'] == 'exact_vector_actions_v9'
        assert r['final']['outer_residual'] < 1e-11
        assert str(small_path) == r['inherited_v8_small_control']
        assert r['source_hashes'][str(small_path)] == digest(small_path)
        replay_path = Path(r['trust_step_replay'])
        replay = read(replay_path)
        assert r['source_hashes'][str(replay_path)] == digest(replay_path)
        assert replay['complete'] and replay['passed'] and replay['chi'] == 16
        assert replay['independent_left_right'] and not replay['direction_constraint']
        assert replay['Jacobian_columns_complete'] == 768
        assert all(c['relative_error'] < 1e-6 for c in replay['comparisons'].values())
        assert max(replay[k] for k in ('right_AL_coefficient_relative_error',
                                       'left_AL_coefficient_relative_error')) < 1e-7
        assert replay['cap_response_error'] < 1e-8
        verify(replay)
        for name in ('mixed_center_vector_response.jl', 'replay_mixed_vector_step.jl',
                     'krylov_trust.jl'):
            path = str(here / name)
            assert replay['source_hashes'][path] == digest(path)
        first = next(row for row in r['rows'] if row['iteration'] == 1)
        assert first['first_update_replay_relative_error'] < 1e-6
        public_path = Path(r['inherited_v8_export_control'])
        public = read(public_path)
        assert r['source_hashes'][str(public_path)] == digest(public_path)
        assert public['complete'] and public['passed'] and public['zero_optimization_steps']
        assert 'reference' in public
        assert public['source_hashes'][str(small_path)] == digest(small_path)
        verify(public)
        for row in r['rows']:
            if 'Jacobian' in row:
                assert row['Jacobian']['cap_response_error'] < 1e-8
                assert not row['Jacobian']['tensor_rank_truncated']


try:
    small = read(small_path)
    assert small['chi'] == 4 and small['initial_perturbation'] > 0
    solver_checks(small)
    reports = []
    for side, source in (('native', native), ('rotated', rotated)):
        r = read(source / 'report.toml')
        reports.append(r)
        solver_checks(r, vector=True)
        assert r['schema_adapter_only'] and r['schema'] == 'independent_bivumps_v1'
        original_path = Path(r['source_solver_report'])
        assert digest(original_path) == r['source_solver_report_sha256']
        assert digest(source / 'source_solver_report.toml') == digest(original_path)
        original = read(original_path)
        assert all(r[k] == value for k, value in original.items()), 'Adapter changed solver report.'
        export = read(source / 'export_audit.toml')
        assert export['complete'] and export['passed'] and export['zero_optimization_steps']
        assert export['left_export_fidelity_error'] < 1e-10
        assert max(export[k]['outer_residual'] for k in ('solved_pair', 'exported_pair')) < 1e-9
        verify(export)
        assert export['source_hashes'][str(original_path)] == digest(original_path)
        for name in ('boundary_1.jls', 'boundary_3.jls', 'independent_pair.jls'):
            expected = export['source_hashes'][str(original_path.parent / name)]
            assert digest(source / name) == expected, 'Public tensor differs from audited export.'
            hashes[str(source / name)] = expected
    assert reports[0]['chi'] == reports[1]['chi']
    status['mixed_solver_preflight_passed'] = True
    status['chi'] = reports[0]['chi']
    save()
    # Delegate stored/exported-state, original equation, RDM, and all signed
    # replica/phase/length checks to the existing implementation unchanged.
    backend = here / 'bare_antiunitary_center_v6.py'
    hashes[str(backend)] = digest(backend)
    command = [sys.executable, str(backend), str(native), str(rotated), str(audit),
               str(out / 'measurement'), str(native_rdm), str(rotated_rdm)]
    status['command'] = command
    save()
    with (out / 'measurement.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result_path = out / 'measurement/status.json'
    result = json.loads(result_path.read_text())
    hashes[str(result_path)] = digest(result_path)
    assert result['complete'] and result['environment_gate_passed'] and result['character_audit_passed']
    assert not result['grown_tensor_used'] and not result['endpoint_maps_used']
    verify(result)
    assert all(digest(p) == h for p, h in hashes.items()), 'Input changed.'
    status.update(complete=True, contraction_passed=result['contraction_passed'],
                  stilde=result.get('stilde'), gaussian_accuracy_certified=False)
    status['interpretation'] = ('Checked v9 solver/replay/export provenance and unchanged physical and bare gates. '
                                'Finite-chi Gaussian accuracy remains a separate question.')
    save()
except Exception as error:
    status['error'] = str(error)
    save()
    raise
