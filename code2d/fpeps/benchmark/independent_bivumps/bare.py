"""Stage only new independent-pair results and call the unchanged direct backend."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from pip._vendor import tomli

assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm.'
native, rotated, audit, out = map(Path, sys.argv[1:5])
assert not out.exists(), 'Refusing overwrite.'
out.mkdir(parents=True)
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes = {}
status = dict(complete=False, schema='independent_bivumps_v1', accepted_entropy=False,
              job_id=os.environ['SLURM_JOB_ID'], source_hashes=hashes,
              script_sha256=digest(__file__),
              grown_tensor_used=False, endpoint_maps_used=False)

def read(p):
    hashes[str(p)] = digest(p)
    return tomli.loads(p.read_text())

def save():
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')

try:
    check = read(audit / 'checks.toml')
    assert check['complete'] and check['passed']
    # Require nonstationary, independently varied left/right derivative controls.
    controls = [r for r in check['checks'] if r['test'] == 'uniform_quotient_derivative']
    assert {r['side'] for r in controls} == {'left', 'right'}
    assert all(abs(complex(r['predicted_real'], r['predicted_imag'])) > 1e-5 for r in controls)
    for source in (native, rotated):
        r = read(source / 'report.toml')
        assert r['schema'] == 'independent_bivumps_v1'
        assert r['independent_left_right'] and not r['direction_constraint']
        strategy = r.get('update_strategy', '')
        assert strategy in ('', 'fixed_damping_0.1',
                            'independent_complex_analytic_Newton',
                            'independent_complex_metric_Newton',
                            'independent_complex_Krylov_Newton',
                            'independent_complex_metric_Krylov_Newton',
                            'independent_complex_ellipsoid_Krylov_Newton',
                            'independent_complex_branch_Krylov_Newton') or strategy.startswith(
                                'gauge_aligned_real_Anderson'), 'Unsupported solver controls.'
        assert r['final']['outer_residual'] < 1e-9
        if not r['converged']:
            # Optional polishing can stall just above 1e-12. Do NOT rewrite its
            # result; require a fresh saved/exported-state audit against the
            # original, explicit 1e-9 environment acceptance threshold.
            saved = read(source / 'saved_audit.toml')
            assert saved['complete'] and saved['passed'] and saved['zero_optimization_steps']
            assert saved['acceptance_tolerance'] <= 1e-9 and saved['outer_residual'] < 1e-9
            assert saved['core_sha256'] == r['core_sha256']
            for f, h in saved['source_hashes'].items():
                assert digest(f) == h
                hashes[f] = h
            status.setdefault('polishing_target_not_met', []).append(dict(
                source=str(source), requested=r['tolerance'],
                actual=saved['outer_residual'], acceptance=saved['acceptance_tolerance']))
        assert r['core_sha256'] == check['core_sha256']
        if r.get('update_strategy', '').startswith('gauge_aligned_real_Anderson'):
            controls = [p for p in source.parent.glob('anderson4_control*/report.toml')
                        if tomli.loads(p.read_text()).get('run_sha256') == r['run_sha256']
                        and tomli.loads(p.read_text()).get('memory') == r.get('memory')]
            alignments = [p for p in source.parent.glob('alignment*.toml')
                          if tomli.loads(p.read_text()).get('anderson_sha256') == r['run_sha256']]
            assert controls and alignments, 'Matching acceleration controls missing.'
            control = read(controls[0])
            alignment = read(alignments[0])
            assert control['complete'] and control['converged']
            assert control['final']['outer_residual'] < 1e-9
            assert alignment['complete'] and alignment['passed']
            assert alignment['anderson_sha256'] == r['run_sha256'] == control['run_sha256']
        if r.get('update_strategy') in ('independent_complex_analytic_Newton',
                                        'independent_complex_metric_Newton',
                                        'independent_complex_Krylov_Newton',
                                        'independent_complex_metric_Krylov_Newton',
                            'independent_complex_ellipsoid_Krylov_Newton',
                            'independent_complex_branch_Krylov_Newton'):
            response_path = Path(r['control']) / 'checks.toml'
            assert digest(response_path) == r['control_sha256']
            response = read(response_path)
            assert response['complete'] and response['passed']
            assert response['independent_left_right'] and not response['direction_constraint']
            if r.get('update_strategy') == 'independent_complex_branch_Krylov_Newton':
                assert response['branch_sha256'] == r['branch_sha256'] == r['response_sha256']
                assert r['final_requires_original_dominant_audit'] and r['final_branch_dominant']
                assert all(c['modulus_rank'] == 1 for c in r['final_branch_caps'].values())
                for f, h in response['source_hashes'].items():
                    assert digest(f) == h
                    hashes[f] = h
                assert any(Path(f).name == 'newton_response_fast.jl' and h == r['base_response_sha256']
                           for f, h in response['source_hashes'].items())
                if r['chi'] > 4:
                    cp = Path(r['critical_control'])
                    assert digest(cp) == r['critical_control_sha256']
                    critical = read(cp)
                    assert critical['complete'] and critical['passed']
                    assert critical['branch_sha256'] == r['branch_sha256']
                    assert critical['core_sha256'] == r['core_sha256']
                    for f, h in critical['source_hashes'].items():
                        assert digest(f) == h
                        hashes[f] = h
            else:
                assert response['response_sha256'] == r['response_sha256']
                if 'base_response_sha256' in r:
                    assert response['base_response_sha256'] == r['base_response_sha256']
            assert response['core_sha256'] == r['core_sha256']
            controls = [p for p in source.parent.glob('newton4_control*/report.toml')
                        if tomli.loads(p.read_text()).get('run_sha256') == r['run_sha256']
                        and tomli.loads(p.read_text()).get('krylov_sha256') == r.get('krylov_sha256')
                        and tomli.loads(p.read_text()).get('maxbasis') == r.get('maxbasis')
                        and tomli.loads(p.read_text()).get('ellipsoid_sha256') == r.get('ellipsoid_sha256')
                        and tomli.loads(p.read_text()).get('rawradius') == r.get('rawradius')]
            assert controls, 'Matching independent Newton convergence control missing.'
            control = read(controls[0])
            assert control['complete'] and control['converged']
            assert control['response_sha256'] == r['response_sha256']
            if 'krylov_sha256' in r:
                assert r['linear_control']['passed'] and control['linear_control']['passed']
            if 'ellipsoid_sha256' in r:
                assert r['ellipsoid_control']['passed'] and control['ellipsoid_control']['passed']
            assert control.get('base_response_sha256') == r.get('base_response_sha256')
            assert control['final']['outer_residual'] < 1e-9
        for f in ('boundary_1.jls', 'boundary_3.jls'):
            hashes[str(source / f)] = digest(source / f)
    stage = out / 'stage'
    stage.mkdir()
    for src, name in [(native / 'boundary_1.jls', 'boundary_1.jls'),
                      (native / 'boundary_3.jls', 'boundary_3.jls'),
                      (rotated / 'boundary_1.jls', 'rotated_boundary.jls')]:
        shutil.copyfile(src, stage / name)
        assert digest(stage / name) == digest(src)
    status['environment_gate_passed'] = True
    save()
    command = ['bash', 'jobs/run_cpu.sh', '--compiled-modules=existing',
               '--pkgimages=existing', '-e',
               'include("benchmark/bare_direct_scan_point.jl"); scan_bare(ARGS[1],ARGS[2];maxdepth=1024)',
               str(stage), str(out / 'bare')]
    status['command'] = command
    for f in ['benchmark/bare_direct_scan_point.jl', 'benchmark/bare_direct_core.jl']:
        hashes[f] = digest(f)
    with (out / 'bare.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = read(out / 'bare/bare_result.toml')
    assert all(digest(p) == h for p, h in hashes.items()), 'Input changed.'
    status.update(complete=True, contraction_passed=result['contraction_passed'],
                  stilde=result.get('stilde'),
                  finite_chi_contraction_passed=result['contraction_passed'],
                  gaussian_accuracy_certified=False)
    # Keep contraction acceptance and accuracy of the physical approximation separate.
    status['interpretation'] = 'Independent boundary equations and bare gates are separate from finite-chi physical accuracy.'
    save()
except Exception as error:
    status['error'] = str(error)
    save()
    raise
