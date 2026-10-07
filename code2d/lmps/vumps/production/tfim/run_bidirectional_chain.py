#!/usr/bin/env python3
"""Resume a Slurm-owned, single-direction TFIM continuation chain."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError('Immutable configuration mismatch: ' + str(path))
    else:
        atomic_json(path, value)


def code_fingerprint(package):
    paths = [package / n for n in ('Project.toml', 'Manifest.toml')]
    for folder in ('src', 'production/tfim'):
        paths += [p for p in (package / folder).rglob('*')
                  if p.is_file() and p.suffix in ('.jl', '.py')]
    records = {str(p.relative_to(package)): sha(p) for p in sorted(set(paths))}
    digest = hashlib.sha256(json.dumps(records, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    return digest, records


def true(value):
    return value is True or str(value).lower() == 'true'


def remove_if_present(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def measurement_complete(marker, source_sha):
    if not marker.exists():
        return False
    try:
        r = json.loads(marker.read_text())
        if r.get('status') != 'complete' or r.get('source_sha256') != source_sha:
            return False
        return all(sha(r[p]) == r[h] for p, h in (
            ('summary_csv', 'summary_sha256'),
            ('boundary_checkpoint', 'boundary_sha256'),
            ('ctm_checkpoint', 'ctm_checkpoint_sha256')))
    except (OSError, KeyError, ValueError):
        return False


def last_csv(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('Empty required CSV: ' + str(path))
    return rows[-1]


def validate_fresh_seed(seed, specification, plan):
    if not true(specification.get('fresh')):
        raise RuntimeError('This campaign requires a freshly constructed endpoint seed')
    forbidden = {'checkpoint_path', 'historical_branch', 'sha256', 'source_h'}
    if forbidden.intersection(specification) or any(k.startswith('old_') for k in specification):
        raise RuntimeError('Historical checkpoint provenance is forbidden for fresh endpoint seeds')
    marker = Path(str(seed) + '.json')
    recipe = json.loads(marker.read_text())
    expected = dict(branch=specification['branch'], h=float(specification['h']),
                    D=int(plan['D']), epsilon=float(specification['epsilon']),
                    seed=int(specification['random_seed']))
    actual = recipe.get('recipe', recipe)
    if (actual.get('type') != 'product_plus_c4v_noise'
            or actual.get('source') != 'generated_from_scratch'
            or actual.get('campaign') != plan['campaign']
            or recipe.get('uses_old_checkpoint') is not False):
        raise RuntimeError('Seed recipe does not certify construction from scratch')
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError('Fresh seed recipe mismatch: ' + key)
    source_sha = sha(seed)
    if source_sha != recipe.get('checkpoint_sha256'):
        raise RuntimeError('Fresh seed file SHA mismatch')
    return source_sha, recipe


def validate_result(point, h, direction, source_sha, code_sha, plan, backend='cuda'):
    r = json.loads((point / 'point_result.json').read_text())
    checks = [true(r.get('terminal')), int(r['D']) == int(plan['D']),
              int(r['ctm_chi']) == int(plan['chi_opt']),
              abs(float(r['h']) - h) < 1e-10, r['branch'] == direction,
              r['source_sha256'] == source_sha, r['code_sha256'] == code_sha,
              true(r.get('ctm_converged')),
              0 <= int(r['completed_iterations']) <= plan['max_steps_per_point'],
              r['stopping_reason'] in ('gradient_converged', 'step_limit'),
              math.isfinite(float(r['projected_gradient_norm']))]
    if backend == 'cpu':
        checks += [r.get('optimization_backend') == 'cpu', true(r.get('validated_cpu')),
                   true(r.get('cpu_strict_audit_passed')),
                   r.get('cpu_gpu_agreement') == 'not_checked']
    else:
        checks += [r.get('optimization_backend', 'cuda') == 'cuda',
                   true(r.get('cpu_gpu_agreement'))]
    checkpoint = point / 'warmup_state.jls'
    checks.append(sha(checkpoint) == r['checkpoint_sha256'])
    if r['stopping_reason'] == 'gradient_converged':
        checks += [true(r['converged']),
                   float(r['projected_gradient_norm']) <= plan['ad_tolerance']]
    if r['stopping_reason'] == 'step_limit':
        checks.append(int(r['completed_iterations']) == plan['max_steps_per_point'])
    if not all(checks):
        raise RuntimeError('Invalid terminal point: ' + str(point))
    return r, checkpoint


def validate_pilot(point, source_sha, code_sha, steps, backend='cuda'):
    audit_file = 'initial_cpu_audit.csv' if backend == 'cpu' else 'initial_cpu_gpu_audit.csv'
    audit = last_csv(point / audit_file)
    bootstrap = last_csv(point / 'bootstrap.csv')
    progress = last_csv(point / 'progress.csv')
    count = int(progress.get('completed_iterations', progress.get('iteration')))
    if not true(audit['passed']) or not true(bootstrap['ctm_converged']):
        raise RuntimeError('Pilot backend audit or initial CTM check failed')
    if backend == 'cpu' and not true(audit.get('tight')):
        raise RuntimeError('CPU pilot requires an actual tight CPU audit')
    if count < steps or not all(math.isfinite(float(progress[k]))
                                for k in ('energy', 'projected_gradient_norm')):
        raise RuntimeError('Pilot progress incomplete or nonfinite')
    return dict(status='passed', accepted_updates=count, source_sha256=source_sha,
                code_sha256=code_sha, checkpoint_sha256=sha(point / 'iteration_checkpoint.jls'),
                optimization_backend=backend, audit_file=audit_file,
                ctm_check_scope='initial bootstrap; not a per-step residual claim')


class Runner:
    def __init__(self, end_time):
        self.end_time = end_time
        self.stop_signal = None
        self.stop_at = None
        self.stop_file = None
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.handle)

    def handle(self, signum, _frame):
        self.stop_signal = signum
        self.stop_at = time.time()
        if self.stop_file:
            self.stop_file.touch()

    def exit_code(self):
        return 128 + self.stop_signal if self.stop_signal in (signal.SIGTERM, signal.SIGINT) else 42

    def run(self, command, env, log, stop_file):
        self.stop_file = stop_file
        remove_if_present(stop_file)
        with log.open('ab', buffering=0) as f:
            child = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            terminated = False
            while child.poll() is None:
                if time.time() >= self.end_time - 180 and not self.stop_signal:
                    self.handle(signal.SIGUSR1, None)
                if self.stop_signal and time.time() - self.stop_at > 120 and not terminated:
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = True
                if self.stop_signal and time.time() - self.stop_at > 150:
                    os.killpg(child.pid, signal.SIGKILL)
                time.sleep(1)
            self.stop_file = None
            return child.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--root', type=Path)
    ap.add_argument('--direction', choices=('increasing_h', 'decreasing_h'))
    ap.add_argument('--seed', type=Path)
    ap.add_argument('--julia', default=os.environ.get('JULIA_EXE', 'julia'))
    ap.add_argument('--backend', choices=('cpu', 'cuda'), default=os.environ.get('TFIM_BACKEND', 'cuda'))
    ap.add_argument('--wall-seconds', type=int, default=85800)
    ap.add_argument('--pilot-steps', type=int, default=2)
    ap.add_argument('--pilot-segment', type=int, default=1)
    ap.add_argument('--print-code-hash', action='store_true')
    args = ap.parse_args()
    package = args.package.resolve()
    code_sha, files = code_fingerprint(package)
    if args.print_code_hash:
        print(code_sha)
        return 0
    if not all((args.plan, args.root, args.direction, args.seed)):
        ap.error('--plan, --root, --direction and fresh output --seed are required')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical chain execution requires a real Slurm allocation')
    plan = json.loads(args.plan.read_text())
    root = args.root.resolve()
    if (root.name != 'D' + str(plan['D']) or root.parent.name != plan['campaign']
            or not plan['campaign'].startswith('qr_bidirectional_fresh_')):
        raise RuntimeError('Output must be the separate fresh campaign/D# in the plan')
    if os.environ.get('TFIM_EXPECTED_CODE_SHA256', code_sha) != code_sha:
        raise RuntimeError('Deployed source fingerprint differs from expected code')
    branch = root / args.direction
    branch.mkdir(parents=True, exist_ok=True)
    lock = (branch / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seed = args.seed.resolve()
    if os.path.commonpath((str(seed), str(root.parent))) != str(root.parent):
        raise RuntimeError('Fresh seed output must be inside this new campaign')
    seed_spec = plan['seeds'][args.direction]
    if not true(seed_spec.get('fresh')) or any(k in seed_spec for k in ('checkpoint_path', 'historical_branch', 'sha256', 'source_h')):
        raise RuntimeError('Only a from-scratch seed specification is allowed')
    if abs(float(seed_spec['h']) - float(plan['directions'][args.direction][0])) > 1e-12:
        raise RuntimeError('Fresh endpoint h must equal first scan point')
    runner = Runner(time.time() + args.wall_seconds)
    julia = [args.julia, '-O3', '--startup-file=no', '--compiled-modules=existing',
             '--project=' + str(package)]
    seed.parent.mkdir(parents=True, exist_ok=True)
    if not seed.exists() and not Path(str(seed) + '.json').exists():
        command = julia + [str(package / 'production/tfim/make_bidirectional_seed.jl'),
                           '--branch', seed_spec['branch'], '--output', str(seed),
                           '--h', str(seed_spec['h']), '--seed', str(seed_spec['random_seed']),
                           '--epsilon', str(seed_spec['epsilon'])]
        seed_env = os.environ.copy()
        for key in ('TFIM_SOURCE_STATE', 'TFIM_SOURCE_SHA256', 'TFIM_SOURCE_H'):
            seed_env.pop(key, None)
        code = runner.run(command, seed_env, branch / 'fresh_seed_generation.log',
                          branch / 'seed_generation_stop')
        if runner.stop_signal:
            return runner.exit_code()
        if code != 0:
            raise RuntimeError('Fresh seed generation failed with exit ' + str(code))
    source_sha, seed_recipe = validate_fresh_seed(seed, seed_spec, plan)
    if seed_recipe.get('script_sha256') != sha(package / 'production/tfim/make_bidirectional_seed.jl'):
        raise RuntimeError('Fresh seed generator version differs from pinned source')
    config = dict(plan=plan, direction=args.direction, seed_path=str(seed),
                  seed_sha256=source_sha, code_sha256=code_sha,
                  seed_recipe=seed_recipe,
                  optimization_backend=args.backend,
                  code_revision=os.environ.get('TFIM_CODE_REVISION', 'unspecified'))
    immutable_json(branch / 'scan_config.json', config)
    immutable_json(branch / 'source_manifest.json', dict(code_sha256=code_sha, files=files))
    source = seed
    results = []
    pending = []
    job = os.environ['SLURM_JOB_ID']
    pilot_path = branch / 'pilot_validation.json'
    driver_name = ('run_gs_qr_bidirectional_cpu_point.jl' if args.backend == 'cpu'
                   else 'run_gs_qr_bidirectional_point.jl')
    driver = package / 'production/tfim' / driver_name
    measurement = package / 'production/tfim/measure_bidirectional_point.jl'
    for index, field in enumerate(plan['directions'][args.direction]):
        if runner.stop_signal or time.time() >= runner.end_time - 300:
            return runner.exit_code()
        h = float(field)
        point = branch / ('h%.8f' % h)
        point.mkdir(exist_ok=True)
        parent = results[-1] if results else None
        immutable_json(point / 'input_provenance.json', dict(
            D=plan['D'], h=h, chi_opt=plan['chi_opt'], direction=args.direction,
            source_checkpoint=str(source), source_sha256=source_sha,
            parent_h=parent['h'] if parent else seed_spec['h'],
            parent_converged=parent['converged'] if parent else None,
            parent_stopping_reason=parent['stopping_reason'] if parent else 'fresh_product_plus_c4v_noise',
            code_sha256=code_sha))
        env = os.environ.copy()
        env.update(TFIM_SOURCE_STATE=str(source), TFIM_SOURCE_SHA256=source_sha,
                   TFIM_POINT_ROOT=str(point), TFIM_H='%.8f' % h, TFIM_BRANCH=args.direction,
                   TFIM_D=str(plan['D']), TFIM_CTM_CHI=str(plan['chi_opt']),
                   TFIM_BACKEND=args.backend,
                   TFIM_AD_TOLERANCE=str(plan['ad_tolerance']),
                   TFIM_SOURCE_H=str(parent['h'] if parent else seed_spec['h']),
                   TFIM_BASE_GIT_COMMIT=plan['base_commit'],
                   TFIM_MAX_STEPS=str(plan['max_steps_per_point']), TFIM_CODE_SHA256=code_sha,
                   TFIM_STOP_FILE=str(point / 'stop_after_step'))
        attempts = 0
        previous_count = None
        while not (point / 'point_result.json').exists():
            pilot = index == 0 and args.pilot_steps > 0 and not pilot_path.exists()
            env['TFIM_SEGMENT_STEPS'] = str(args.pilot_segment if pilot else plan['max_steps_per_point'])
            env['TFIM_RUN_SECONDS'] = str(max(1, int(runner.end_time - time.time() - 240)))
            atomic_json(branch / 'chain_status.json', dict(status='optimizing', h=h,
                index=index, completed_points=len(results), direction=args.direction,
                slurm_job_id=job, host=socket.gethostname(), code_sha256=code_sha))
            log = point / ('optimizer_%s_%d_%d.log' % (job, int(time.time()), attempts))
            code = runner.run(julia + [str(driver)], env, log, point / 'stop_after_step')
            if runner.stop_signal:
                return runner.exit_code()
            if code not in (0, 42):
                raise RuntimeError('Optimizer failed with exit %s: %s' % (code, log))
            if code == 0 and not (point / 'point_result.json').exists():
                raise RuntimeError('Optimizer exit 0 without terminal point_result.json')
            if (point / 'point_result.json').exists():
                break
            progress = last_csv(point / 'progress.csv')
            count = int(progress.get('completed_iterations', progress.get('iteration')))
            if count == previous_count:
                raise RuntimeError('Resumed optimizer made no accepted-update progress')
            previous_count = count
            attempts += 1
            if pilot and count >= args.pilot_steps:
                atomic_json(pilot_path, validate_pilot(point, source_sha, code_sha, args.pilot_steps, args.backend))
        result, checkpoint = validate_result(point, h, args.direction, source_sha, code_sha, plan, args.backend)
        if index == 0 and not pilot_path.exists():
            atomic_json(pilot_path, dict(status='passed_via_validated_terminal_endpoint',
                        code_sha256=code_sha, checkpoint_sha256=result['checkpoint_sha256']))
        # A step-limit endpoint remains explicitly unconverged but is the next source by policy.
        results.append(result)
        measure_dir = point / 'measurement'
        measured = measure_dir / 'result.json'
        measurement_ok = measurement_complete(measured, result['checkpoint_sha256'])
        if not measurement_ok:
            log = point / ('measurement_%s_%d.log' % (job, int(time.time())))
            code = runner.run(julia + [str(measurement), '--point-dir', str(point),
                              '--expected-h', str(h), '--expected-branch', args.direction],
                              env, log, point / 'stop_after_step')
            measurement_ok = code == 0 and measurement_complete(measured, result['checkpoint_sha256'])
            if not measurement_ok:
                pending.append(str(point))
                atomic_json(point / 'measurement_pending.json', dict(status='pending', exit_code=code,
                            source_sha256=result['checkpoint_sha256'], log=str(log)))
            else:
                remove_if_present(point / 'measurement_pending.json')
        source, source_sha = checkpoint, result['checkpoint_sha256']
        atomic_json(branch / 'chain_status.json', dict(status='continuing', completed_points=len(results),
                    total_points=len(plan['grid']), unconverged_points=[r['h'] for r in results if not true(r['converged'])],
                    measurement_pending=pending, results=results, code_sha256=code_sha))
        if runner.stop_signal:
            return runner.exit_code()
    atomic_json(branch / 'chain_status.json', dict(
        status='complete' if not pending else 'optimization_complete_measurement_pending',
        completed_points=len(results), total_points=len(plan['grid']), results=results,
        measurement_pending=pending, code_sha256=code_sha))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('CHAIN_ERROR:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
