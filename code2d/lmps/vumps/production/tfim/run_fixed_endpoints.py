#!/usr/bin/env python3
"""Independent TFIM fields from a frozen, freshly optimized endpoint.

Deploy outside the immutable physics package: resuming the unfinished endpoint
must retain its original package fingerprint and Julia configuration hash.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time


def load_helpers(package):
    path = package / 'production/tfim/run_bidirectional_chain.py'
    spec = importlib.util.spec_from_file_location('frozen_chain_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point_environment(base_env, plan, point, h, branch, source, source_sha, source_h):
    """The fixed source is explicit; previous field results never enter here."""
    env = base_env.copy()
    env.update(TFIM_SOURCE_STATE=str(source), TFIM_SOURCE_SHA256=source_sha,
               TFIM_SOURCE_H=str(source_h), TFIM_POINT_ROOT=str(point),
               TFIM_H='%.8f' % h, TFIM_BRANCH=branch, TFIM_D=str(plan['D']),
               TFIM_CTM_CHI=str(plan['chi_opt']), TFIM_BACKEND='cpu',
               TFIM_AD_TOLERANCE=str(plan['ad_tolerance']),
               TFIM_MAX_STEPS=str(plan['max_steps_per_point']),
               TFIM_SEGMENT_STEPS=str(plan['max_steps_per_point']),
               TFIM_CODE_SHA256=plan['physics_code_sha256'],
               TFIM_CODE_REVISION=plan['physics_revision'],
               TFIM_BASE_GIT_COMMIT=plan['base_commit'],
               TFIM_CTM_TOLERANCE='1e-9', TFIM_GRADIENT_TOLERANCE='2e-7',
               TFIM_BASE_SEED='20261006', TFIM_STOP_FILE=str(point / 'stop_after_step'))
    return env


class Scan:
    def __init__(self, args, plan, helper):
        self.args, self.plan, self.c = args, plan, helper
        self.runner = helper.Runner(time.time() + args.wall_seconds)
        self.julia = [args.julia, '-O3', '--startup-file=no', '--compiled-modules=existing',
                      '--project=' + str(args.package)]
        self.job = os.environ['SLURM_JOB_ID']
        self.driver = args.package / 'production/tfim/run_gs_qr_bidirectional_cpu_point.jl'
        self.measure_driver = args.package / 'production/tfim/measure_bidirectional_point.jl'

    def check_stop(self):
        if self.runner.stop_signal or time.time() >= self.runner.end_time - 300:
            raise SystemExit(self.runner.exit_code())

    def optimize(self, point, h, branch, source, source_sha, source_h):
        point.mkdir(parents=True, exist_ok=True)
        env = point_environment(os.environ, self.plan, point, h, branch,
                                source, source_sha, source_h)
        previous_count = None
        while not (point / 'point_result.json').exists():
            self.check_stop()
            env['TFIM_RUN_SECONDS'] = str(max(1, int(self.runner.end_time - time.time() - 240)))
            log = point / ('optimizer_%s_%d.log' % (self.job, time.time_ns()))
            code = self.runner.run(self.julia + [str(self.driver)], env, log, point / 'stop_after_step')
            self.check_stop()
            if code not in (0, 42):
                raise RuntimeError('Optimizer failed: %s; exit=%s' % (log, code))
            if (point / 'point_result.json').exists():
                break
            if code == 0:
                raise RuntimeError('Optimizer exited without terminal audit: ' + str(point))
            count = int(self.c.last_csv(point / 'progress.csv')['completed_iterations'])
            if count == previous_count:
                raise RuntimeError('Optimizer restart made no progress: ' + str(point))
            previous_count = count
        return self.c.validate_result(point, h, branch, source_sha,
                                      self.plan['physics_code_sha256'], self.plan, 'cpu')

    def measure(self, point, h, branch, source_sha):
        marker = point / 'measurement/result.json'
        if not self.c.measurement_complete(marker, source_sha):
            self.check_stop()
            log = point / ('measurement_%s_%d.log' % (self.job, time.time_ns()))
            env = os.environ.copy()
            env['TFIM_CODE_REVISION'] = self.plan['physics_revision']
            code = self.runner.run(self.julia + [str(self.measure_driver), '--point-dir', str(point),
                '--expected-h', str(h), '--expected-branch', branch], env, log, point / 'stop_after_step')
            self.check_stop()
            if code != 0 or not self.c.measurement_complete(marker, source_sha):
                self.c.atomic_json(point / 'measurement_pending.json', dict(
                    status='pending', exit_code=code, log=str(log), source_sha256=source_sha))
                return False
        self.c.remove_if_present(point / 'measurement_pending.json')
        return True

    def endpoint(self, spec):
        """Resume only the original endpoint; never invoke the old chain loop."""
        point = Path(spec['point']).resolve()
        with (point.parent / 'controller.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = json.loads((point.parent / 'scan_config.json').read_text())
            original_plan = config['plan']
            if config['code_sha256'] != self.plan['physics_code_sha256']:
                raise RuntimeError('Original endpoint physics fingerprint differs')
            for key in ('D', 'chi_opt', 'ad_tolerance', 'max_steps_per_point', 'base_commit'):
                if original_plan[key] != self.plan[key]:
                    raise RuntimeError('Endpoint configuration differs: ' + key)
            if config['direction'] != spec['original_branch']:
                raise RuntimeError('Endpoint branch does not match saved fresh-seed configuration')
            recipe_spec = original_plan['seeds'][spec['original_branch']]
            if float(recipe_spec['h']) != float(spec['h']):
                raise RuntimeError('Only the fresh endpoint can be resumed as an anchor')
            seed = Path(config['seed_path'])
            seed_sha, recipe = self.c.validate_fresh_seed(seed, recipe_spec, original_plan)
            if seed_sha != config['seed_sha256'] or recipe != config['seed_recipe']:
                raise RuntimeError('Original endpoint seed provenance changed')
            result, checkpoint = self.optimize(point, float(spec['h']), spec['original_branch'],
                                              seed, seed_sha, float(spec['h']))
            measured = self.measure(point, float(spec['h']), spec['original_branch'], result['checkpoint_sha256'])
        return result, checkpoint, measured

    def run(self):
        args, plan, c = self.args, self.plan, self.c
        spec = plan['endpoints'][args.endpoint]
        branch = args.root / spec['directory']
        branch.mkdir(parents=True, exist_ok=True)
        with (branch / 'controller.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            c.immutable_json(branch / 'scan_config.json', dict(plan=plan, endpoint=args.endpoint,
                policy='fixed_endpoint_independent_fields', controller_sha256=c.sha(__file__),
                controller_revision=os.environ['TFIM_CONTROLLER_REVISION']))
            status_path = branch / 'scan_status.json'
            c.atomic_json(status_path, dict(status='preparing_endpoint', h=spec['h'],
                                          slurm_job_id=self.job, host=socket.gethostname()))
            endpoint_result, original_checkpoint, endpoint_measured = self.endpoint(spec)
            fixed_sha = endpoint_result['checkpoint_sha256']
            fixed_h = float(spec['h'])
            fixed_source = args.root.parent / 'seeds' / ('endpoint_h%.8f.jls' % fixed_h)
            fixed_source.parent.mkdir(parents=True, exist_ok=True)
            if not fixed_source.exists():
                temporary = fixed_source.with_name(fixed_source.name + '.tmp.' + self.job)
                shutil.copyfile(original_checkpoint, temporary)
                if c.sha(temporary) != fixed_sha:
                    raise RuntimeError('Endpoint snapshot copy changed content')
                os.replace(temporary, fixed_source)
            if c.sha(fixed_source) != fixed_sha:
                raise RuntimeError('Frozen endpoint checkpoint changed')
            provenance = dict(policy='fixed_endpoint_independent_fields', source_h=fixed_h,
                source_checkpoint=str(fixed_source), source_sha256=fixed_sha,
                original_checkpoint=str(original_checkpoint), original_point=spec['point'],
                endpoint_result=endpoint_result, original_fresh_seed=True,
                physics_code_sha256=plan['physics_code_sha256'], controller_sha256=c.sha(__file__))
            c.immutable_json(Path(str(fixed_source) + '.json'), provenance)
            c.immutable_json(branch / 'fixed_source.json', provenance)
            results, pending = [], []
            for h in spec['fields']:
                self.check_stop()
                h = float(h)
                point = branch / ('h%.8f' % h)
                point.mkdir(exist_ok=True)
                if c.sha(fixed_source) != fixed_sha:
                    raise RuntimeError('Fixed source no longer matches pinned endpoint')
                input_record = dict(
                    D=plan['D'], h=h, chi_opt=plan['chi_opt'], branch=spec['branch_label'],
                    initialization='fixed_endpoint', source_checkpoint=str(fixed_source),
                    source_sha256=fixed_sha, source_h=fixed_h,
                    source_converged=endpoint_result['converged'],
                    source_stopping_reason=endpoint_result['stopping_reason'],
                    previous_field_used=False, physics_code_sha256=plan['physics_code_sha256'],
                    controller_sha256=c.sha(__file__))
                if h == fixed_h:
                    input_record.update(initialization='reference_to_fresh_endpoint',
                        role='endpoint_reference',
                        actual_optimization_input_checkpoint=endpoint_result['source_state'],
                        actual_optimization_input_sha256=endpoint_result['source_sha256'])
                c.immutable_json(point / 'input_provenance.json', input_record)
                c.atomic_json(status_path, dict(status='optimizing', h=h, fixed_source_h=fixed_h,
                    fixed_source_sha256=fixed_sha, completed_points=len(results),
                    total_points=len(spec['fields']), slurm_job_id=self.job, results=results,
                    measurement_pending=pending, host=socket.gethostname()))
                if h == fixed_h:
                    # Reuse the audited anchor at its own h without another optimization.
                    c.immutable_json(point / 'endpoint_reference.json', dict(
                        **provenance, role='endpoint_reused_at_own_field',
                        optimization_point=spec['point'],
                        measurement_result=str(Path(spec['point']) / 'measurement/result.json')))
                    result, measured = endpoint_result, endpoint_measured
                    actual_point = Path(spec['point'])
                else:
                    result, checkpoint = self.optimize(point, h, spec['branch_label'],
                                                       fixed_source, fixed_sha, fixed_h)
                    measured = self.measure(point, h, spec['branch_label'], result['checkpoint_sha256'])
                    actual_point = point
                if not measured:
                    pending.append(str(actual_point))
                results.append(dict(h=h, branch=spec['branch_label'],
                    initialization='fixed_endpoint', source_h=fixed_h, source_sha256=fixed_sha,
                    converged=result['converged'], stopping_reason=result['stopping_reason'],
                    completed_iterations=result['completed_iterations'],
                    projected_gradient_norm=result['projected_gradient_norm'],
                    checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
                    point=str(actual_point), measurement_complete=measured,
                    endpoint_reference=(h == fixed_h)))
                c.atomic_json(status_path, dict(status='continuing_independent_fields',
                    fixed_source_h=fixed_h, fixed_source_sha256=fixed_sha,
                    completed_points=len(results), total_points=len(spec['fields']),
                    results=results, measurement_pending=pending, slurm_job_id=self.job))
            c.atomic_json(status_path, dict(status='complete' if not pending else 'measurement_pending',
                fixed_source_h=fixed_h, fixed_source_sha256=fixed_sha, results=results,
                completed_points=len(results), total_points=len(spec['fields']),
                measurement_pending=pending, slurm_job_id=self.job))
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--endpoint', choices=('low', 'high'), required=True)
    ap.add_argument('--julia', required=True)
    ap.add_argument('--wall-seconds', type=int, default=85800)
    args = ap.parse_args()
    args.package, args.root = args.package.resolve(), args.root.resolve()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical execution requires Slurm')
    plan = json.loads(args.plan.read_text())
    if args.root.name != 'D' + str(plan['D']) or args.root.parent.name != plan['campaign']:
        raise RuntimeError('Fixed-endpoint campaign output root mismatch')
    c = load_helpers(args.package)
    if c.code_fingerprint(args.package)[0] != plan['physics_code_sha256']:
        raise RuntimeError('Frozen physics package hash mismatch')
    if c.sha(__file__) != os.environ['TFIM_CONTROLLER_SHA256']:
        raise RuntimeError('Standalone controller hash mismatch')
    return Scan(args, plan, c).run()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('FIXED_ENDPOINT_ERROR:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
