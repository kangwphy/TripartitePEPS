#!/usr/bin/env python3
"""Run original-plan fields independently from one frozen endpoint.

This controller is separate from the sequential scan controller.  It is used
only for fields that have not started yet, so each Slurm task can be placed on
an exclusive node without changing the frozen physics package or endpoint.
"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys


def import_fixed(path):
    spec = importlib.util.spec_from_file_location('fixed_endpoint_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_manifest(plan, manifest):
    if manifest['campaign'] != plan['campaign']:
        raise RuntimeError('Parallel manifest campaign differs from plan')
    for key in ('D', 'chi_opt', 'physics_code_sha256', 'physics_revision'):
        if manifest[key] != plan[key]:
            raise RuntimeError('Parallel manifest differs: ' + key)
    fields = [float(h) for h in manifest['fields']]
    key = lambda h: 'h%.8f' % float(h)
    if not fields or len(fields) != len({key(h) for h in fields}):
        raise RuntimeError('Empty or duplicated parallel fields')
    if not set(key(h) for h in fields).issubset({key(h) for h in plan['grid']}):
        raise RuntimeError('Parallel fields must be original-plan fields')
    tasks = manifest['tasks']
    expected = {('low', float(h)) for h in fields}
    actual = {(t['endpoint'], float(t['h'])) for t in tasks}
    if actual != expected or len(tasks) != len(actual):
        raise RuntimeError('Parallel tasks must cover each low-endpoint field exactly once')
    spec = plan['endpoints']['low']
    if any(t['source_h'] != spec['h'] for t in tasks):
        raise RuntimeError('Parallel task has wrong source endpoint')


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical execution requires a Slurm allocation')
    args.package, args.root = args.package.resolve(), args.root.resolve()
    fixed = import_fixed(args.controller)
    c = fixed.load_helpers(args.package)
    plan = json.loads((args.root.parent / 'campaign_plan.json').read_text())
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(plan, manifest)
    if not 0 <= args.task_index < len(manifest['tasks']):
        raise RuntimeError('Task index out of range')
    if args.root.name != 'D' + str(plan['D']) or args.root.parent.name != plan['campaign']:
        raise RuntimeError('Output root differs from campaign')
    if c.code_fingerprint(args.package)[0] != plan['physics_code_sha256']:
        raise RuntimeError('Frozen physics package changed')
    if c.sha(args.controller) != manifest['base_controller_sha256']:
        raise RuntimeError('Standalone controller changed')
    task = manifest['tasks'][args.task_index]
    spec = plan['endpoints'][task['endpoint']]
    branch = args.root / spec['directory']
    source_record = json.loads((branch / 'fixed_source.json').read_text())
    source = Path(source_record['source_checkpoint']).resolve()
    source_sha = source_record['source_sha256']
    if source_sha != task['source_sha256'] or source_record['source_h'] != task['source_h']:
        raise RuntimeError('Frozen endpoint differs from manifest')
    if c.sha(source) != source_sha or c.sha(source_record['original_checkpoint']) != source_sha:
        raise RuntimeError('Frozen endpoint file hash mismatch')
    endpoint_marker = json.loads((Path(spec['point']) / 'point_result.json').read_text())
    endpoint, _ = c.validate_result(Path(spec['point']), float(spec['h']),
                                    spec['original_branch'], endpoint_marker['source_sha256'],
                                    plan['physics_code_sha256'], plan, 'cpu')
    if endpoint['checkpoint_sha256'] != source_sha:
        raise RuntimeError('Validated endpoint is not the pinned source')

    h = float(task['h'])
    point = branch / ('h%.8f' % h)
    point.mkdir(exist_ok=True)
    with (point / 'parallel_controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (point / 'point_result.json').exists():
            result, _ = c.validate_result(point, h, spec['branch_label'], source_sha,
                                          plan['physics_code_sha256'], plan, 'cpu')
            measured = c.measurement_complete(point / 'measurement/result.json',
                                              result['checkpoint_sha256'])
            return 0 if measured else 1
        provenance = dict(D=plan['D'], h=h, chi_opt=plan['chi_opt'],
            branch=spec['branch_label'], initialization='fixed_endpoint',
            source_checkpoint=str(source), source_sha256=source_sha, source_h=spec['h'],
            source_converged=endpoint['converged'],
            source_stopping_reason=endpoint['stopping_reason'], previous_field_used=False,
            physics_code_sha256=plan['physics_code_sha256'],
            controller_sha256=manifest['base_controller_sha256'],
            parallel_manifest=manifest['id'], parallel_worker_sha256=c.sha(__file__))
        provenance_path = point / 'input_provenance.json'
        if provenance_path.exists():
            # The sequential controller may have already initialized this
            # point before it was safely stopped.  Reuse its record when the
            # fixed-endpoint provenance agrees, rather than rewriting an
            # immutable file with a different controller label.
            existing = json.loads(provenance_path.read_text())
            for key in ('D', 'h', 'source_h', 'source_sha256',
                        'physics_code_sha256', 'previous_field_used'):
                if existing.get(key) != provenance[key]:
                    raise RuntimeError('Existing provenance differs: ' + key)
        else:
            c.immutable_json(provenance_path, provenance)
        common = dict(parallel_manifest=manifest['id'], h=h, endpoint_h=spec['h'],
            source_sha256=source_sha, slurm_job_id=os.environ['SLURM_JOB_ID'],
            array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), task_index=args.task_index,
            host=socket.gethostname())
        status = point / 'parallel_status.json'
        c.atomic_json(status, dict(common, status='optimizing'))
        # A prior Slurm signal may have left this per-point stop marker behind.
        (point / 'stop_after_step').unlink(missing_ok=True)
        scan = fixed.Scan(args, plan, c)
        result, _ = scan.optimize(point, h, spec['branch_label'], source, source_sha, spec['h'])
        c.atomic_json(status, dict(common, status='measuring', result=result))
        measured = scan.measure(point, h, spec['branch_label'], result['checkpoint_sha256'])
        c.atomic_json(status, dict(common, status='complete' if measured else 'measurement_pending',
                                   result=result, measurement_complete=measured))
        return 0 if measured else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--controller', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--task-index', type=int, required=True)
    ap.add_argument('--julia', required=True)
    ap.add_argument('--wall-seconds', type=int, default=85800)
    return run(ap.parse_args())


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('FIXED_PARALLEL_ERROR:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
