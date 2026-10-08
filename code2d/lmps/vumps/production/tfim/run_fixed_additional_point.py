#!/usr/bin/env python3
"""Add disjoint fields to an active fixed-endpoint campaign, one Slurm task each.

The original controller and physics package stay byte-identical. This worker
only writes its own new h directory, never the original branch scan_status.
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


def validate_addition(plan, addition):
    if addition['campaign'] != plan['campaign']:
        raise RuntimeError('Addition campaign differs from original plan')
    for key in ('D', 'chi_opt', 'physics_code_sha256', 'physics_revision'):
        if addition[key] != plan[key]:
            raise RuntimeError('Addition configuration differs: ' + key)
    fields = addition['fields']
    key = lambda h: 'h%.8f' % float(h)
    if not all(math.isfinite(float(h)) for h in fields):
        raise RuntimeError('Nonfinite added field')
    if len(fields) != len({key(h) for h in fields}) or not fields:
        raise RuntimeError('Empty or duplicated added fields')
    if {key(h) for h in fields}.intersection(key(h) for h in plan['grid']):
        raise RuntimeError('Added fields overlap the active original scan')
    expected = {(endpoint, float(h)) for endpoint in plan['endpoints'] for h in fields}
    tasks = [(r['endpoint'], float(r['h'])) for r in addition['tasks']]
    if len(tasks) != len(set(tasks)) or set(tasks) != expected:
        raise RuntimeError('Addition must contain each added h exactly once per endpoint')
    for task in addition['tasks']:
        spec = plan['endpoints'][task['endpoint']]
        if task['source_h'] != spec['h'] or key(task['h']) in {key(h) for h in spec['fields']}:
            raise RuntimeError('Additional task overlaps original controller or uses wrong endpoint')


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical execution requires a Slurm allocation')
    args.package, args.root = args.package.resolve(), args.root.resolve()
    fixed = import_fixed(args.controller)
    c = fixed.load_helpers(args.package)
    plan = json.loads((args.root.parent/'campaign_plan.json').read_text())
    addition = json.loads(args.addition.read_text())
    validate_addition(plan, addition)
    if args.root.name != 'D'+str(plan['D']) or args.root.parent.name != plan['campaign']:
        raise RuntimeError('Output root differs from campaign')
    if c.code_fingerprint(args.package)[0] != plan['physics_code_sha256']:
        raise RuntimeError('Frozen physics package changed')
    if c.sha(args.controller) != addition['base_controller_sha256']:
        raise RuntimeError('Original standalone controller changed')
    if c.sha(__file__) != os.environ['TFIM_ADDITIONAL_WORKER_SHA256']:
        raise RuntimeError('Additional worker differs from submitted version')
    if not 0 <= args.task_index < len(addition['tasks']):
        raise RuntimeError('Task index out of range')
    task = addition['tasks'][args.task_index]
    spec = plan['endpoints'][task['endpoint']]
    branch = args.root/spec['directory']
    source_record = json.loads((branch/'fixed_source.json').read_text())
    source = Path(source_record['source_checkpoint']).resolve()
    source_sha = source_record['source_sha256']
    if source_sha != task['source_sha256'] or source_record['source_h'] != task['source_h']:
        raise RuntimeError('Frozen endpoint differs from addition manifest')
    if c.sha(source) != source_sha or c.sha(source_record['original_checkpoint']) != source_sha:
        raise RuntimeError('Frozen/original endpoint file hash mismatch')
    endpoint_marker = json.loads((Path(spec['point'])/'point_result.json').read_text())
    endpoint, _ = c.validate_result(Path(spec['point']), float(spec['h']), spec['original_branch'],
        endpoint_marker['source_sha256'], plan['physics_code_sha256'], plan, 'cpu')
    if endpoint['checkpoint_sha256'] != source_sha:
        raise RuntimeError('Validated endpoint is not the pinned source')
    point = branch/('h%.8f' % task['h'])
    point.mkdir(exist_ok=True)
    with (point/'additional_controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = dict(addition=addition, addition_sha256=c.sha(args.addition), task=task,
            task_index=args.task_index, worker_sha256=c.sha(__file__),
            worker_revision=os.environ['TFIM_ADDITIONAL_REVISION'],
            base_controller_sha256=addition['base_controller_sha256'])
        c.immutable_json(point/'additional_config.json', config)
        c.immutable_json(point/'input_provenance.json', dict(
            D=plan['D'], h=task['h'], chi_opt=plan['chi_opt'], branch=spec['branch_label'],
            initialization='fixed_endpoint', source_checkpoint=str(source),
            source_sha256=source_sha, source_h=spec['h'],
            source_converged=endpoint['converged'], source_stopping_reason=endpoint['stopping_reason'],
            previous_field_used=False, physics_code_sha256=plan['physics_code_sha256'],
            controller_sha256=addition['base_controller_sha256'],
            additional_worker_sha256=c.sha(__file__), addition_id=addition['id']))
        scan = fixed.Scan(args, plan, c)
        status_path = point/'additional_status.json'
        common = dict(addition_id=addition['id'], h=task['h'], endpoint_h=spec['h'],
            source_sha256=source_sha, slurm_job_id=os.environ['SLURM_JOB_ID'],
            array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), task_index=args.task_index,
            host=socket.gethostname())
        c.atomic_json(status_path, dict(common, status='optimizing'))
        result, _ = scan.optimize(point, float(task['h']), spec['branch_label'],
                                  source, source_sha, float(spec['h']))
        c.atomic_json(status_path, dict(common, status='measuring', result=result))
        measured = scan.measure(point, float(task['h']), spec['branch_label'], result['checkpoint_sha256'])
        c.atomic_json(status_path, dict(common, status='complete' if measured else 'measurement_pending',
                                       result=result, measurement_complete=measured))
        return 0 if measured else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--controller', type=Path, required=True)
    ap.add_argument('--addition', type=Path, required=True)
    ap.add_argument('--task-index', type=int, required=True)
    ap.add_argument('--julia', required=True)
    ap.add_argument('--wall-seconds', type=int, default=85800)
    return run(ap.parse_args())


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('FIXED_ADDITIONAL_ERROR:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
