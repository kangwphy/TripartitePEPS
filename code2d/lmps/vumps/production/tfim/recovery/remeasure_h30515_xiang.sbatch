#!/bin/bash
#SBATCH --job-name=D3-30515-VUMPS-recovery
#SBATCH --partition=fat6326
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --chdir=/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006/D3/from_h3.03800000/h3.05150000
#SBATCH --output=/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006/D3/from_h3.03800000/h3.05150000/recovery_remeasure_%j.out
#SBATCH --error=/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006/D3/from_h3.03800000/h3.05150000/recovery_remeasure_%j.err

# Run only the original frozen measurement. This does not resume optimization.
set -euo pipefail
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export JULIA_DEPOT_PATH=/home/xiang_15/chenjl/TFIM/.julia_depot_tfim
export TFIM_CODE_REVISION=b2d3924c5b83aa97e518f94107ee3c7185ffa6b6
exec python3 - <<'PY'
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

POINT = Path('/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006/D3/from_h3.03800000/h3.05150000')
PACKAGE = Path('/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_bidirectional_fresh_20261006/code/code2d/lmps/vumps')
JULIA = '/home/xiang_15/chenjl/TFIM/runtime/julia-1.12.6/bin/julia'
SCRIPT = PACKAGE / 'production/tfim/measure_bidirectional_point.jl'
SOURCE_SHA = '769278599bbfdb7ab5dca7fa8aa9fbb846a8052a85231027dee34b9f936dce2e'
INPUT_SHA = 'c207e7ad2a8a21b5a9547541e063ae5b5ea92ed4dd9ed7c7f2603d3b743ac811'
SCRIPT_SHA = 'af80823d63ec0106258fc9006d3e8f1585986c9318329cccb2a2e15e38c5a4b9'
CODE_SHA = '7f315606819f56cf47cb9805728f19167ca80c2dc45484619151100387101abd'
SEED = 2331450600


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tree_hashes(root):
    return {str(path.relative_to(root)): sha(path)
            for path in sorted(root.rglob('*')) if path.is_file()}


def code_fingerprint():
    paths = [PACKAGE / name for name in ('Project.toml', 'Manifest.toml')]
    for folder in ('src', 'production/tfim'):
        paths.extend(path for path in (PACKAGE / folder).rglob('*')
                     if path.is_file() and path.suffix in ('.jl', '.py'))
    records = {str(path.relative_to(PACKAGE)): sha(path)
               for path in sorted(set(paths))}
    value = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def single_csv(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, 'Expected exactly one measurement row: ' + str(path))
    return rows[0]


def boolean(value):
    return value is True or str(value).lower() == 'true'


job = os.environ.get('SLURM_JOB_ID', '')
require(job.isdigit(), 'Run only in a real Slurm allocation')
require(int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) == 16, 'Expected 16 allocated CPU threads')
require(POINT.is_dir() and SCRIPT.is_file(), 'Missing pinned point or frozen measurement script')
lock = (POINT / 'additional_controller.lock').open('a+')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError('An additional worker/recovery owns this point; no measurement files changed')

audit_root = POINT / 'recovery' / ('vumps_' + job)
audit_root.mkdir(parents=True, exist_ok=False)
audit_path = audit_root / 'job_audit.json'
status_path = POINT / 'additional_status.json'
measurement = POINT / 'measurement'
gs = POINT / 'warmup_state.jls'
child = None
started = time.time()
audit = dict(status='checking', slurm_job_id=job, host=socket.gethostname(),
             point_dir=str(POINT), source_sha256_expected=SOURCE_SHA,
             optimization_input_sha256=INPUT_SHA, seed=SEED,
             requested_vumps_maxiter=5000, vumps_tolerance=1e-9,
             vumps_diagnostic_tolerance=1e-8, boundary_chi=48,
             gs_optimization_resumed=False, cpu_threads=16,
             script=str(SCRIPT), expected_script_sha256=SCRIPT_SHA,
             physics_revision=os.environ['TFIM_CODE_REVISION'],
             expected_physics_code_sha256=CODE_SHA,
             lock_path=str(POINT / 'additional_controller.lock'))
atomic_json(audit_path, audit)


def on_signal(signum, _frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    raise RuntimeError('Recovery interrupted by signal ' + str(signum))


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
    signal.signal(sig, on_signal)

try:
    require(sha(gs) == SOURCE_SHA, 'Final GS SHA differs from the pinned endpoint')
    require(sha(SCRIPT) == SCRIPT_SHA, 'Frozen measurement script changed')
    require(code_fingerprint() == CODE_SHA, 'Frozen physics package changed')
    original_status = json.loads(status_path.read_text())
    marker = json.loads((POINT / 'point_result.json').read_text())
    provenance = json.loads((POINT / 'input_provenance.json').read_text())
    require(marker['checkpoint_sha256'] == SOURCE_SHA and marker['terminal'] is True,
            'Point is not the pinned terminal GS')
    require(marker['h'] == 3.0515 and marker['branch'] == 'from_low_endpoint', 'Wrong target point')
    require(marker['converged'] is False and marker['stopping_reason'] == 'step_limit' and
            marker['completed_iterations'] == 1000, 'Unexpected GS convergence/budget metadata')
    require(original_status['result'] == marker, 'additional_status result is not paired with the GS marker')
    require(original_status['source_sha256'] == INPUT_SHA and
            provenance['source_sha256'] == INPUT_SHA and
            provenance['initialization'] == 'fixed_endpoint', 'Original input provenance differs')
    require(measurement.is_dir(), 'Original measurement folder missing')

    # Copy every pre-existing measurement file, including both serialized states.
    original_files = tree_hashes(measurement)
    backup = audit_root / 'measurement_before'
    shutil.copytree(measurement, backup, symlinks=False)
    require(tree_hashes(backup) == original_files, 'Measurement backup failed its SHA verification')
    immutable_names = ('point_result.json', 'input_provenance.json', 'additional_config.json',
                       'final_summary.csv', 'accepted.csv')
    immutable = {name: sha(POINT / name) for name in immutable_names if (POINT / name).is_file()}
    for name in (*immutable, 'additional_status.json', 'measurement_pending.json'):
        source = POINT / name
        if source.is_file():
            shutil.copy2(source, audit_root / name)
    original_status_sha = sha(status_path)
    audit.update(status='backed_up', source_sha256_before=sha(gs),
                 original_measurement_files=original_files,
                 original_measurement_result_sha256=original_files.get('result.json'),
                 original_additional_status_sha256=original_status_sha,
                 original_gs_result=marker, immutable_point_files=immutable,
                 measurement_backup=str(backup), backup_verified=True)
    atomic_json(audit_path, audit)

    command = [JULIA, '-O3', '--startup-file=no', '--compiled-modules=existing',
               '--project=' + str(PACKAGE), str(SCRIPT), '--point-dir', str(POINT),
               '--source-sha', SOURCE_SHA, '--expected-h', '3.0515',
               '--expected-branch', 'from_low_endpoint', '--chi', '48',
               '--seed', str(SEED), '--vumps-maxiter', '5000']
    audit.update(status='measuring', command=command)
    atomic_json(audit_path, audit)
    with (audit_root / 'measurement.log').open('wb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=os.environ.copy())
        exit_code = child.wait(timeout=6600)
    audit['measurement_exit_code'] = exit_code
    require(sha(gs) == SOURCE_SHA, 'GS checkpoint changed during measurement')
    require(exit_code == 0, 'Measurement failed; original files remain in measurement_before')

    measured = json.loads((measurement / 'result.json').read_text())
    require(measured['status'] == 'complete' and measured['measurement_converged'] is True and
            measured['ctm_converged'] is True and measured['vumps_converged'] is True and
            measured['boundary_roundtrip_verified'] is True, 'Incomplete measurement result marker')
    require(measured['source_sha256'] == SOURCE_SHA and measured['gs_converged'] is False and
            measured['gs_stopping_reason'] == 'step_limit' and
            measured['optimization_input_sha256'] == INPUT_SHA, 'Measurement changed GS/source metadata')
    require(measured['script_sha256'] == SCRIPT_SHA, 'Measurement used another script')
    for path_key, hash_key, filename in (
        ('summary_csv', 'summary_sha256', 'summary.csv'),
        ('boundary_checkpoint', 'boundary_sha256', 'boundary_state.jls'),
        ('ctm_checkpoint', 'ctm_checkpoint_sha256', 'ctm_state.jls'),
    ):
        target = Path(measured[path_key])
        require(target.resolve() == (measurement / filename).resolve(), 'Unexpected measurement output path')
        require(sha(target) == measured[hash_key], 'Measurement output hash mismatch: ' + filename)
    summary = single_csv(measurement / 'summary.csv')
    require(summary['source_sha256'] == SOURCE_SHA and summary['optimization_input_sha256'] == INPUT_SHA,
            'Summary is not paired with the pinned GS/input')
    require(float(summary['h']) == 3.0515 and summary['branch'] == 'from_low_endpoint', 'Summary labels differ')
    require(not boolean(summary['gs_converged']) and summary['gs_stopping_reason'] == 'step_limit',
            'Summary incorrectly marks GS converged')
    require(int(summary['boundary_chi']) == 48 and int(summary['seed']) == SEED, 'Boundary chi/seed changed')
    require(float(summary['vumps_tolerance']) == 1e-9 and
            float(summary['vumps_diagnostic_tolerance']) == 1e-8, 'VUMPS criteria changed')
    limits = dict(vumps_galerkin=1e-9, vumps_left_residual=1e-8,
                  vumps_right_residual=1e-8, vumps_center_residual=1e-8,
                  ctm_convergence_error=1e-9, peps_match_relative=1e-12,
                  boundary_tensor_roundtrip_relative_error=1e-14)
    for key, limit in limits.items():
        value = float(summary[key])
        require(math.isfinite(value) and 0 <= value <= limit, 'Diagnostic failed: ' + key)
    for key in ('vumps_converged', 'vumps_reported_converged', 'ctm_converged', 'boundary_roundtrip_verified'):
        require(boolean(summary[key]), 'Required diagnostic flag false: ' + key)
    for key in ('xi_M', 'xi_CTM_horizontal', 'xi_CTM_vertical', 'x_VUMPS', 'z_VUMPS',
                'x_CTM', 'z_CTM', 'energy_CTM'):
        value = float(summary[key])
        require(math.isfinite(value) and (not key.startswith('xi_') or value > 0), 'Invalid observable: ' + key)
    require(abs(float(summary['energy_CTM']) - float(summary['final_checkpoint_energy'])) <= 1e-10,
            'Measured energy differs from unchanged GS beyond the original criterion')
    require(all(sha(POINT / name) == digest for name, digest in immutable.items()),
            'An original GS/provenance file changed')
    require(sha(status_path) == original_status_sha, 'additional_status changed despite exclusive point lock')
    require(code_fingerprint() == CODE_SHA and sha(gs) == SOURCE_SHA, 'Frozen code or GS changed')

    # Preserve all original source/init and GS-result fields exactly. Completion
    # here means measurement completion, never GS gradient convergence.
    updated = copy.deepcopy(original_status)
    updated.update(status='complete', measurement_complete=True, measurement=measured,
                   measurement_recovery=dict(slurm_job_id=job, audit=str(audit_path),
                                             seed=SEED, vumps_maxiter=5000,
                                             source_sha256=SOURCE_SHA,
                                             previous_additional_status_sha256=original_status_sha))
    require(updated['result'] == marker and updated['source_sha256'] == INPUT_SHA,
            'Recovery would overwrite original optimization/source provenance')
    atomic_json(status_path, updated)
    # The historical pending marker is backed up above; it must not continue
    # to describe this point as pending once the validated measurement exists.
    pending = POINT / 'measurement_pending.json'
    if pending.exists():
        pending.unlink()
    audit.update(status='complete', source_sha256_after=sha(gs),
                 new_measurement_result_sha256=sha(measurement / 'result.json'),
                 new_additional_status_sha256=sha(status_path),
                 new_measurement_files=tree_hashes(measurement),
                 measurement=measured, gs_converged=False,
                 gs_stopping_reason='step_limit', seconds=time.time() - started)
    atomic_json(audit_path, audit)
    print('RECOVERY_COMPLETE source_sha256=' + SOURCE_SHA + ' gs_converged=false', flush=True)
except BaseException as exception:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    audit.update(status='failed', error=str(exception), seconds=time.time() - started,
                 source_sha256_after=sha(gs) if gs.is_file() else None,
                 gs_unchanged=gs.is_file() and sha(gs) == SOURCE_SHA)
    atomic_json(audit_path, audit)
    print('RECOVERY_FAILED: ' + str(exception), file=sys.stderr, flush=True)
    raise
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
PY
