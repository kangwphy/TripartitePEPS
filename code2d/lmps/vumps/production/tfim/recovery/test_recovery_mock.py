from pathlib import Path
import csv,fcntl,hashlib,json,os,subprocess,sys,tempfile
source=Path('outputs/d3_fixed_endpoints_20261006/recovery/remeasure_h30515_xiang.sbatch').read_text()
body=source.split("exec python3 - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(package):
 ps=[package/n for n in ('Project.toml','Manifest.toml')]
 for folder in ('src','production/tfim'):
  ps.extend(p for p in (package/folder).rglob('*') if p.is_file() and p.suffix in ('.jl','.py'))
 rec={str(p.relative_to(package)):digest(p) for p in sorted(set(ps))}
 return hashlib.sha256(json.dumps(rec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
report={'test_scope':'Mock filesystem/controller and fake measurement executable only; no Julia/scientific work or remote changes.','cases':[]}
with tempfile.TemporaryDirectory(prefix='d3_recovery_review_') as tmp:
 for index,mode in enumerate(('success','measurement_failure','invalid_gs_flag','gs_changed','lock_busy'),1):
  root=Path(tmp)/mode; point=root/'point'; point.mkdir(parents=True); package=root/'package'
  (package/'production/tfim').mkdir(parents=True); (package/'src').mkdir(); (package/'Project.toml').write_text('mock');(package/'Manifest.toml').write_text('mock')
  script=package/'production/tfim/measure_bidirectional_point.jl'; script.write_text('frozen mock measurement')
  gs=point/'warmup_state.jls'; gs.write_bytes(b'fixed final GS mock'); gs_sha=digest(gs); script_sha=digest(script)
  input_sha='c207e7ad2a8a21b5a9547541e063ae5b5ea92ed4dd9ed7c7f2603d3b743ac811'
  marker={'checkpoint_sha256':gs_sha,'terminal':True,'h':3.0515,'branch':'from_low_endpoint','converged':False,'stopping_reason':'step_limit','completed_iterations':1000,'source_sha256':input_sha,'source_state':'fixed original source'}
  oldstatus={'result':marker,'source_sha256':input_sha,'status':'measurement_pending','measurement_complete':False,'addition_id':'original_id','host':'original_host','slurm_job_id':'original_job','task_index':2}
  (point/'point_result.json').write_text(json.dumps(marker)); (point/'additional_status.json').write_text(json.dumps(oldstatus)); (point/'input_provenance.json').write_text(json.dumps({'source_sha256':input_sha,'initialization':'fixed_endpoint'})); (point/'additional_config.json').write_text('{"original":true}')
  (point/'final_summary.csv').write_text('unchanged GS'); (point/'accepted.csv').write_text('unchanged GS'); (point/'measurement_pending.json').write_text('{"old_failure":true}')
  measurement=point/'measurement';measurement.mkdir();(measurement/'result.json').write_text('{"status":"failed"}');(measurement/'boundary_unconverged.jls').write_bytes(b'old boundary');(measurement/'ctm_state.jls').write_bytes(b'old CTM')
  prior={p.name:digest(p) for p in measurement.iterdir()}; old_status_sha=digest(point/'additional_status.json')
  fake=root/'fake_julia';fake.write_text('''#!/usr/bin/env python3
import csv,hashlib,json,os,sys
from pathlib import Path
mode=os.environ['FAKE_MODE']; point=Path(sys.argv[sys.argv.index('--point-dir')+1]); dest=point/'measurement'
if mode=='measurement_failure':
 (dest/'result.json').write_text('{"status":"failed","error":"mock failure"}')
 sys.exit(7)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_sha=sys.argv[sys.argv.index('--source-sha')+1]
input_sha='c207e7ad2a8a21b5a9547541e063ae5b5ea92ed4dd9ed7c7f2603d3b743ac811'
script=next(Path(p) for p in sys.argv if p.endswith('measure_bidirectional_point.jl'))
summary={'source_sha256':source_sha,'optimization_input_sha256':input_sha,'h':3.0515,'branch':'from_low_endpoint','gs_converged':False,'gs_stopping_reason':'step_limit','boundary_chi':48,'seed':2331450600,'vumps_tolerance':1e-9,'vumps_diagnostic_tolerance':1e-8,'vumps_galerkin':1e-10,'vumps_left_residual':1e-12,'vumps_right_residual':1e-12,'vumps_center_residual':1e-12,'ctm_convergence_error':1e-11,'peps_match_relative':0,'boundary_tensor_roundtrip_relative_error':0,'vumps_converged':True,'vumps_reported_converged':True,'ctm_converged':True,'boundary_roundtrip_verified':True,'xi_M':9.2,'xi_CTM_horizontal':9.21,'xi_CTM_vertical':9.21,'x_VUMPS':.9,'z_VUMPS':.003,'x_CTM':.9,'z_CTM':.003,'energy_CTM':-3.24,'final_checkpoint_energy':-3.24}
with (dest/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary));w.writeheader();w.writerow(summary)
for name in ['boundary_state.jls','ctm_state.jls']:(dest/name).write_bytes(('new '+name).encode())
r={'status':'complete','measurement_converged':True,'ctm_converged':True,'vumps_converged':True,'boundary_roundtrip_verified':True,'source_sha256':source_sha,'gs_converged':mode=='invalid_gs_flag','gs_stopping_reason':'step_limit','optimization_input_sha256':input_sha,'script_sha256':sha(script)}
for p,h,n in [('summary_csv','summary_sha256','summary.csv'),('boundary_checkpoint','boundary_sha256','boundary_state.jls'),('ctm_checkpoint','ctm_checkpoint_sha256','ctm_state.jls')]:r[p]=str(dest/n);r[h]=sha(dest/n)
(dest/'result.json').write_text(json.dumps(r))
if mode=='gs_changed':(point/'warmup_state.jls').write_bytes(b'changed mock GS')
''');fake.chmod(0o755)
  transformed=body.replace("POINT = Path('/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_fixed_endpoints_20261006/D3/from_h3.03800000/h3.05150000')",'POINT = Path('+repr(str(point))+')').replace("PACKAGE = Path('/home/xiang_15/chenjl/TFIM/PEPS/campaigns/qr_bidirectional_fresh_20261006/code/code2d/lmps/vumps')",'PACKAGE = Path('+repr(str(package))+')').replace("JULIA = '/home/xiang_15/chenjl/TFIM/runtime/julia-1.12.6/bin/julia'",'JULIA = '+repr(str(fake))).replace('769278599bbfdb7ab5dca7fa8aa9fbb846a8052a85231027dee34b9f936dce2e',gs_sha).replace('af80823d63ec0106258fc9006d3e8f1585986c9318329cccb2a2e15e38c5a4b9',script_sha).replace('7f315606819f56cf47cb9805728f19167ca80c2dc45484619151100387101abd',fingerprint(package))
  env=dict(os.environ,SLURM_JOB_ID=str(99000+index),SLURM_CPUS_PER_TASK='16',TFIM_CODE_REVISION='mock',FAKE_MODE=mode)
  lock=None
  if mode=='lock_busy':
   lock=(point/'additional_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  result=subprocess.run([sys.executable,'-c',transformed],env=env,capture_output=True,text=True)
  if lock:lock.close()
  auditroot=point/'recovery'/('vumps_'+str(99000+index))
  if mode=='success':
   assert result.returncode==0,result.stderr
   updated=json.loads((point/'additional_status.json').read_text());assert updated['status']=='complete' and updated['measurement_complete'] is True
   assert updated['result']==marker and updated['source_sha256']==input_sha
   assert all(updated[k]==v for k,v in oldstatus.items() if k not in ('status','measurement_complete'))
   assert digest(gs)==gs_sha
  else:
   assert result.returncode!=0,mode
   assert digest(point/'additional_status.json')==old_status_sha,mode
  if mode=='lock_busy':
   assert not auditroot.exists();assert {p.name:digest(p) for p in measurement.iterdir()}==prior
  else:
   backup=auditroot/'measurement_before';assert {p.name:digest(p) for p in backup.iterdir()}==prior
   audit=json.loads((auditroot/'job_audit.json').read_text());assert audit['original_measurement_result_sha256']==prior['result.json']
   assert audit['status']==('complete' if mode=='success' else 'failed')
  report['cases'].append({'case':mode,'passed':True,'exit_code':result.returncode})
report['all_passed']=all(r['passed'] for r in report['cases'])
out=Path('outputs/d3_fixed_endpoints_20261006/recovery/local_validation.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
