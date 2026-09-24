# Independent bivariational boundary-MPS experiments

Full-parity audit (2026-09-22), jobs 11416352/11416357: the saved chi4 and
nonstationary chi16 row-sandwich channels have peripheral roots approximately
+1 and -1 in opposite parity sectors. Their pooled naive xi diverges, while
self/pair channels remain gapped. The -1 mode has numerically zero residue
in the measured normal/anomalous correlators; their decay lengths are
4.7883258552 / 69.4752899119. These are operator decay lengths, not pooled
spectral xi. Details and commands: `transfer_degeneracy_20260922.md`.
This does not certify chi16 convergence or explain its stalled update.

Current work (2026-09-21) targets chi16. `newton_block_minimax_v12.jl`
solves the two independent boundary-MPS center equations using trust steps
that minimize the largest of eight linearized residual-block norms. It is
**not a standard biVUMPS update implementation** and
does **not update CTM edges/corners**. The relation to bivariational VUMPS
is the target AC/C equations, not the iteration rule. The mixed-center
metric whitening is also distinct from a global biorthogonal MPS gauge.
One input originated in an eigCTMRG experiment, but the current iterates
are two independent MPS tensors. Historical directory names do not identify
the currently running numerical method.

Bounded follow-up (2026-09-22): `geodesic_center_response_trial_v2.jl`
tests directional second-order corrections at the completed chi16 v12
endpoint, using the already checked full Jacobian. The initial direct
second-difference trial 11416337 finished without evaluable candidates:
seven curvature checks failed and one minimax dual solve was uncertified.
Job 11416342 completed using differences of analytic first responses in the
same fixed chart, retaining all thresholds. Its best passing trial candidate
reduces the original maximum from 4.4094181550884164e-5 to
4.406674786690303e-5 (about 0.06 percent); it is not converged or physically
accepted. This is a bounded trial, not another continuous solver.
Definitions and exact preempt commands are in
`geodesic_center_trial.md`. No chi8 solve is repeated.

The v12 run 11416128 completed 24 updates with original maximum
4.4094181550884164e-5 and remains unconverged. Caps pass, but center-root
eigenvalue errors remain about 2e-6. Final diagnostic xi_MPS_R/L are about
15.82034 and xi_pair about 69.06224. Full-tangent stationarity audit 11416266
completed: the all-block gradient hull excludes zero in both fixed and full
spaces, but the Euclidean descent direction needs a step near 2.16e-9.
Step-scale diagnostic 11416279 confirms a tiny actual decrease there, below
the repair-gain margin; a regularized Jacobian metric gives a resolved but
still small decrease to 4.4079778562720285e-5. This is not promoted to another
continuous solver. No automatic continuation of v12 is submitted. Its independently frozen seventh
iterate was remeasured by 11416180: original maximum 4.445720416934819e-5,
xi_MPS_R/L approximately 15.6424, xi_pair approximately 67.9498. Physical
measurement 11416182/11416183 passes raw RDM checks but all three maximum
Gaussian correlation errors increase relative to the first minimax step.
The bounded secant-model replay 11416242 completed: at radius .02 two secant
corrections reduce the original maximum to 4.4338523251007445e-5, while the
exact tangent step at that radius increases it. All .04 candidates fail.
Independent continuous `newton_minimax_secant_v13.jl` job 11416246 reproduced
that first step exactly and saved all 12 updates, ending at original maximum
4.41365111282206e-5. The process then FAILED (exit 1) in final reporting:
`m8_roots` is undefined in its include chain. Its original files are retained;
there is no completed solver report. Independent checkpoint audit 11416310
uses `recover_secant_checkpoint.jl`, which explicitly imports the helper,
checks terminal Slurm evidence and checkpoint hashes, and performs zero
optimization updates. Its result is a separate audit, not a repaired success
flag for the failed solver. Neither run supplies an accepted entropy point.
Metric sweep 11416285 completed 20 candidates; larger steps/weaker
regularization did not improve on the earlier eta=1e-12, h=.01 candidate.
Exact Gaussian row-weight comparison 11416312 completed: both minimax and
secant trajectories worsen its relative error to about 3.47e-6. Physical
measurement 11416313 and comparison 11416327 also completed. Secant final
raw RDMs pass, but all three maximum complex Gaussian correlator errors
increase versus its initial state. No automatic longer solve is submitted
on the strength of these residual-only improvements. The accepted curve
collection is unchanged.
No chi8 optimization is being repeated. Commands and scope are in
`data/mpbp_20260920/minimax16_later_submissions.md`.

Earlier improvement experiment `newton_weak_curvature_v10.jl` uses
the same two-boundary center residual, with a measured six-mode Hessian
correction to the GN step. It is not full Newton or standard biVUMPS.
Actual chi16 best8 geometry 11415957 found negative curvature omitted by
GN; anchored derivative audit 11415985 passed. Trial 11415992 showed a
small original-maximum improvement at radius 0.04, from 4.66248e-5 to
4.65514e-5; both methods reject radius 0.08. Independent remeasurement
11416017 confirms that candidate but its center-root errors still fail.
Continuation 11416019 completed without convergence; its first-step AL differences are
below 9.5e-10 and residual relative difference is 1.33e-9 versus the trial.
It stopped at iteration 6 on the unchanged Hessian FD consistency check.
This verifies the new loop's first update, not convergence. Gaussian/RDM
measurements 11416026/11416027 and comparison 11416028 completed: all raw
RDM checks pass, but maximum complex Gaussian errors increase for all three
correlators. Refined cross diagnostic 11416042 confirms omitted curvature
terms matter. Seven-mode trial 11416054 predicts gains better but none of
its candidates improves the original maximum. Adaptive six-mode control
11416051 continues with the same FD threshold and distinct latest/best saves.
Completed trial 11416100 tested a convex maximum-block trust objective,
with a numerical primal/dual gap check; analytic control 11416066 passed.
This is a step-objective experiment, not a relaxed final gate or a converged
boundary. No new accepted entropy or chi8 run.
Commands and evidence: `data/mpbp_20260920/mixed_vector_geometry16_submissions.md`.

The isolated `newton_mixed_vector_v9.jl` now implements the same equations
and trust loop with exact vector actions for the changing center H.
Actual chi16 derivative check 11415817 and full 768-column first-step
replay 11415824 both passed; the latter reproduces the original right/left
AL coefficients within 3.05e-13 and the residual within 2.48e-10 relatively.
The full Jacobian took 130.18 seconds in that replay. Dependent longer
chi16 solve 11415830 completed 48 steps without convergence. Its best original
maximum is 4.6624800464266064e-5, at iteration 8. This is not a convergence claim;
the existing v8 chi4 regression is inherited evidence, not an exact-source
v9 test. The original solvers remain separate. See
`data/mpbp_20260920/mixed_vector_response16_submissions.md` for commands.

The frozen original chi16 iteration-1 snapshot passed raw RDM consistency
checks but did not uniformly improve Gaussian correlators: maximum normal
and connected-nn errors increased. Residual decrease alone is insufficient.
Its xi values and comparisons are recorded in
`data/mpbp_20260920/mixed_failed16_submissions.md`; these are not accepted
entropy/curve points. All new work targets chi16, without new chi8 controls.

Both v9 solves remain nonconverged; consult their live progress rather than
assuming the latest iterate is the saved best. Frozen iteration-4 xi/physical/
Gaussian checks (11415888 -> 11415890 -> 11415891) completed, as did the
archived-reference comparison plot 11415929. Its RDMs pass, but the different
Gaussian errors do not improve uniformly. Plot/CSV/JSON are in the existing
direct_chi_curve directory under independent_mixed16_vector4_physical_comparison.
Rotated iteration-1 audit 11415851 and full replay 11415852 both passed;
accelerated run 11415853 has started. Exact commands and scope are in the
vector-response submission record above. Standalone note v13 (build
11416046, 30 pages) includes the matched physical comparison; it is
a fixed snapshot, not a live status monitor.

`bare_mixed_vector_v9.py` connects completed v9 public exports to the
unchanged bare/direct backend. It adds v9 and actual-input replay provenance
checks and retains physical, saved-state, phase and length gates. Test
11415887 verifies rejection of the real failed chi16 input before measurement;
this is a rejection-only test, not a successful v9 entropy benchmark.
Full end-to-end verification awaits converged environments in both directions.

The exact-source chi4 control passes saved-state, physical correlator/RDM,
and bare/direct regression. S_tilde=0.49038830734025396; depth-by-depth
differences from the selected chi4 measurement are at most 1.14e-13.
The CTM-seeded chi8 continuation is nonstationary. The separate known-chi8
recovery test completed: the .001 perturbation returned to the reference
state with residual 1.49e-13 and per-site fidelity errors below 3e-15.
Its initial .002 perturbation lost the unique overlap cap. These completed
controls establish local recovery, not convergence at larger chi.

The active target is now the actual failed native/rotated chi16 v6
snapshots, whose original residuals are about 7.8e-5. Their v8 full
mixed-metric derivative checks passed (11415573/11415574); independent
12-step v8 solves were launched as 11415576/11415577 on preempt. No further
chi8 computation is planned. This is a launch snapshot, not a new accepted
chi16 result. Exact commands and diagnostic reports are indexed in
`../../data/mpbp_20260920/mixed_failed16_submissions.md`.
Current implementation and commands: `mixed_center_response.md` and
`../MP_BP_ARTICLE_20260920.md`; standalone note:
`PEPS/notes/fpeps_mpbp_technical_zh/main.pdf`.

Pitt CRC execution rule: every new CPU/GPU computation must use explicit
`--partition=preempt`, including all tests, analysis and plotting. Historical
submission records below are provenance, not permission to reuse an older
partition. Keep the original numerical records intact.

Historical update (2026-09-20): the v6 center-field Newton control passes the
complete chi4 chain. Bare S_tilde=0.4903883073401403, differing from the
selected value by 1.14e-13. Native/rotated chi16 pilots stop on the
symmetry-response check, with best full residuals 7.77e-5 / 7.81e-5;
neither is accepted. All three conditional measurement tasks skipped them.
The compiled v33 note is verified (24 pages). The new literature-informed
eig-CTMRG experiment is isolated in `../eigctm/`; no accepted chi24/32 data
exist yet. Exact v6 records are indexed by the runtime snapshot
`progress_after_center_v6_and_mpbp_reading.json`.

The following v5 update is historical.

Current update: v5 point-gradient Newton passes the complete chi4 benchmark.
Full/saved residuals are 8.68e-14 / 1.92e-13; raw RDMs pass; bare/direct
S_tilde differs from the previous result by 1.14e-13. The frozen Gaussian
comparison covers all three correlators at r=1...128.

Native chi16 pilot 11357767 completed 20 steps without convergence. Its raw
horizontal gradient fell about 28-fold, but the full original residual rose
from 1.0755e-4 to 2.0567e-4. The best export is the initial state, and the
conditional measurement job correctly skipped it. No selected xi or entropy
point changed. Read-only refinement 11357882 compares the full, horizontal and parallel
gradients with AC/C equations. Original 11357793 stopped on a coarse
derivative stencil; its failure is preserved and the threshold unchanged.

The initial actual-direction finite-difference/adjacent-step failures remain
recorded. Fourth/sixth-order response accuracy is reviewed explicitly in
point_gradient16_evidence.toml; no nonlinear or physical/entropy threshold was
relaxed. The rotated point-response control passed; no rotated v5 solve was
launched after the native trajectory failed to improve the full equations.

Current outcomes and exact command indexes are in the runtime data file
`point_gradient_v5_diagnosis.md`; method definitions are in the benchmark
file `point_gradient_newton.md`. The compiled v32 note is verified (23 pages),
subsection 1.26. Chi24/32 convergence and accepted measurements remain open.

The following paragraphs retain earlier intermediate states; use the update
above and final reports for current outcomes.

Current chi16 diagnosis: both twenty-step symmetry-preserving solves stopped
at the step limit, with best full residuals 8.62e-5 / 8.63e-5. Both raw RDM
checks pass. The native trial has diagnostic xi_physical=65.29 and improved
long-distance anomalous correlations, but worse short-distance density
correlations. It is not accepted for the entropy or xi curves.

Native continuation 11357325 stopped at iteration 5 on its weighted
gradient-symmetry diagnostic; rotated continuation 11357337 is still running.
An exact-state replay and reprojection audit (11357395/11357397), plus the
unchanged projected-Anderson solver initialized by the Newton state (11357408),
are the next bounded tests. Details, Gaussian errors and exact submission
manifests are in `antiunitary_newton16_diagnosis.md` in this data directory.
The compiled v29 note contains the method; the diagnosis file and final
reports provide the newer runtime outcomes. No selected curve point changed.

Latest verified result: the v2 nonstationary chi4 control and full validation
chain (11357214, 11357217--11357221) passed. Full solver residual is
2.04621e-13; saved/exported residual is 2.50305e-13. Raw 1--3 site RDM
Hermiticity errors are at most 1.12e-15. Bare/direct S_tilde is
0.49038830734025396, exactly matching the previous reported value.

Native chi16 job 11357223 and independently rotated job 11357230 are now
running. Native full residual reached 8.66e-5 by iteration 13, but is still
far above the unchanged gate. The rotated tangent and seed RDM checks have
passed. Dependent diagnostics and bare measurements are recorded in
`antiunitary_newton16_measurement_submissions.json`; no new large-chi curve
point is accepted. Compiled note v29 (20 pages) and its visual/source checks
are complete. The older paragraphs below describe intermediate outcomes.

## Current continuation: symmetry-preserving Newton

Update: chi4 control 11357208 reached full residual 1.68969e-9 but failed
its purely relative gradient-symmetry check. Frozen-state audit 11357213
passed: discarded gradient component 1.55e-14, full residual change after
reprojection 1.58e-15. Isolated v2 uses a source-checked absolute-plus-relative
symmetry check while preserving full convergence and physical/bare gates.
Job 11357214 reruns the nonstationary chi4 control; 11357217--11357221 form
its conditional stored/physical/bare chain, followed by chi16 job 11357223.
No new chi16 entropy or xi has been accepted. Exact commands and the cancelled
pending launcher replacement are in `antiunitary_newton_v2_validation_submissions.json`.
The compiled v28 PDF has been visually checked; the near-zero guard amendment
is also documented in `benchmark/independent_bivumps/antiunitary_newton.md`.


The ten-step branch/maxbasis1536 run 11356870 is complete but not converged:
best full residual 9.09883368e-5, final-iterate residual 9.98614344e-5.
All tracked caps stayed dominant at the saved iterates. The branch-gradient
norm decreased while the full residual did not improve substantially. This
route does not supply an accepted chi16 entropy point.

Both independent antiunitary tangent audits passed (11357204/11357205).
Chi4 uses an independently perturbed nonstationary pair; chi16 uses the
previous physical symmetric seed. Real tangent dimensions are 96 -> 48 and
1536 -> 768. Maximum symmetry errors are 5.32e-13 and 2.05e-11, respectively;
the worst of the best finite-difference Hessian errors is 9.17e-9 / 5.49e-6.
The audit checks gradient/Hessian invariance, commutation with the Schmidt
metric and finite-retraction symmetry; no relation between R and L is imposed.

`newton_antiunitary_branch.jl` implements the independently constrained Newton
update, with full complex residuals, final cap and center dominance retained.
Its perturbed chi4 convergence control is job 11357208; inspect its final
report before scheduling the physical/bare chain and larger-chi continuation.
`bare_antiunitary_newton.py` adds explicit checks for the new tangent controls,
solver source hashes and full physical RDM checks before the unchanged bare
backend. No selected curve values or production files have changed.

Derivation and commands: `benchmark/independent_bivumps/antiunitary_newton.md`,
`antiunitary_tangent_submissions.json`,
`antiunitary_newton_control_submission.json`.
The compiled v27 note predates this latest tangent audit; this paragraph and
the final reports are the current runtime record.

## Current outcome and next calculation

Both new chi4 validation chains are complete: the damped-center projected
solver reproduces bare/direct S_tilde exactly at the reported precision;
the maxbasis=1536 metric-Newton control differs by 1.14e-13. Stored-state,
raw 1--3 site RDM, phase and length checks remain separate requirements.

Neither new chi16 route is accepted. The projected/damped run 11356478 stops
at iteration 51, best residual 0.01584346 (the initial state). The larger
Krylov run 11356848 completes ten steps at best residual 9.0987638e-5,
compared with 9.1038278e-5 initially. Increasing inner accuracy alone did not
resolve the outer convergence problem.

The actual unprojected Anderson failure has now been reproduced and localized:
zero-step reconstruction preserves its equations, while two row-cap eigenvalue
moduli cross at alpha in [3.5e-4, 3.6e-4]. The individual complex eigenvalues
remain isolated (minimum sampled relative separation 0.03574). Selecting the
largest modulus changes the row-quotient phase and makes the residual jump.
This diagnostic concerns that captured trial, not the selected chi16 curve
point. Figure/data: `data/direct_chi_curve/independent_anderson_cap_branch16.*`.

A frozen-iterate linear audit also shows that 192 Krylov vectors are inadequate:
relative residual 0.725 at dimension 192 versus 0.000867 at 1464. At trust
radius 1e-5 the linear residual is still 0.981, so adequate inner accuracy does
not itself guarantee a useful nonlinear step.

The next bounded calculation combines the existing analytic cap continuation
with maxbasis=1536. `branch_full1536_submissions.json` records the actual-state
derivative audit, new chi4 controls, and conditional chi16 run 11356870. Its
final acceptance still requires dominant caps from the original core and all
full equations; physical and bare checks are required before promoting entropy.
The original chi4/32 branch-response controls were rechecked against source
hashes. No production files, selected curve points or acceptance gates changed.

Additional submissions: `rebuild_continuity_submissions.json`,
`anderson_cap_branch16_nfs_retry_submissions.json`,
`krylov_capacity_submission.json`, `krylov_full1536_submissions.json`.
The htc-n86 retry is an observed I/O stall, not a numerical failure; future
submissions exclude both htc-n81 and htc-n86 in addition to earlier exclusions.
The current technical update is build v27; inspect its build/visual verification
records. The records below retain earlier experiments chronologically.


## Actual-step reconstruction continuity audit

`anderson_targeted_trace.jl` is an arithmetic-preserving diagnostic copy of
`anderson_targeted.jl`; it captures each actual current pair and Anderson
step before the trial loop. `audit_rebuild_continuity.jl` then reconstructs
the saved MPS at alpha=0 and along that same direction down to alpha=1e-10.
It compares unaligned/aligned full equations, physical state overlaps, row
quotients and all four cap gaps. This tests whether the large residual at a
small trial step is a reconstruction discontinuity or occurs along the actual
state update. It does not change a solver guard or accept entropy.

The zero-step chi4 control 11356618 passes: full residual change 5.79e-17,
row quotient relative change 6.35e-16, aligned tensor change 5.01e-16.
The chi16 trace, replay check and continuity audit are recorded in
`rebuild_continuity_submissions.json`. Their outcome must be read from the
final reports; pending diagnostic files are not convergence evidence.


## Latest validation and center-step experiment

The projected chi4 solver, saved-state audit, both-direction physical RDM
checks, and bare/direct crosscheck have passed. The entropy is
0.49038830734025396, equal to the previous chi4 result; the largest difference
at all common contraction depths is 1.27e-13. Evidence:
`independent_antiunitary4_bare_crosscheck.json`.

The projected chi16 **initial guess** passes raw 1--3 site physical RDM
consistency (Hermiticity errors below 1.8e-15), but its full equation residual
is 0.01584. Its xi_pair=326.44 is not an accepted curve point. Unprojected
continuation 11356435 failed its Anderson guard after two accepted steps.
Projected continuation 11356417 failed at the second center proposal:
the overlap cap residual is 4.50e-15 but its modulus gap is 2.76e-14.
The unchanged 1e-9 uniqueness gate rejects this alignment. The corresponding
unstarted native validation jobs were cancelled; logs and progress remain.

`anderson_antiunitary_damped_v2.jl` now backtracks the AC/C retraction BEFORE
candidate gauge alignment. This is separate from the subsequent Anderson
line search, which cannot catch an earlier proposal-construction failure.
The local center step is tried at beta=1 down to 1/512; independent variables,
full complex residuals, metric gates and terminal root dominance are retained.
`last_iteration_pair.jls` preserves the actual pre-proposal state for diagnosis.
The new strategy has its own chi4 solver/alignment/saved/RDM/bare validation
chain and a four-step chi16 diagnostic; these are experiments, not accepted
large-chi results. Commands are in `antiunitary_damped_v2_submissions.json`.
The first damped variant failed while storing vector metadata in a Float64
dictionary; its logs are retained in `antiunitary_damped_v1_failure.json`.
V2 fixes that container and propagates unexpected programming errors. This
implementation failure is distinct from numerical nonconvergence.

The v2 nonstationary chi4 solver has passed (11356469): full residual
5.81e-13 and dominant AC/C roots; its saved/RDM/bare controls are still running.
The bounded chi16 diagnostic (11356471) uses beta=1/2 at the previously failing
second proposal, but residuals then rise to 0.09789 after four steps. It removes
that exception without demonstrating convergence. The longer native run is
conditional on the complete new chi4 validation chain:
`antiunitary_damped_full16_submission.json` (11356478).

Other commands and failure records:

- `antiunitary_projected16_failure.json`: exact failing stage and cap errors.
- `antiunitary_projected16_validation_submissions.json`: independent rotated
  chi16 seed/solve. The rotated solve also failed at the second full center
  proposal; unstarted dependents were cancelled. See
  `antiunitary_rotated_projected16_failure.json`.
- `antiunitary_nfs_retry_submissions.json`: the htc-n81 jobs were stuck in
  `rpc_wait_bit_killable` with negligible CPU use and were replaced on another
  node. These cancellations are infrastructure failures, not numerical ones.

All commands use Slurm and fresh output paths. Current exclusions also include
htc-n81. Existing selected curve inputs and production code are unchanged.
The current compiled note is v26 (18 pages), verified by
`note_build_verification_v26.json` and `note_visual_check_v26.json`.
Earlier version references below describe historical checks.


## Independent antiunitary experiment

The native center-tracked chi16 run from iteration zero completed 600 steps
at best full residual 0.00381939 (11356329), not converged. The independent
rotated solve (11356316) instead reproduces the complex stationary pair:
xi_pair=77.99169 and the opposite row-quotient imaginary part. Its physical
RDM Hermiticity check also fails (11356341/11356342). These are distinct
failures; neither supplies a replacement curve point.

`antiunitary_pair.jl` now checks the candidate physical antiunitary in the
original and rotated *independent* equations, including independently
perturbed nonstationary R and L. AC/C pencil covariance errors are below
1.5e-15, residual covariance below 1.1e-13; the actual complex chi16 pencil
also passes (6.9e-14). This does not import a legacy direction-tied solver.
The left map is obtained by conjugating the south antiunitary through the
actual spatial-bra map and its inverse. No relation R=move(S) is imposed.

`antiunitary_seed.jl` constructs separate Takagi charts and explicitly projects
each state to real coordinates, with full physical and virtual spaces retained.
The old chi4 identity control passes. The old chi16 coefficient change 4.4e-8
fails the unchanged 1e-8 identity threshold; that failed control is retained.
A full-rank chi16 synthetic test with known real tensors and random complex
virtual gauges passes (11356360), recovery errors below 6.3e-15.
Projection of the complex chi16 branch is explicitly a state change, not a
gauge: imaginary coefficient fraction 0.07721, per-site overlap 0.999999386.
Its residual is 0.01584, so the output is only an initial guess.

`anderson_antiunitary.jl` is a separate projected fixed-point experiment. Each
side has its own antiunitary projector and its own canonicalization; all full
complex equations and terminal AC/C dominance remain acceptance requirements.
The new strategy is deliberately unsupported by old entropy adapters.
`bare_antiunitary.py` additionally requires source-checked raw occupation RDM
consistency at 1e-8 for BOTH directions before running the unchanged bare
backend. Final physical errors against Gaussian remain a separate benchmark.

Submission records in the data folder:

- `antiunitary_pair_audit_submissions.json`: equation covariance controls.
- `antiunitary_projection_retry_submissions.json`: full-rank gauge controls
  and explicit projected chi16 seed; retains the failed identity check.
- `antiunitary_seed_solver_v2_submissions.json`: unprojected continuation and
  Newton from the projected seed, plus physical seed diagnostics.
- `antiunitary_projected_solver_control_submissions.json`: nonstationary chi4
  projected solver and known-gauge alignment controls.
- `antiunitary_projected_control_validation_submissions.json`: stored-state,
  physical RDM and full bare/direct chi4 validation chain.
- `antiunitary_projected16_submission.json`: chi16 projected solver, conditional
  on the complete chi4 validation chain. Inspect Slurm and final report flags;
  a submission or low intermediate residual does not certify completion.

From `PEPS/code2d/fpeps`, use fresh output paths, for example:

```bash
sbatch --clusters=htc --partition=htc --time=02:00:00 --mem=24G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/anderson_antiunitary.jl \
  SYMMETRIC_INDEPENDENT_SEED NEW_OUT \
  data/independent_bivumps_20260919/antiunitary_pair_random4.toml 300 0 1e-11
```

The verified PDF v24 has 17 pages; pages 16--17 describe the antiunitary
maps, independent projection diagram, limitations and actual submission files.
See `note_build_verification_v24.json` / `note_visual_check_v24.json`.
Selected entropy/xi curve inputs are unchanged.

## Actual-iterate and virtual-parity diagnostics

Two different failure mechanisms have now been separated; neither is repaired
by relaxing a residual or phase threshold.

1. `anderson_capture.jl` reproduces the chi16 failure and records the exact
   candidate/reference tensors before the exception. `alignment_failure_audit.jl`
   finds an AC root switch at iteration 24: the largest-modulus roots have
   center overlap about 1e-14, while the nearby rank-3 root has both overlaps
   above 0.99997. Both parity sectors of the rejected candidate overlap are
   nearly zero. See `captured_alignment16_audit.toml`.
2. `anderson_targeted.jl` follows the AC root nearest the current generalized
   Rayleigh quotient, and targets C at z_AC/q. It retains independent left/right
   variables and the original full residual, and additionally requires dominant
   terminal AC/C roots. Its actual-state step and chi4/alignment controls passed.
   A full proposal can increase the residual; this is not a descent guarantee.
   Chi16 continuation 11356303 now passes the full equations (9.14e-12)
   and terminal center dominance, with xi_pair=77.99169 and xi_MPS=15.17357.
   Physical validation now finds two/three-site RDM Hermiticity errors
   7.26e-5 / 1.42e-4, so it fails the existing 1e-8 physical consistency
   standard. It is not an accepted entropy point. See
   `targeted16_validation_submissions.json` and `physical_rdm_audit_targeted16.json`.
3. `seam_parity_spectrum.jl` adds an explicit spectator charge to diagnose both
   seam sectors. The new chi8 AC/BC isolated leading mode is in the odd sector;
   the selected chi8 has the same modulus in the even sector. This explains why
   recomputing the rotated direction alone did not repair the bare seam.
4. `virtual_parity_shift.jl` fuses a one-dimensional odd line into each virtual
   bond using graded tensor arithmetic. Full finite-ring amplitudes at lengths
   1--4 change by exactly the global supertrace sign -1; applying it twice restores
   +1. Actual chi4/native8/rotated8 equation checks passed. Native8 correlators
   and RDMs before/after the transformation agree to 2.6e-15 or better.
   Bare/direct still requires an independently validated cap sector and the
   full signed replica phase/length checks.

The new transform adapter is `bare_parity_shift_v2.py`. It verifies exact
transformation tests, source hashes, upstream solver controls and saved-state
audits, then runs the unchanged wide bare backend. V1 only omitted the default
strategy value for the oldest chi4 report; its failed precheck is retained.
`bare_targeted.py` additionally enforces terminal center-root dominance for the
tracked-center solver. The original production adapters are unchanged.

Actual submission records are `alignment_capture_submissions.json`,
`targeted_center_submissions.json`, `targeted_center_bare_control_submissions.json`,
`virtual_parity_shift_submissions.json`, `parity_shift_validation_submissions.json`,
and `parity_shift4_adapter_retry_submissions.json` in the experiment data folder.
Use fresh output paths. For example:

```bash
sbatch --clusters=htc --partition=htc --time=00:25:00 --mem=12G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/virtual_parity_shift.jl SOURCE NEW_OUT

sbatch --clusters=htc --partition=htc --time=00:40:00 --mem=16G \
  jobs/run_python.sh benchmark/independent_bivumps/bare_parity_shift_v2.py \
  NATIVE ROTATED data/independent_bivumps_20260919/chi4_equations_nonstationary NEW_MEASUREMENT
```

Run `audit_saved.jl` on all transformed and upstream source folders first.
Parity relabeling changes the seam sectors and must be applied consistently;
do not bypass `solve_seam_caps` when a selected sector is degenerate.
The matched chi12/16 plateau diagnosis is in
`selected12_vs_selected16_diagnosis.md`; it compares the actual selected curve
states, not the failed cold continuation. Large-chi convergence is still open.

## Current verified state

The selected curve now contains chi **4, 6, 8, 12, 16**. The authoritative
selection is `data/independent_bivumps_20260919/curve_points.json`; the
source-checked numbers are in `results.csv/json`. Chi16 uses memory-32 Anderson
boundaries, with fresh saved/exported residuals 1.24e-11 / 2.80e-11 and
bare/direct S_tilde **0.7598174580070918**. Its last-three phase error is
7.87e-10 and entropy range 1.33e-10. The stricter optional 1e-11 polishing
flags remain false. The physical approximation still has Gaussian errors.

Plots in `data/direct_chi_curve/`:
`independent_bivumps_stilde_chi.{png,pdf}` and
`independent_bivumps_stilde_lnxi.{png,pdf}`. The latter labels chi and retains
the reversals at chi6 and chi16. The rebuilt technical PDF is
`PEPS/notes/fpeps_signs_zh/independent_bivumps.pdf` (17 pages at build v24);
verified source manifest `note_build_verification_v24.json`, job 11356415.
Pages 16--17 were rendered and inspected (`note_visual_check_v24.json`);
preceding unchanged pages retain the earlier checks.
This revision includes the selected chi12/16 plateau, captured center-root
jump, tracked-center equations, graded virtual parity tensor diagram and
all successful chi4/8 bare crosschecks. The new chi16 equation solution
fails physical RDM consistency; the selected curve is unchanged.

The cached Newton solver independently recovers both chi16 warm starts.
Its separate bare crosscheck passed (11355444 / `measurement_newton16_wide`):
S_tilde=0.7598174580161867, differing by 9.095e-12 from the Anderson value.
`independent_chi16_bare_crosscheck.json` (11355458) verifies both input chains
and all shared depths, whose maximum entropy difference is 1.206e-10.
Wide and small bare
backends match exactly at chi4 (`wide_backend_control.json`). Neither backend
uses grown-tail or endpoint maps for entropy.

**Still incomplete:** cold rank expansion and larger chi. Ordinary Newton
from the new chi12 seed to chi16 stops after 20 steps at residual 0.00232846.
Native chi24 job 11355329 completed 20 steps at best residual 0.00150866,
and is not an accepted curve point. A separate
Schmidt-coordinate experiment is in `newton_metric_response.jl` and
`newton_metric.jl`: X = Xhat (C C')^(-1/2), separately for each state.
It transforms the exact gradient and Hessian, truncates no physical rank,
and keeps the original full residual gate. Its independent derivative control
and chi4 solver control precede the cold chi16 trial; commands and dependencies
are recorded in `metric_newton_submissions.json` (11355448--450).
The first two controls passed (Hessian finite differences 1.5e-9--8.3e-9;
chi4 residual 6.22e-14 after three steps). Cold chi16 completed 25 steps at
best residual 0.00189124 and failed the original gate.

The matrix-free versions are `newton_krylov.jl` (ordinary coordinates) and
`newton_metric_krylov.jl` (Schmidt coordinates), sharing `krylov_trust.jl`.
They minimize the linearized stationary-equation residual inside a Krylov
subspace, retaining both signs of Hessian curvature. The subspace limit is
192 in the current controls and runs. Hitting this limit is not a successful
linear solve: inspect `Krylov.stop_reason` and its residual trace.
The indefinite dense KKT comparison passed. Both chi4 controls converged;
the Schmidt version also passed saved-state and bare/direct checks (11355618,
11355619). `independent_chi4_metric_krylov_bare_crosscheck.json` (11355632)
verifies agreement with the original chi4 bare value to 1.14e-13.

Chi16 job 11355612 ended with `Newton_trust_step_failed`, best full residual
1.3655e-6. Its near-unit per-site overlaps with chi12 and tiny added Schmidt
weights support collapse toward the lower-rank branch; see
`cold16_vs_chi12_overlap.toml` (11355662). This failed cold start does not replace
the selected, independently verified chi16 point. A derivative audit on its
immutable saved state (11355655) failed: even the zero-step gradient changes by
1.23e-4 relatively (5.9e-10 absolutely). No tolerance was relaxed.

Original Schmidt chi32 job 11355614 also ended at `Newton_trust_step_failed`,
best residual 0.01757599. The ellipsoid variant remains a separate active test.
Ordinary chi24 continuation 11355596 completed 120 steps without improving
its initial best residual 0.00150866. Schmidt chi24 11355613 also completed
120 steps, best 0.000579326. The ellipsoid chi24 trial 11355644 completed with
best 0.000817995. These are failed experiments, not new curve points.
Recheck `sacct --clusters=htc` before relying on this
snapshot. The chi32 source is an independently compressed, perturbed row
absorption of each chi16 boundary (`Schmidt_seed32_from16_refined`); its initial
full residual is 0.0548428. This is an initial guess, not a converged environment.
Some row-absorption controls failed peripheral-spectrum or canonical/fidelity
checks; their directories and errors are retained without relaxing the guards.

A separate experiment, `newton_ellipsoid_krylov.jl`, replaces post-scaling of
the entire step with a joint trust ellipsoid:
`norm(y/radius)^2 + norm(P*Q*y/0.5)^2 <= 1`.
Here Q is the real Krylov basis and P maps Schmidt coordinates to tensor
coordinates. `ellipsoid_trust.jl` solves the reduced constrained least-squares
problem by Cholesky and SVD, without discarding physical bond directions.
It checks an independent dense KKT solution before optimizing any MPS.
Small-chi control job 11355631 passed: three steps to full residual 3.04e-12;
the dense KKT comparison has relative error below 1.1e-12. Larger chi24/32
trials 11355644/11355645 use the same initial states as the ordinary Schmidt
Krylov runs, so the effect of the trust geometry can be compared. Saved-state
and bare controls 11355646/11355647 passed. The full crosscheck 11355678 gives
chi4 entropy difference 4.55e-13. No existing solver or its acceptance
gate was changed. Exact commands are in `ellipsoid_krylov_large_submissions.json`.

To submit the matrix-free continuation (SOURCE already has the target chi):

```bash
sbatch --clusters=htc --partition=htc --time=02:00:00 --mem=24G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/newton_metric_krylov.jl \
  SOURCE NEW_OUT data/independent_bivumps_20260919/newton_response4_metric \
  120 0 1e-9 192
```

The final arguments are iteration count, initial right-state perturbation,
outer tolerance, and Krylov basis limit. Always use a new output directory.
Both spatial directions must independently pass before invoking `bare_wide.py`;
run `audit_saved.jl` on each completed output first. The entropy adapter rejects
unknown solver strategies and requires a chi4 convergence control matching the
exact solver/helper hashes and Krylov limit. Ellipsoid inputs additionally
require the matching ellipsoid-helper hash, raw radius, and dense KKT control.
Do not infer large-chi convergence from controls.

`solver_progress.py NEW_FIGURE_STEM` freezes live progress and authoritative
Slurm status, then plots full outer residuals and mixed-metric singular ratios.
Snapshot job 11355640 is in `data/direct_chi_curve/` as
`independent_bivumps_convergence_11355612_24_32.{png,pdf,json}`. It is a solver
diagnostic, not an entropy curve; iterations refer to each continuation start.

`audit_response_saved.jl SNAPSHOT.jls NEW_OUT` diagnoses derivatives on a frozen
actual iterate, rather than assuming the chi4 tests cover badly conditioned
larger states. `actual_iterate_response_submissions.json` records chi16/32 jobs
11355655/11355656 and immutable snapshot hashes. Both actual-state derivative
audits failed. At chi32, the left-real finite-difference direction changes the
quotient abruptly for steps 1e-5 to 1e-4, but remains nearby at 1e-6.
`cap_branch_probe.jl` (11355686) confirmed a largest-modulus crossing: the two
row eigenvalues are well separated in the complex plane, but their relative
modulus gap is only 3.61e-8. A +1e-5 perturbation changes the selected quotient
from approximately 3.58350-0.00029i to -3.57757+0.20619i. The eigenpairs have
small residuals, so this is root selection, not a failed inner eigensolve.
The diagnostic plot is `independent_cap_branch_chi32.{png,pdf,json}` in
`direct_chi_curve`. `branch_overlap.jl FIRST SECOND
OUT.toml` independently checks normalized per-site overlaps and Schmidt weights
beyond FIRST's parity ranks; it produces no entropy input.

`power_seed_projected.jl` is a separate initialization trial addressing the
failed chi12-to-16 SvdCut preparation: it projects the grown tensor explicitly
with the library's graded leg convention, perturbs before canonicalization,
and verifies the raw-versus-canonical normalized transfer fidelity. Preparation
11355679 passed its exact-gauge, parity-space, canonical and fidelity guards,
but its initial full residual is 1.13448, so it is not a good environment yet.
Job 11355687 (`native_chi16_projected_ellipsoid`) failed the response-cap
isolation guard at iteration 37, without improving the initial best residual.
Commands and hashes are in `projected_seed16_submissions.json`; this does
not replace a failed older initialization or imply convergence.

## Continuous eigenvalue branch experiment

`branch_response.jl` retains the exact signed contractions and derivatives,
but follows each cap eigenvalue continuously by complex distance. It records
temporary subdominance rather than treating a modulus crossing as a smooth
maximum. `bt_canonical` separately aligns each state's virtual unitary and
site phase so canonicalization preserves the raw tensors used for root labels.
This is a separate Newton solver for the independent stationary equations,
not a new name for the package's one-state VUMPS or for a postprocessing SVD.

Both derivative controls passed with matching source hashes:

- chi4, 11355692: four directional Hessian errors 6.4e-9--1.3e-8.
- frozen difficult chi32, 11355693: minimum errors over tested step sizes
  4.82e-7, 4.33e-7, 4.19e-7, 4.32e-7 for right real/imag and left real/imag.
  Some perturbations explicitly follow a rank-2 modulus root. The quotient
  remains continuous and canonicalization changes it by only 1.27e-14.

Solver control 11355703 reached full chi4 residual 3.04e-12 in three steps,
with all final roots dominant. Saved audit 11355710 and bare measurement
11355711 passed. Crosscheck 11355717,
`independent_chi4_branch_krylov_bare_crosscheck.json`, verifies source hashes,
shared lengths and entropy agreement within 1.14e-13.

`newton_branch_krylov.jl` requires the original `bv_audit` to pass AND every
tracked final cap to be dominant. A subdominant stationary solution cannot
enter the entropy adapter. For chi>4 it additionally verifies the actual-state
chi32 derivative control and all transitive source hashes. Commands are in
`branch_large_submissions.json`: 11355712 continues chi32 and 11355713 uses a
Gaussian ring chi16 initial state. The latter is only a warm start; its fresh
independent residual is 0.0248, not the legacy restricted-parameter residual.
Neither run is an accepted entropy point.

```bash
sbatch --clusters=htc --partition=htc --time=03:00:00 --mem=24G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/newton_branch_krylov.jl \
  SOURCE NEW_OUT data/independent_bivumps_20260919/branch_response4 \
  80 0 1e-9 192
```

Krylov dimension 192 is an optimization subspace limit, not MPS chi. The
Gaussian chi16 trial exhausts it while leaving relative linear residual about
0.6. `branch_basis768_submissions.json` records a separate limit-768 chi4
control (11355720), the same Gaussian initial state (11355721), and chi24
continuation (11355722, also dependent on completed 11355613). This changes no
equation or convergence tolerance. Existing running source files are unchanged.

## Dominance-constrained step experiment (under validation)

The continuous chi32 run moves onto subdominant cap roots (modulus ranks up to
3 for overlap and 6 for the row); its small tracked gradient does not reduce
the original full residual, which remains about 1. A separate trial now keeps
the same stationary equations but constrains the proposed step to remain on
the dominant side of the spectral crossings.

`dominance_response.jl` differentiates the two log-modulus gaps, one for the
row channel and one for overlap, using both independent MPS variables and
the same signed contractions. `dominance_trust.jl` minimizes reduced Newton
least squares inside the existing trust ellipsoid and two linearized gap
half-spaces. It enumerates active constraint sets and reduces each affine
slice to an SVD trust-ball solve. Random indefinite controls check feasibility,
dual signs, stationarity and complementarity, including two active constraints.
The full actual proposal must separately retain rank-1 caps and a positive
gap to the actual runner-up, not just the root used in the linearization.

`dominance_audit.jl SOURCE OUT` checks four independent real/imaginary gap
derivatives against finite differences while tracking both competing roots.
Jobs 11355737 and 11355738 use perturbed chi4 and the immutable difficult chi32
snapshot. The chi4 derivatives passed (absolute errors 3.0e-11--1.7e-10).
The actual chi32 audit also passed with source hashes verified: four directional
absolute errors 7.03e-12, 5.17e-12, 3.38e-11, 4.74e-11. These validate the
gap gradients at that difficult state, not full optimization convergence.
Standalone QP control 11355769 covered all four active sets; maximum KKT error
3.86e-15. `newton_dominance_krylov.jl` additionally requires the already passed
branch Hessian controls and all source hashes. Its chi4 control 11355759
converged in three steps to full residual 3.036e-12, with all caps dominant.
Saved-state audit 11355775 passed at 3.038e-12. Pending 11355749 was cancelled before execution to
strengthen the actual-runner-up check; its previous source and cancellation
record are retained. This experiment uses the isolated `bare_dominance.py`
adapter; original adapters and bare backends are unchanged. Followups are
recorded in `dominance_followup_submissions.json`: bare chi4 11355776 and
chi32 trial 11355777, the latter also gated by the actual chi32 derivative
audit. Crosscheck 11355790 follows the bare control. No new curve point is
accepted from these experiments.

The new bare chi4 control completed (11355776): S_tilde=0.4903883073401403,
passing the original sign/phase/length gates. Crosscheck 11355790 passed,
with difference 1.14e-13 from the selected chi4 value and all source hashes
verified. This validates the new solver-to-bare adapter at chi4. The chi32
dominance-constrained trial 11355777 is running and remains unaccepted.

```bash
sbatch --clusters=htc --partition=htc --time=02:00:00 --mem=24G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/newton_dominance_krylov.jl \
  SOURCE NEW_OUT data/independent_bivumps_20260919/dominance_response4 \
  20 0 1e-9 192
```

The matching actual-state control is `dominance_response32/checks.toml`.
After both spatial directions finish, run `audit_saved.jl` on each and use
`bare_dominance.py NATIVE ROTATED EQUATION_AUDIT NEW_MEASUREMENT`; do not
stage an unconverged native or rotated input.

Read-only rank diagnostics: `branch_overlap.jl` compares the completed chi24
Schmidt best state with accepted chi16 (11355778), and the failed Gaussian
chi16 branch run with chi12 (11355779). The Gaussian trial has per-site overlaps
0.858/0.861 with chi12, despite small weights above the chi12 Schmidt ranks;
it cannot be described as the same chi12 state. Its 192-direction optimization
completed 120 steps with best full residual 0.00365692. Neither overlap nor a
small Schmidt tail certifies convergence or physical accuracy.
The chi24 comparison completed: right/left per-site fidelities with chi16 are
0.9999997942 / 0.9999990951, and total weights beyond the chi16 parity ranks
are 2.9186e-8 / 6.6686e-8. This supports proximity to a low-rank branch;
it does not certify global equality or accepted entropy.

`slopes.py` verifies every Gaussian source (original numpy dictionary and
later JSON records), reproduces per-junction finite-size fits, and computes
finite-interval secants of the new bare curve. Job 11355461 passed; artifacts
are `data/direct_chi_curve/independent_bivumps_lnxi_slopes.{csv,json,md}`.
The physical-xi slopes for chi8->12 and chi12->16 are 0.154951650 and
0.080831466 respectively; Gaussian L>=24 fit-window values span
0.155215976--0.155354945. This does not establish asymptotic agreement.
Run with `jobs/run_python.sh benchmark/independent_bivumps/slopes.py` via
the same Slurm command prefix used below; retain the exact source hashes.

To repeat that isolated experiment (use new output directories):

```bash
sbatch --clusters=htc --partition=htc --time=00:30:00 --mem=16G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/newton_metric_response.jl SOURCE CONTROL_OUT
# After the exact-source derivative control passes:
sbatch --clusters=htc --partition=htc --time=01:00:00 --mem=16G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/newton_metric.jl SOURCE OUT CONTROL_OUT 25 0 1e-9
```

The remaining paragraphs document the implementation and earlier trials;
their intermediate status descriptions are historical snapshots.

This directory is separate from all legacy constrained-pair optimizers. `R` and
`L` are independently stored and independently updated uniform MPS in the same
bra/ket chart. No `R=move(S)` constraint is used. Legacy checkpoints are warm
starts only, never accepted new results.

`core.jl` builds mixed environments with the two states' own left/right canonical
tensors, constructs site and bond pencils, and whitens their actual overlap
metrics with separate dual bases. Both left and right local eigenvectors are
updated from the SAME pencil. Storage uses ordinary canonical form for each
state; biorthogonal coordinates are used within each paired center solve.

`audit.jl` compares the spatial and bra-chart networks on independently perturbed
complex tensors and checks both independent uniform quotient derivatives using
finite differences. `run.jl` performs paired updates, re-evaluates all outer
residuals after regauging, saves native spatial boundaries and all-sector MPS/pair
correlation lengths. No entropy is certified by this solver alone.

Run from `PEPS/code2d/fpeps`, through Slurm:

```bash
sbatch --clusters=htc --partition=htc --time=00:30:00 --mem=12G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/audit.jl SOURCE NEW_AUDIT_DIRECTORY
sbatch --clusters=htc --partition=htc --time=00:30:00 --mem=12G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/run.jl SOURCE NEW_RUN_DIRECTORY 60 0.001
```

The arguments after the directories are maximum iterations and initial right-only
perturbation amplitude. Both source boundaries must have the desired chi.
Numerical failures are preserved in distinct output directories, never overwritten.
An optional fifth argument sets the outer tolerance (e.g. `1e-12`).

`run_damped.jl` is an explicitly separate fixed-damping (`alpha=0.1`) variant.
It allows transient outer-residual increases, guarded at 0.25; final convergence
still requires the same fresh independent equations. It is being tested on chi=8
after monotone line search stalls. Its results must retain `update_strategy`.

`expand.jl SOURCE OUT EVEN ODD [NOISE]` independently embeds both accepted new
boundaries and perturbs newly admitted bond directions. Both `(3,3)` and `(4,2)`
chi=6 cold expansions stalled, so they are NOT entropy inputs. Separately, a
legacy `(4,2)` chi=6 warm start with independent perturbation successfully reduced
the new residual by many orders of magnitude; its final polish is still running.

`bare.py NATIVE ROTATED EQUATION_AUDIT OUT` checks the new schema and fresh outer
convergence, stages byte-identical native boundaries, and calls the unchanged
bare/direct backend. `observables.jl` and `physical_modes.jl` adapt only report
metadata for the existing signed measurements. `collect.py` reads only the new
schema. `plot.py` writes a separately named series to `data/direct_chi_curve/`.

Verified chi=4: native residual 7.99e-13, rotated residual 9.71e-13;
bare S_tilde=0.49038830734025396; xi_pair=4.26189282894571;
xi_MPS,R=xi_MPS,L=1.170877010086 (all parity sectors; even-only is half);
physical normal/anomalous xi=4.788325855198712.
Replica length/phase checks passed. Finite-chi Gaussian accuracy is NOT certified:
e.g. the normal correlator at r=7 differs by about 0.00433 from the frozen Gaussian
reference. Report physical errors separately from solver and contraction residuals.

Technical note compiled successfully as `PEPS/notes/fpeps_signs_zh/main.pdf` and
`independent_bivumps.pdf`; build evidence is `note_build_verification_v2.json`.
The diagram on main.pdf page 12 and standalone page 2 were visually checked.

Updated results: chi=4,6,8,12 independent boundaries and bare/direct measurements
have passed. `results.csv/json` are the source-checked aggregate; the new plot is
`data/direct_chi_curve/independent_bivumps_stilde_chi.{png,pdf,json}`.
The later note build manifests (`v3`, `v4`, `v5`) include subsequent details/results.

`anderson.jl` is the successful accelerator at chi=8 and 12. It aligns each MPS's
own virtual basis with a mixed-overlap polar unitary before storing real/imaginary
secant differences (memory=8). Each alignment verifies canonical identities and
fidelity invariance. Known random gauge controls are in `alignment4.toml`.
Bare measurements require controls matching the exact accelerator source hash.

Chi=16 cold continuation initially failed at its first guarded step. The current
accelerator extends the line-search grid down to 1/1024; its new control is
`anderson4_control_v2` with `alignment4_v2.toml`. Old chi=8/12 results retain their
frozen `source_run.jl` and original hash, so this change does not relabel them.
Both continuation from chi=12 and independently perturbed old chi=16 initial
guesses are being tested; see `chi16_smallsteps_warm_submissions.json`.

`audit_saved.jl` rechecks stored and exported states without optimization. The
original 1e-9 environment gate is distinct from optional 1e-12 polishing: rotated
chi=6 stalled at 1.88e-12, passed the original gate on fresh audit, and retains its
failed stricter-polishing flag. This distinction is recorded in measurement status.

New data root: `data/independent_bivumps_20260919/`.
The draft technical note is `PEPS/notes/fpeps_signs_zh/independent_bivumps.tex`.

Initial submissions: 11353606 failed the deliberately explicit bra-map involution
check; the map is antiunitary but not self-inverse. 11353609 uses the actual
coefficient-map inverse. 11353607 checks the equations independently of that
inverse. These job IDs are provenance, not evidence that any check passed.

## Chi=16 update diagnostics and separate Newton experiment

`map_diagnostics.jl` now demonstrates a weak-Schmidt-coordinate problem:
the old native chi=16 stationary seed has minimum Schmidt value 2.066e-6.
At a right-only 1e-8 perturbation, one paired update moves AL by 1.870e-4,
but AC by only 1.173e-7; the full residual decreases from 2.713e-7 to
4.996e-8 on that single step. This is evidence of coordinate amplification,
not proof that it is the sole convergence problem. The unperturbed seed's
dominant local root has modulus rank 1, and its mixed-metric singular ratio
is about 0.459; the failure cannot simply be attributed to a wrong local root
or a singular overlap metric at that seed.

`anderson_centers.jl` uses individually gauge/center-phase aligned AC and C
coordinates. Its known-gauge control and chi4 convergence control passed.
Both chi16 runs with perturbation 0.001 still failed. AL-coordinate Anderson
with perturbation 1e-6 and memory 8 stopped at best residual 1.215e-6.
None of these chi16 trials is an accepted entropy input. Memory-32 tests are
recorded separately in `newton_response_submissions.json`; check live Slurm.

`newton_response.jl` implements an analytic stationary-equation correction with
two independent complex, parity-preserving MPS variables. Four bordered cap
response equations differentiate the actual mixed row/overlap channels.
`newton.jl` uses the resulting real Hessian of Re log(q) in two independent
isometric tangent charts, with an indefinite least-squares trust step. It
does NOT call a legacy direction-tied optimizer and is explicitly reported
as a Newton globalization experiment, not an unmodified VUMPS iteration.
Acceptance still requires the same full independent AC/C audit in `core.jl`.

Submit the response control first; the solver requires its exact source hash:

```bash
sbatch --clusters=htc --partition=htc --time=02:00:00 --mem=32G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/newton_response.jl SOURCE NEW_CONTROL_DIR
sbatch --clusters=htc --partition=htc --time=02:00:00 --mem=32G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/newton.jl SOURCE NEW_RUN_DIR CONTROL_DIR 12 0.001 1e-9
```

The dense Newton pilot is bounded at 2048 real tangent coordinates. A missing
derivative control, failed Hessian check, or failed full outer residual cannot
be promoted to an accepted result. `bare.py` now requires matching controls
for both Anderson coordinate variants and for any Newton-generated input.

Verified Newton controls: `newton_response4/checks.toml` passes independent
real/imaginary directions on both sides at a nonstationary pair (Hessian finite
difference errors 1.3e-9 to 4.8e-9). `newton4_control/report.toml` converges after
three Newton steps: outer residual 2.726e-3 -> 2.802e-5 -> 1.685e-9 -> 5.617e-14.
It reproduces xi_pair=4.261892828943499. This is a small-chi solver control;
chi16 and its entropy still need separate acceptance. Bare and export controls
are submitted in `newton4_measurement_submissions.json`.

The exact-adjoint cache (`newton_response_fast.jl`, `newton_fast.jl`) avoids
rebuilding a full effective matrix for every derivative. It caches the linear,
graded row-growth map and uses its adjoint; an explicit adjoint identity and
independent scalar/Hessian difference tests pass. Both native and rotated chi16
warm starts now converge independently (2.94e-11 and 2.97e-12), in two and three
Newton steps respectively. These use perturbation 1e-6, so they do not establish
convergence from arbitrary or rank-expanded initial states. The latter is being
tested separately as `native_chi16_newton_cold`.

`newton_wide.jl` uses the same cached response and trust equations, with an
explicit dense-workspace bound of 6144 real coordinates (chi32). Its own chi4
control passed. Native chi24 is being initialized by independently expanding
the accepted native chi16 Newton pair, with new-sector noise 0.0003. Read
`chi24_newton_submissions.json` together with `chi24_resource_adjustment.json`:
the first chi24 solve was reduced to a two-hour/32 GiB allocation, and the
embedding job to 30 minutes after an oversized resource reservation delayed it.

`expand.jl` preserves failed optional-polishing flags. A seed that misses a
stricter target may be used only when its actual stored/exported states pass
the original 1e-9 gate and all audit/source hashes match.

`bare_wide.py` stages ONLY new independent pairs and calls the existing,
unchanged `wide_bare_direct_scan_point.jl` / `scan_wide_bare` through chi32.
It requires a stored/exported audit for every input and checks the signed-cycle
character comparison in addition to the original phase/length gates. Its small
chi comparison is `measurement_wide4_control`; it never uses grown or endmap
entropy tensors. `physical_modes_wide.jl` extends only the explicit dimension
bound of the full-spectrum physical-mode measurement to chi32, retaining the
actual-correlator reconstruction checks.

Curve selection is now explicit in `data/independent_bivumps_20260919/curve_points.json`.
Run `results.py` before `plot.py`; the latter consumes the source-checked table.
This excludes solver-control measurements and duplicate chi values. The real
four-point regression passed with every old numeric field unchanged.


## Matched projection of two independently grown boundaries

`paired_schur.jl` and `paired_seed.jl` are a new initialization route. Both
spatial boundaries first absorb their own signed row. In the common ket chart,
one mixed overlap channel provides the left and right caps l and r. Within
each FermionParity block, ordered Schur decompositions select the retained
left/right invariant subspaces of D = r*l. Their cross-overlap SVD supplies
dual maps J*U = I. The partner boundary maps are then derived from the same
l,r and the square root of the retained D, rather than selected by an
independent one-state Schmidt cut.

The projection is a finite-rank initialization heuristic, not a proof of
fidelity optimality or monotonic correlation length. The retained/discarded
modulus gap, retained cross-overlap conditioning, dual identities, signed
full-rank restoration, and ordinary canonicalization are checked explicitly.
An unresolved cut or singular retained pairing stops the preparation. No
physical-rank cutoff is silently changed.

Pure-matrix control `paired_schur_control.toml` (11355819) passed all eight
cases, including defective discarded Jordan blocks. It compares with known
invariant projectors, with errors below 1.4e-15. Full-rank chi4 graded MPS control 11355824 passed: tensor restoration
errors below 9.4e-14, both state fidelities within 1.2e-15 of unity,
paired spectral ratio change 9.0e-14, and full independent residual
8.63e-13. The stored roundtrip also passed. Evidence is
`paired_fullrank4/preparation.toml`; all its source hashes were checked
before submitting the truncated chi4-to-8 preparation.

```bash
# Run inside PEPS/code2d/fpeps; first check paired_schur_control.toml.
sbatch --clusters=htc --partition=htc --time=00:20:00 --mem=8G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/paired_seed.jl \
  data/independent_bivumps_20260919/native_chi4 NEW_OUT \
  data/independent_bivumps_20260919/paired_schur_control.toml \
  2 2 false 0
```

The final arguments are retained even/odd ranks, whether to grow the two
boundaries, and the relative tensor perturbation **before** constructing the
mixed caps. With growth enabled, preparation outputs are initial guesses;
the saved `converged` and `bivumps_converged` flags are false. Subsequent
independent optimization and all original entropy/physical checks are required.
Exact submission records include the node exclusions used in this workspace.

## Frozen-state initialization near a modulus crossing

Dominance job 11355777 stopped at its initial margin check without a Newton
step. The final metric-solver checkpoint differs from the earlier frozen
chi32 state on which derivatives had been audited. A spatial export/reimport
of that earlier pair failed its spectral-invariance guard (11355806).
Dependent job 11355807 was cancelled before execution; no solver result was
produced. `export_snapshot_audit.jl` (11355827) measured a 1.8911e-10
absolute gap change, exceeding the unchanged 1e-10 roundtrip guard.
The row gap stayed positive (3.6089e-8 to 3.6278e-8) and the state
fidelity was 1.0000000000000064: this is a strict conversion check,
not an observed dominant-root swap or a physical excitation gap.

`newton_snapshot_dominance.jl` adds a direct frozen-pair input to the same
constrained Newton equations, avoiding an unnecessary spatial conversion at
initialization. The optional eighth argument is the pair file. This path
checks both tensor spaces, the PEPS transfer tensor, and that the snapshot
hash is exactly the chi32 state used by the critical derivative control.
It retains the final spatial export and saved-state audit requirements.
A separate chi4 regression control with matching solver hash is required.


The first truncated matched chi4-to-8 preparation passed (11355830):
initial full residual 0.00711677, xi_pair 5.28298621, xi_MPS R/L
1.65749067 / 1.65551191. These are seed diagnostics. Independent Anderson
and dominance-constrained Newton trials started from this SAME pair;
see `paired_seed8_solver_submissions.json` (11355849/11355850).

The full-rank chi16 test of the square-root balanced maps failed
(11355834): the partner left-inverse error was 8.70e-8, above 1e-9.
The source and failed preparation remain unchanged. The new
`paired_schur_qr.jl` computes the equivalent partner invariant ranges
`range(r' * J_R')` and `range(l * U_R)` by QR, then biorthogonalizes their
cross-overlap by SVD. It avoids inversion of the retained density square
root. The paired projectors and retained subspaces are unchanged in exact
arithmetic; equal numerical representations of both caps are no longer
imposed. The cap product and both partner invariance relations are checked.

The QR matrix control 11355851 passed 12 analytically known projector
cases, including a retained spectrum spanning 1e-10 and discarded Jordan
blocks. All source hashes were checked before submitting a full-rank
chi16 control and a chi4-to-8 preparation for comparison with the first
implementation (`paired_qr_state_submissions.json`, 11355852/11355853;
use that record for authoritative job IDs).


The matched chi4-to-8 seed has now converged under the independent Anderson
map (11355849): 30 steps, full residual 4.75376e-10, xi_pair 10.8218030646,
xi_MPS R/L 2.6764831539 / 2.6764831545. These agree closely with the selected
chi8 solution, but stored-state, bare and observable crosschecks remain
separate. A tighter 1e-11 polish and normalized-overlap comparison were
submitted; see `paired_seed8_followup_submissions.json`. The unmodified
selected curve is still authoritative.

The unconstrained branch chi32 trial 11355712 completed 80 steps without
passing: best original residual 0.01757599 was at its initial state; the final
tracked branch had residual 1.00629 and all four tracked roots at modulus
rank 2. This confirms that branch continuity alone did not find the required
dominant stationary solution. It is a failed experiment, not an entropy point.

The direct snapshot input control 11355829 recovered chi4 in three steps,
full residual 3.03601e-12; its saved-state audit 11355844 passed at
3.03777e-12. Bare/direct job 11355847 passed; independent crosschecking is
recorded in `paired_seed8_followup_submissions.json`. The new chi32 trial is
11355845, `native_chi32_dominance_exact_snapshot`; recheck Slurm and its
report rather than interpreting the existence of an input file as convergence.


The QR full-rank chi16 control 11355852 passed: raw tensor restoration
errors 1.62e-15 / 1.06e-15, matched metric errors <=1.13e-15, xi_pair
23.81041186819383 -> 23.810411868192585, and full independent residual
1.23859e-11. This fixes the tested square-root-inverse roundoff without
changing the retained rank or any tolerance. The QR chi4-to-8 seed also
completed (11355853), initial residual 0.00711676549300214. Its explicit
gauge-equivalence comparison is job 11355857.

Only after the full chi16 control passed were the new chi12-to-16 and
chi16-to-24 preparations submitted (11355858/11355859). See
`paired_qr_large_seed_submissions.json`. These are preparation jobs, not
accepted environments. Both subsequent optimizers must retain independent
left/right variables and the unchanged original acceptance gates.

The same original paired chi8 seed also converged under dominance-constrained
Newton (11355850): 12 steps, full residual 9.05442e-10, xi_pair 10.8218031224,
all four caps dominant. This is a second solver convergence from the paired seed. Similar xi
does not by itself establish state equivalence; export and observable/entropy
checks are still required.


The original selected8-versus-new8 overlap check (11355855) stopped at an
unresolved even-sector cap gap 5.98e-10. The all-parity dense read-only
diagnostic 11355868 found normalized overlap radii [0.6882352120, 1.0]
for BOTH spatial boundaries. This suggests an odd virtual gauge relation;
it is not evidence of a failed physical contraction. Physical correlators,
RDMs and bare/direct comparisons were therefore submitted explicitly
(`paired_polished8_measurement_submissions.json`). No solver cap guard changed.

Matched QR seed chi16 passed preparation at initial residual 0.00545315;
Anderson memory32 and dominance Newton trials are listed in
`paired_qr_large_followup_submissions.json`. Chi24 preparation 11355859
failed the original mixed-metric singular-ratio gate (2.65407e-13 < 1e-12)
after its projection identities passed. The next preparation increases only
the initial independent noise from .001 to .01. Keep this failure distinct
from the corrected square-root basis-construction error.


Physical-space comparison now passed (11355869/11355872/11355875): all
128-distance normal/anomalous/connected-density correlators agree with the
selected chi8 within 8.97e-13 / 7.55e-13 / 2.89e-14. One-to-three-site RDM
entries agree within 9.0e-13. The odd-sector overlap is therefore compatible
with physically equivalent boundaries in these direct tests. Finite-chi
Gaussian errors remain 2.77998e-3 / 1.59502e-4 / 5.79447e-5. Evidence and
plot: `data/direct_chi_curve/independent_pairseed8_physical_comparison.*`.

Bare job 11355867 did not begin contraction: the old selected rotated chi8
was missing the zero-step saved audit required by the newer wide preflight.
The new native audit had already passed. Its missing rotated audit and a
fresh-output bare retry were submitted, followed by the full shared-depth
comparison (`paired_polished8_bare_v2_submissions.json`). All original
preflight gates and state tensors were retained.


## Current matched-seed submission example

Run from `/ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps`. Use new output directories
for every trial. The following prepares chi16 from the independently solved
chi12 pair, then runs **independent** R/L variables at the target chi:

```bash
sbatch --clusters=htc --partition=htc --time=00:40:00 --mem=24G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing \
  benchmark/independent_bivumps/paired_seed_qr.jl \
  data/independent_bivumps_20260919/native_chi12_anderson NEW_SEED \
  data/independent_bivumps_20260919/paired_schur_qr_control.toml \
  8 8 true 0.001

# First require NEW_SEED/preparation.toml complete=true and inspect diagnostics.
sbatch --clusters=htc --partition=htc --time=02:00:00 --mem=24G \
  jobs/run_cpu.sh --compiled-modules=existing --pkgimages=existing -e \
  'include("benchmark/independent_bivumps/anderson.jl"); bv_anderson(ARGS[1],ARGS[2];maxiter=160,tol=1e-11,memory=32)' \
  NEW_SEED NEW_SOLVE
```

The matching small-chi Anderson memory32 control is already part of the
measurement preflight. The independent Newton alternative and exact node
exclusions are in `paired_qr_large_followup_submissions.json`. After a solve
passes the original 1e-9 full gate, run `audit_saved.jl NEW_SOLVE`. Both
spatial orientations must have a passing `saved_audit.toml` before using
`bare_wide.py NATIVE ROTATED chi4_equations_nonstationary NEW_MEASUREMENT`.
All these commands must be submitted through Slurm. The bare adapter
requires the full control directory path, e.g.
`data/independent_bivumps_20260919/chi4_equations_nonstationary`.

For physical checking, submit `observables.jl NEW_SOLVE NEW_PHYSICAL`, then
`benchmark/compare_joint_correlations.py NEW_PHYSICAL` with `jobs/run_python.sh`.
Report all finite-chi Gaussian errors. Similar xi or a stationary solver alone
is insufficient to accept a new entropy point. The physical comparison
script `compare_physical_states.py FIRST_PHYSICAL SECOND_PHYSICAL FIGURE_STEM`
keeps complex phases and checks one-to-three-site RDMs.


Latest run snapshot: `progress_after_paired_qr_and_physical_checks.json`.
The chi16 Anderson trial 11355877 failed at the next-state gauge-alignment
cap (relative residual 3.62e-6, modulus gap .1295), after best full residual
.0042505; it did not produce an accepted environment. The independent
Newton trial remains separate. A future alignment diagnostic should capture
candidate/reference tensors and inspect BOTH parity sectors before altering
any guard. Noise .01 chi24 preparation completed but has full residual 4.355,
complex row quotient 3.58012+.414657i, and nondecaying paired spectrum;
it is not a usable physical environment. Intermediate noise .003 was
submitted as 11355890. Check authoritative status before continuing.


Bare v2 (11355883) reached the actual seam audit and failed: one-copy AC
leading modulus gap 2.02494e-12 was unresolved. Both independent stored
boundary audits had passed. This is distinct from the earlier missing-file
preflight failure. The combination used new native8 (odd overlap sector
relative to old native8) with OLD rotated8. Local physical agreement does
not certify this cross-direction seam choice. No cap selection, isolation
guard, or entropy backend was changed. Dependent comparison 11355884 was
cancelled before execution. The new chain `paired_rotated8_submissions.json`
independently prepares/solves rotated8 through the matched QR route, audits
it, then runs bare with BOTH new directions and compares all shared depths.
Until that passes, no new paired-seed entropy is accepted.
