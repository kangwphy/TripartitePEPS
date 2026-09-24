"""Read saved results and build this note's tables/figures under Slurm preempt.

No boundary is optimized; no accepted curve or existing numerical result is
overwritten. New Gaussian comparisons use the existing independent oracle.
"""
import csv
import hashlib
import json
import os
from statistics import median
from pathlib import Path
import subprocess
import sys

from pip._vendor import tomli
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

assert os.environ.get("SLURM_JOB_PARTITION") == "preempt", "preempt required"
assert os.environ.get("SLURM_JOB_ID", "").isdigit(), "submit through Slurm"
note = Path(__file__).resolve().parent
root = note.parent.parent / "code2d/fpeps"
os.chdir(root)
base = root / "data/mpbp_20260920"
out = note / "generated"
out.mkdir(exist_ok=True)
sources = [Path(__file__).resolve()]


def read_json(path):
    sources.append(path)
    return json.loads(path.read_text())


def read_toml(path):
    sources.append(path)
    return tomli.loads(path.read_text())


def sci(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def table(name, columns, header, rows):
    text = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
            " & ".join(header) + r" \\", r"\midrule"]
    text.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    text.extend([r"\bottomrule", r"\end{tabular}"])
    (out / name).write_text("\n".join(text) + "\n")


old = read_json(root / "data/independent_bivumps_20260919/results.json")
table("old_chi_table.tex", "rrrrrrr",
      [r"$\chi$", r"$(\chi_e,\chi_o)$", r"$\widetilde S_{\rm direct}$",
       r"$\xi_{\rm MPS,R}$", r"$\xi_{\rm pair}$", r"$\xi_{\rm phy}$", r"$\epsilon_{\rm joint}$"],
      [[r["chi"], f"({r['even']},{r['odd']})", f"{r['stilde']:.10f}",
        f"{r['xi_mps_right']:.5f}", f"{r['xi_pair']:.5f}",
        f"{r['xi_physical']:.5f}", sci(r['max_outer_residual'])]
       for r in old["rows"] if r["chi"] in (4, 6, 8, 12, 16)])

rows, comparisons = [], {}
for step in (0, 5, 15, 35, 108, 135):
    folder = base / "chi16_correlations" / f"step{step}"
    report = read_toml(folder / "report.toml")
    assert report["complete"] and "error" not in report
    assert report["chi"] == 16 and report["w"] == 1
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    comparison_path = folder / "gaussian_comparison.json"
    if not comparison_path.exists():
        subprocess.run([sys.executable, "benchmark/eigctm/compare_gaussian.py",
                        str(folder)], check=True)
    comparison = read_json(comparison_path)
    for path, expected in comparison["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    audit = report["boundary_audit"][0]
    row = dict(step=step, joint_residual=report["joint_residual"],
               xi_mps_right=audit["xi_mps_right"]["xi"],
               xi_mps_left=audit["xi_mps_left"]["xi"],
               xi_pair=audit["xi_pair"]["xi"],
               rdm_checks_passed=report["rdm_checks_passed"],
               max_rdm_hermiticity=max(r["hermiticity_error"] for r in report["rdms"]),
               min_rdm_eigenvalue=min(r["min_eigenvalue"] for r in report["rdms"]),
               max_local_error=max(v["direct_error"] for v in report["observables"].values()))
    for operator in ("normal", "anomalous", "nn"):
        subset = [r for r in comparison["rows"] if r["operator"] == operator]
        row["error_" + operator] = max(r["absolute_error"] for r in subset)
        row["grid_drift_" + operator] = max(r["gaussian_grid_drift"] for r in subset)
    assert row["rdm_checks_passed"], row
    rows.append(row)
    comparisons[step] = comparison

table("trajectory_table.tex", "rrrrr",
      [r"总步数 $n$", r"$\epsilon_{\rm joint}$", r"$\xi_{\rm MPS,R}$",
       r"$\xi_{\rm MPS,L}$", r"$\xi_{\rm pair}$"],
      [[r["step"], sci(r["joint_residual"]), f"{r['xi_mps_right']:.6f}",
        f"{r['xi_mps_left']:.6f}", f"{r['xi_pair']:.6f}"] for r in rows])
table("gaussian_table.tex", "rrrr",
      [r"总步数 $n$", r"$\max|\Delta C_N|$", r"$\max|\Delta C_A|$",
       r"$\max|\Delta C_{nn}|$"],
      [[r["step"], sci(r["error_normal"]), sci(r["error_anomalous"]),
        sci(r["error_nn"])] for r in rows])

trajectory = [dict(step=0, joint_residual=rows[0]["joint_residual"])]
xi = [{key: r[key] for key in ("step", "xi_mps_right", "xi_mps_left", "xi_pair")}
      for r in rows if r["step"] <= 5]
fidelity = []
for offset, folder in ((0, "dense_mpbp_iter_chi16_v1"),
                       (5, "dense_mpbp_chi16_continue_with_xi"),
                       (35, "dense_mpbp_chi16_steps36_135")):
    data = read_toml(base / folder / "report.toml")
    assert data["complete"] and data["stop_reason"] == "maxiter"
    for r in data["rows"]:
        assert r["complete_step"]
        step = offset + r["iteration"] + 1
        trajectory.append(dict(step=step, joint_residual=r["after_joint_residual"]))
        if "xi_pair" in r:
            xi.append(dict(step=step, **{key: r[key][0]["xi"]
                for key in ("xi_mps_right", "xi_mps_left", "xi_pair")}))
            fidelity.append(dict(step=step, error=max(r["edge_fidelity_errors"])))
assert [r["step"] for r in trajectory] == list(range(136))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
axes[0].semilogy([r["step"] for r in trajectory],
                 [r["joint_residual"] for r in trajectory], lw=1.7)
axes[0].axhline(1e-9, color="0.4", ls="--", label=r"$10^{-9}$ gate")
axes[0].set(ylabel="Original joint residual", xlabel="MP-BP update n")
axes[0].legend(fontsize=9)
for key, label, style in (("xi_mps_right", r"$\xi_{\mathrm{MPS,R}}$", "-"),
                          ("xi_mps_left", r"$\xi_{\mathrm{MPS,L}}$", "--"),
                          ("xi_pair", r"$\xi_{\mathrm{pair}}$", "-")):
    axes[1].plot([r["step"] for r in xi], [r[key] for r in xi], style, label=label)
axes[1].set(ylabel="Correlation length", xlabel="MP-BP update n")
axes[1].legend(fontsize=9)
axes[2].semilogy([r["step"] for r in fidelity], [r["error"] for r in fidelity])
axes[2].set(ylabel=r"max $|1-|f_{\rm site}||$", xlabel="MP-BP update n")
for ax in axes:
    ax.grid(alpha=.22)
fig.suptitle(r"$w=1$, $\chi=16$: nonstationary dense-Schur MP-BP trajectory")
for ext in ("pdf", "png"):
    fig.savefig(out / f"long_trajectory.{ext}", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for key, label in (("normal", r"$\langle c^\dagger_0 c_r\rangle$"),
                    ("anomalous", r"$\langle c_0 c_r\rangle$"),
                    ("nn", r"$\langle n_0 n_r\rangle$")):
    axes[0].semilogy([r["step"] for r in rows], [r["error_" + key] for r in rows],
                     ".-", label=label)
axes[0].set(xlabel="MP-BP update n", ylabel="Max complex absolute error (x and y)")
axes[0].legend(fontsize=9)
for step in (0, 5, 35, 108, 135):
    subset = [r for r in comparisons[step]["rows"]
              if r["operator"] == "normal" and r["direction"] == "x" and r["r"] % 2 == 1]
    axes[1].semilogy([r["r"] for r in subset],
                     [max(r["absolute_error"], 1e-17) for r in subset], ".-", label=f"n={step}")
axes[1].set(xlabel="Odd separation r in x direction", ylabel="Normal correlator absolute error")
axes[1].legend(fontsize=9, ncol=2)
for ax in axes:
    ax.grid(alpha=.22)
fig.suptitle("Signed CTM correlators vs exact Gaussian covariance")
for ext in ("pdf", "png"):
    fig.savefig(out / f"gaussian_diagnostics.{ext}", dpi=180)
plt.close(fig)

tail = read_toml(base / "mpbp_candidate_bare_chi16_slurm_tail_v1/report.toml")
assert tail["converged"] and not tail["accepted_entropy"] and tail["boundary_gate_bypassed"]
table("bare_tail_table.tex", "rrrr",
      [r"LMPS 深度 $\ell$", r"$\widetilde S$（固定第 5 步环境）", r"相位误差", r"端点残差"],
      [[r["depth"], f"{r['stilde_real_diagnostic']:.13f}",
        sci(r["phase_error"]), sci(r["endpoint_residual"])] for r in tail["samples"]])

center_control = read_toml(base / "center_gradient_aligned8_v1.toml")
assert center_control["complete"] and center_control["passed"]
assert center_control["nonstationary"] and not center_control["accepted_entropy"]
for path, expected in center_control["source_hashes"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

newton_merit = read_json(base / "center_newton_aligned8_v1/merit_audit.json")
assert newton_merit["complete"] and not newton_merit["converged"]
newton_source = Path(newton_merit["source"])
assert hashlib.sha256(newton_source.read_bytes()).hexdigest() == newton_merit["source_sha256"]
table("newton_merit_table.tex", "rrrrr",
      [r"步数", r"原验收残差", r"max raw", r"max whitened", r"$\sigma_{\min}/\sigma_{\max}$"],
      [[r["iteration"], sci(r["residual"]), sci(r["max_raw"]), sci(r["max_white"]),
        sci(r["metric_ratio"])] for r in newton_merit["rows"] if r["iteration"] in (0, 1, 2, 5, 8, 12)])

guard_control = read_toml(base / "center_newton_v7_control4/report.toml")
guard_run = read_toml(base / "center_newton_guard8_v1/report.toml")
assert guard_control["complete"] and guard_control["converged"]
assert guard_run["complete"] and not guard_run["converged"]
assert guard_run["stop_reason"] == "Newton_trust_step_failed"
assert guard_control["run_sha256"] == guard_run["run_sha256"]
for report in (guard_control, guard_run):
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
trials = guard_run["rows"][-1]["trials"]
assert len(trials) == 12 and all(t["original_residual_decreased"] is False for t in trials)

mixed_controls = [read_toml(base / f"mixed_center_response{chi}_v1.toml") for chi in (4, 8)]
mixed_small = read_toml(base / "newton_mixed4_control_v1/report.toml")
mixed_export = read_toml(base / "newton_mixed4_public_v1/export_audit.toml")
assert mixed_small["complete"] and mixed_small["converged"]
assert mixed_export["complete"] and mixed_export["passed"]
for report in mixed_controls:
    assert report["complete"] and report["passed"] and report["nonstationary"]
for report in mixed_controls + [mixed_small, mixed_export]:
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

mixed_progress = read_toml(base / "newton_mixed8_v1/report.toml")
assert mixed_progress["complete"] and not mixed_progress["accepted_entropy"]
assert not mixed_progress["converged"] and mixed_progress["stop_reason"] == "maxiter"
assert mixed_progress["job_id"] == "11415141"
for path, expected in mixed_progress["source_hashes"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
table("mixed_progress_table.tex", "rr",
      [r"v8 更新步数", r"原验收残差（未收敛）"],
      [[r["iteration"], sci(r["outer_residual"])] for r in mixed_progress["rows"]])

mixed_physical = read_json(base / "newton_mixed4_physical_comparison_v1.json")
mixed_bare = read_json(base / "newton_mixed4_bare_v1/status.json")
mixed_bare_comparison = read_json(base / "newton_mixed4_bare_comparison_v1.json")
assert mixed_physical["complete"] and mixed_physical["physical_match"]
assert mixed_bare["complete"] and mixed_bare["contraction_passed"]
assert mixed_bare_comparison["complete"] and mixed_bare_comparison["passed"]
assert not mixed_bare["accepted_entropy"] and mixed_bare["chi4_reference_regression"]
for report in (mixed_physical, mixed_bare, mixed_bare_comparison):
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

mixed_continue = read_toml(base / "newton_mixed8_continue_v1/solve/report.toml")
mixed_recovery = read_toml(base / "newton_mixed8_reference_recovery_a001_v1/report.toml")
mixed_recovery_export = read_toml(base / "newton_mixed8_reference_public_a001_v1/export_audit.toml")
mixed_caps = read_toml(base / "mixed_reference8_cap_scan_v1.toml")
assert mixed_continue["complete"] and not mixed_continue["converged"]
assert mixed_recovery["complete"] and mixed_recovery["converged"]
assert mixed_recovery_export["complete"] and mixed_recovery_export["passed"]
assert mixed_caps["complete"] and mixed_caps["diagnostic_only"]
for report in (mixed_continue, mixed_recovery, mixed_recovery_export, mixed_caps):
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
table("mixed8_branches.tex", "lrrrr",
      ["初值路线", r"原残差", r"$\xi_{\rm MPS,R}$", r"$\xi_{\rm MPS,L}$", r"$\xi_{\rm pair}$"],
      [[label, sci(r["final"]["outer_residual"])] +
       [f'{r[k]["xi"]:.6f}' for k in ("xi_mps_right", "xi_mps_left", "xi_pair")]
       for label, r in (("CTM 初值（未收敛）", mixed_continue), ("已知解附近（收敛）", mixed_recovery))])

failed16_controls = [read_toml(base / f"mixed_center_failed16_{direction}_control_v1.toml")
                     for direction in ("native", "rotated")]
metric16 = read_toml(base / "mixed_metric_actions16_v1.toml")
full_action16 = read_toml(base / "mixed_full_actions16_v1.toml")
for report in failed16_controls + [metric16, full_action16]:
    assert report["complete"] and report["passed"] and not report["accepted_entropy"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
for report in failed16_controls:
    assert not report["perturbed"] and report["nonstationary"]
table("failed16_controls.tex", "lrrrr",
      ["方向", "原方程残差", "度量恒等式误差", "四组 FD 最坏误差", "实线性误差"],
      [[label, sci(r["original"]["outer_residual"]), sci(r["white_identity_error"]),
        sci(max(min(s["relative_error"] for s in d["stencils"]) for d in r["derivative_checks"])),
        sci(r["real_linearity_error"])]
       for label, r in zip(("原方向", "旋转方向"), failed16_controls)])
timing = full_action16["rows"][0]
table("full_action16.tex", "lr",
      ["完整响应比较项目", "数值"],
      [["导数最大相对差异", sci(max(r["derivative_relative_error"] for r in full_action16["rows"]))],
       ["原残差有限差分最坏误差", sci(max(min(s["relative_error"] for s in r["stencils"]) for r in full_action16["rows"]))],
       ["实线性误差", sci(full_action16["real_linearity_error"])],
       ["原完整导数耗时中位数（秒）", f'{median(timing["original_seconds"]):.3f}'],
       ["改写后完整导数耗时中位数（秒）", f'{median(timing["action_seconds"]):.3f}']])

vector16 = read_toml(base / "mixed_vector_response16_v1.toml")
replay16 = read_toml(base / "mixed_vector_step16_v1.toml")
physical16 = read_json(base / "newton_mixed16_native_iter1_gaussian.json")
xi16 = read_toml(base / "newton_mixed16_native_iter1_xi.toml")
for report in (vector16, replay16, physical16, xi16):
    assert report["complete"] and not report["accepted_entropy"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert vector16["passed"] and replay16["passed"] and xi16["passed"]
assert all(r["complete"] and r["passed"] for r in physical16["rdm_checks"])
assert not physical16["accepted_curve_point"] and not physical16["gaussian_accuracy_certified"]
timing = vector16["rows"][0]
table("vector_action16.tex", "lr", ["完整向量响应比较项目", "数值"],
      [["四组导数最大相对差异", sci(max(r["derivative_relative_error"] for r in vector16["rows"]))],
       ["原残差有限差分最坏误差", sci(max(min(s["relative_error"] for s in r["stencils"]) for r in vector16["rows"]))],
       ["实线性误差", sci(vector16["real_linearity_error"])],
       ["原完整导数耗时中位数（秒）", f'{median(timing["original_seconds"]):.3f}'],
       ["向量响应耗时中位数（秒）", f'{median(timing["vector_seconds"]):.3f}']])
physical_rows = []
for key, label in (("normal", "normal"), ("anomalous", "anomalous"), ("nn", "connected nn")):
    windows = physical16["comparisons"][key]["windows"]
    window = next(w for w in windows if w["r_min"] == 1 and w["r_max"] == 128)
    physical_rows.append([label] + [sci(v) for v in window["max_absolute_errors"]])
table("mixed16_first_step_physical.tex", "lrr",
      ["复关联函数最大绝对误差", "输入", "第一次更新后"], physical_rows)
table("vector_step16.tex", "lr", ["完整更新重放比较项目", "数值"],
      [["原残差相对差异", sci(replay16["comparisons"]["outer_residual"]["relative_error"])],
       [r"右 $A_L$ 系数相对差异", sci(replay16["right_AL_coefficient_relative_error"])],
       [r"左 $A_L$ 系数相对差异", sci(replay16["left_AL_coefficient_relative_error"])],
       ["实际下降相对差异", sci(replay16["comparisons"]["actual_gain"]["relative_error"])],
       ["768 列 Jacobian 构造时间（秒）", f'{replay16["Jacobian_seconds"]:.2f}']])

rotated_replay16 = read_toml(base / "mixed_vector_step16_rotated_v1.toml")
physical4 = read_json(base / "newton_mixed16_native_vector_iter4_gaussian.json")
xi4 = read_toml(base / "newton_mixed16_native_vector_iter4_xi.toml")
comparison4 = read_json(root / "data/direct_chi_curve/independent_mixed16_vector4_physical_comparison.json")
for report in (rotated_replay16, physical4, xi4, comparison4):
    assert report["complete"] and not report["accepted_entropy"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert rotated_replay16["passed"] and xi4["passed"]
assert all(r["complete"] and r["passed"] for r in physical4["rdm_checks"])
assert not comparison4["selected_curve_changed"] and not physical4["accepted_curve_point"]
plot4 = root / "data/direct_chi_curve/independent_mixed16_vector4_physical_comparison.pdf"
sources.append(plot4)
four_rows = []
for label, zh in zip(comparison4["labels"], ("旧参考环境", "此次初值", "第一次更新", "第四次更新")):
    values = []
    for op in ("normal", "anomalous", "nn"):
        row = next(r for r in comparison4["rows"] if r["label"] == label and r["operator"] == op
                   and r["r_min"] == 1 and r["r_max"] == 128)
        values.append(sci(row["max_absolute_error"]))
    four_rows.append([zh] + values)
table("mixed16_four_states.tex", "lrrr",
      ["环境", "normal 最大误差", "anomalous 最大误差", "connected nn 最大误差"], four_rows)

geometry16 = read_toml(base / "mixed_vector_geometry16_best8_v1/report.toml")
anchored16 = read_toml(base / "mixed_vector_anchored16_best8_v1.toml")
for report in (geometry16, anchored16):
    assert report["complete"] and report["passed"] and report["chi"] == 16
    assert report["diagnostic_only"] and not report["accepted_entropy"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
curvature_rows = []
for probe in geometry16["curvature_probes"][:4]:
    assert all(s["valid"] for s in probe["stencils"])
    curvature_rows.append([probe["direction"].replace("_", r"\_")]
                          + [sci(probe["Gauss_Newton_curvature"])]
                          + [sci(s["merit_curvature"]) for s in probe["stencils"][:2]])
table("mixed16_curvature.tex", "lrrr",
      ["方向", "GN 曲率", "$h=10^{-3}$", "$h=3\\times10^{-4}$"], curvature_rows)

curvature16 = read_toml(base / "mixed_vector_curvature16_best8_v1/report.toml")
curvature_xi16 = read_toml(base / "mixed_vector_curvature16_r004_xi.toml")
for report in (curvature16, curvature_xi16):
    assert report["complete"] and report["passed"] and report["chi"] == 16
    assert not report["accepted_entropy"] and not report["accepted_curve_point"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
trial_rows = []
for radius in (0.01, 0.02, 0.04, 0.08):
    pair = [next(r for r in curvature16["trials"] if r["radius"] == radius and r["method"] == method)
            for method in ("original_GN", "weak_curvature")]
    assert all(r["valid"] for r in pair)
    trial_rows.append([f"{radius:.2f}"] + [sci(r["original_residual"]) for r in pair]
                      + [f'{r["gain_ratio"]:.4f}' for r in pair])
table("mixed16_curvature_trials.tex", "rrrrr",
      ["半径", "GN 最大残差", "修正最大残差", "GN 下降比", "修正下降比"], trial_rows)

curvature_physical16 = read_json(base / "mixed_vector_curvature_r004_gaussian.json")
assert curvature_physical16["complete"] and not curvature_physical16["accepted_entropy"]
assert not curvature_physical16["gaussian_accuracy_certified"]
assert all(r["complete"] and r["passed"] for r in curvature_physical16["rdm_checks"])
for path, expected in curvature_physical16["source_hashes"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
curved_physical_rows = []
for op, label in (("normal", "normal"), ("anomalous", "anomalous"), ("nn", "connected nn")):
    window = next(w for w in curvature_physical16["comparisons"][op]["windows"]
                  if w["r_min"] == 1 and w["r_max"] == 128)
    curved_physical_rows.append([label] + [sci(v) for v in window["max_absolute_errors"]])
table("mixed16_curvature_physical.tex", "lrr",
      ["复关联函数最大误差", "修正前", "修正后"], curved_physical_rows)

native_vector_final16 = read_toml(base / "newton_mixed16_native_vector_v1/report.toml")
assert native_vector_final16["complete"] and not native_vector_final16["converged"]
assert native_vector_final16["stop_reason"] == "maxiter"
assert len(native_vector_final16["rows"]) == 49
assert native_vector_final16["exported_best_iteration"] == 8
assert native_vector_final16["final"]["outer_residual"] == min(r["outer_residual"] for r in native_vector_final16["rows"])
for path, expected in native_vector_final16["source_hashes"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

refined_cross16 = read_toml(base / "mixed_vector_curvature16_cross_refined_v1/report.toml")
coupled_trial16 = read_toml(base / "mixed_vector_curvature16_coupled_trial_v1/report.toml")
minimax_control = read_toml(base / "minimax_trust_control_v1.toml")
minimax_trial16 = read_toml(base / "mixed_vector_minimax16_trial_v1/report.toml")
minimax_xi16 = read_toml(base / "mixed_vector_minimax16_r001_xi.toml")
for report in (refined_cross16, coupled_trial16, minimax_control, minimax_trial16, minimax_xi16):
    assert report["complete"] and report["passed"] and not report["accepted_entropy"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
minimax_rows = []
for radius in (0.001, 0.01, 0.04):
    gn = next(r for r in minimax_trial16["trials"] if r["radius"] == radius and r["method"] == "original_GN")
    mm = next(r for r in minimax_trial16["trials"] if r["radius"] == radius and r["method"] == "block_minimax")
    assert gn["valid"] and mm["valid"] and mm["dual_solver"]["passed"]
    assert mm["original_max_improved"] and mm["passes_minimax_merit_rule"]
    minimax_rows.append([f"{radius:g}", sci(gn["original_residual"]), sci(mm["original_residual"]),
                        f'{mm["minimax_gain_ratio"]:.4f}', sci(mm["dual_solver"]["relative_duality_gap"])])
table("mixed16_minimax_trials.tex", "rrrrr",
      ["半径", "GN 最大残差", "minimax 最大残差", "minimax 下降比", "对偶相对间隙"], minimax_rows)

minimax_later_xi = read_toml(base / "minimax16_later_snapshot_v1/audit.toml")
minimax_later_physical = read_json(base / "minimax16_later_gaussian.json")
secant_trial = read_toml(base / "minimax16_secant_trial_v1/report.toml")
secant_roundoff = read_toml(base / "minimax16_secant_field_audit_v1.toml")
secant_replay = read_toml(base / "minimax16_secant_replay_v2/report.toml")
adaptive_curvature = read_toml(base / "newton_weak_curvature16_native_adaptive_v1/report.toml")
for report in (minimax_later_xi, minimax_later_physical, secant_trial,
               secant_roundoff, secant_replay, adaptive_curvature):
    assert report["complete"] and not report["accepted_entropy"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert minimax_later_xi["passed"] and minimax_later_xi["iteration"] == 7
assert all(r["passed"] for r in minimax_later_physical["rdm_checks"])
assert not minimax_later_physical["gaussian_accuracy_certified"]
assert secant_trial["passed"] and secant_roundoff["passed"] and secant_replay["passed"]
assert not adaptive_curvature["converged"] and adaptive_curvature["exported_best_iteration"] == 1
later_physical_rows = []
for op, label in (("normal", "normal"), ("anomalous", "anomalous"), ("nn", "connected nn")):
    window = next(w for w in minimax_later_physical["comparisons"][op]["windows"]
                  if w["r_min"] == 1 and w["r_max"] == 128)
    later_physical_rows.append([label] + [sci(v) for v in window["max_absolute_errors"]])
table("mixed16_minimax_later_physical.tex", "lrr",
      ["复关联函数最大误差", "minimax 第一步", "minimax 第七步"], later_physical_rows)
secant_rows = []
for trial in secant_replay["trials"]:
    if trial.get("valid", False):
        assert trial["dual_solver"]["passed"]
        if trial["passes_merit_rule"]:
            assert trial["actual_gain"] > 100 * trial["projection_merit_bound"]
            assert trial["original_max_improved"] and trial["predicted_gain"] > 0
        secant_rows.append([f'{trial["radius"]:g}', trial["method"].replace("_", r"\_"),
                           sci(trial["original_residual"]), f'{trial["gain_ratio"]:.4f}',
                           "通过" if trial["passes_merit_rule"] else "拒绝"])
    else:
        secant_rows.append([f'{trial["radius"]:g}', trial["method"].replace("_", r"\_"),
                           "--", "--", "检查失败"])
table("mixed16_secant_trials.tex", "rlrrl",
      ["半径", "模型", "原最大残差", "下降比", "试步验收"], secant_rows)

terminal_secant = read_toml(base / "secant16_terminal_audit_v1/audit.toml")
norm_comparison = read_json(base / "minimax16_norm_comparison_v1.json")
terminal_physical = read_json(base / "secant16_final_gaussian_v1.json")
minimax_final = read_toml(base / "newton_block_minimax16_native_v1/report.toml")
stationarity = read_toml(base / "minimax16_final_stationarity_v1/report.toml")
descent_metric = read_toml(base / "minimax16_descent_metric_v1/report.toml")
descent_sweep = read_toml(base / "minimax16_descent_sweep_v2/report.toml")
for report in (terminal_secant, norm_comparison, terminal_physical, minimax_final,
               stationarity, descent_metric, descent_sweep):
    assert report["complete"] and not report["accepted_entropy"]
    for path, expected in report["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert terminal_secant["passed"] and terminal_secant["zero_optimization_steps"]
assert terminal_secant["source_solver_state"] == "FAILED" and not terminal_secant["original_equations_passed"]
assert not minimax_final["converged"] and minimax_final["exported_best_iteration"] == 24
assert stationarity["passed"] and descent_metric["passed"] and descent_sweep["passed"]
norm_labels = ("GN 最佳第 8 步", "minimax 第 24 步", "secant 第 12 步")
table("mixed16_terminal_norm.tex", "lrrr",
      ["独立轨迹", "原最大残差", "行权重实部", "精确行权重相对误差"],
      [[label, sci(s["exported"]["original_maximum"]),
        f'{s["exported"]["row_real"]:.12f}', sci(s["exported"]["complex_relative_error"])]
       for label, s in zip(norm_labels, norm_comparison["summaries"])])
terminal_physical_rows = []
for op, label in (("normal", "normal"), ("anomalous", "anomalous"), ("nn", "connected nn")):
    window = next(w for w in terminal_physical["comparisons"][op]["windows"]
                  if w["r_min"] == 1 and w["r_max"] == 128)
    terminal_physical_rows.append([label] + [sci(v) for v in window["max_absolute_errors"]])
table("mixed16_terminal_physical.tex", "lrr",
      ["复关联最大误差", "secant 初态（minimax 第 7 步）", "secant 第 12 步"], terminal_physical_rows)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
for ax, label, summary in zip(axes, ("GN", "Block minimax", "Secant minimax"), norm_comparison["summaries"]):
    seq = [r for r in norm_comparison["rows"] if r["source"] == summary["source"]]
    steps = [int(r["state"].split("_")[-1]) for r in seq]
    ax.plot(steps, [r["original_maximum"] for r in seq], color="C0")
    ax.set(xlabel="Update", ylabel="Original maximum residual", title=label)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    right = ax.twinx()
    right.plot(steps, [r["complex_relative_error"] for r in seq], color="C1", ls="--")
    right.set_ylabel("Exact norm relative error", color="C1")
    right.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=.2)
fig.suptitle("chi=16 diagnostics: blue=residual, orange=Gaussian norm error; none converged")
for ext in ("png", "pdf"):
    fig.savefig(out / f"mixed16_norm_trajectory.{ext}", dpi=180)
plt.close(fig)

result = dict(complete=True, accepted_entropy=False, rows=rows,
              chi16_terminal_secant=terminal_secant, chi16_norm_comparison=norm_comparison,
              chi16_terminal_physical=terminal_physical, chi16_minimax_completed=minimax_final,
              chi16_stationarity=stationarity, chi16_descent_metric=descent_metric,
              chi16_descent_sweep=descent_sweep,
              chi16_minimax_later_xi=minimax_later_xi,
              chi16_minimax_later_physical=minimax_later_physical,
              chi16_secant_original_trial=secant_trial,
              chi16_secant_roundoff_audit=secant_roundoff,
              chi16_secant_replay=secant_replay,
              chi16_adaptive_curvature_completed=adaptive_curvature,
              chi16_refined_cross_curvature=refined_cross16,
              chi16_coupled_subspace_trial=coupled_trial16,
              minimax_analytic_control=minimax_control,
              chi16_minimax_trial=minimax_trial16,
              chi16_minimax_xi=minimax_xi16,
              chi16_native_vector_completed=native_vector_final16,
              chi16_curvature_physical=curvature_physical16,
              chi16_curvature_trials=curvature16,
              chi16_curvature_candidate_xi=curvature_xi16,
              chi16_best8_geometry=geometry16,
              chi16_anchored_response=anchored16,
              center_derivative_control=center_control,
              newton_merit=newton_merit,
              newton_guard_control=guard_control, newton_guard_run=guard_run,
              mixed_response_controls=mixed_controls, mixed_small_control=mixed_small,
              mixed_export_control=mixed_export,
              mixed_completed_solver_report=mixed_progress,
              mixed_physical_regression=mixed_physical,
              mixed_bare_regression=mixed_bare,
              mixed_bare_comparison=mixed_bare_comparison,
              mixed_chi8_continuation=mixed_continue,
              mixed_chi8_recovery=mixed_recovery,
              mixed_chi8_recovery_export=mixed_recovery_export,
              mixed_chi8_cap_scan=mixed_caps,
              failed_chi16_derivative_controls=failed16_controls,
              chi16_metric_action_check=metric16,
              chi16_full_action_check=full_action16,
              chi16_vector_action_check=vector16,
              chi16_vector_step_replay=replay16,
              chi16_first_step_physical=physical16,
              chi16_first_step_xi=xi16,
              chi16_rotated_vector_step_replay=rotated_replay16,
              chi16_fourth_step_physical=physical4,
              chi16_fourth_step_xi=xi4,
              chi16_four_states=comparison4,
              trajectory=trajectory, xi_trajectory=xi,
              source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sources}, job_id=os.environ["SLURM_JOB_ID"])
(out / "data_snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
with (out / "gaussian_diagnostics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
