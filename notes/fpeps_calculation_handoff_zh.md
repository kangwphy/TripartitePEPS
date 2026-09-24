# fPEPS 计算交接：现状、数据与 Slurm 提交

最新补充：χ=12 的 depth=4096 续算已经完成，bare S̃=0.8377789765909256，
长度漂移 1.82e-12、相位误差 2.80e-9，收缩检查通过；原生输入未收敛的标记不变。

**2026-09-17 bare 扫描更新：χ=6/8 的 S̃=0.524257409377/0.578186036365，
长度、相位通过；χ=12/16 暂为实部诊断 0.837778976592/0.855621333865。
χ=12 长度尚未通过，已保持同一几何续算到 depth=4096（11268585）；
χ=16 相位约 6.81e-7 未过。详细表与状态见符号 note 的实现补充。
“收缩通过”仍不代表重构或目标态物理精度认证。**

**2026-09-17 方法纠正：本文原有熵结果及第 5.6 节命令属于 grown-tail / grown-turn，不能当作用户要求的 bare direct。用户明确要求改用实际两列／三列固定点的 bare 构造。旧数据和命令仅保留作历史诊断；新实现位于 `benchmark/bare_direct_core.jl`，首轮 χ=4 记录在 `data/bare_direct_20260917/`。尚无已验收 bare 熵值。**

新的 bare 定义、检查状态和可复制提交命令见
[bare/direct 计算记录](../code2d/fpeps/data/bare_direct_20260917/README.md)。
2026-09-17 已按用户明确要求启动 bare χ=4；下文旧快照中“等待恢复确认”不再适用于这批新作业。

**已完成更新：gapless w=1、χ=4 的 bare/direct S̃=0.4396138979194575，
旧 grown-tail 为 0.4347199926434087（新值高约 1.126%）。完整四副本／周期分解
误差 3.20e-15，长度漂移 1.22e-10，相位误差 5.57e-13，均通过。
重构闭合 logz 偏差 3.27e-4，仍未通过一致性门槛，保持诊断标记。
二站点 Gaussian RDM 误差由 1.914e-3 降至 1.344e-3，但 r=31 的 anomalous
误差仍为 98.48%，不能宣称已获得物理精度认证。**
详见[符号与实现 note 的新 benchmark 节](fpeps_signs_zh/implementation.md)及
[新旧关联图](../code2d/fpeps/data/direct_chi_curve/bare_vs_grown_chi4_correlations.png)。

更新：2026-09-13。本文是当前状态快照，历史日志中的“running”不能当作当前队列状态。

2026-09-14 补充：[w=1 的 north/south 转移算子局域证明](fpeps_direction_intertwiner_w1_zh.md)已完成。该证明不覆盖 south→rotated；其简单 W 整行变换在新检查中失败，因此下文历史三方向流程不能据此获得新的物理认证。

## 1. 在计算什么

目标是在保留原有 RK-Ising、TFIM 和 Gaussian 路线的前提下，用严格 D=2 的 KSVC fPEPS 计算三方纠缠量

\[
\widetilde S=-\log\frac{Z_4Z_1^2}{Z_{2A}Z_{2B}Z_{2C}}.
\]

目前重点是 gapless 参数 w=1。用户要求的主路线是无限平面 iPEPS 的边界 MPS 与 bare/direct 收缩，G、B 分别来自实际空间边界的两列、三列固定点；下表的历史熵并未采用这一构造，而是 grown-tail。χ 是边界 MPS 的总 bond dimension；(χ_even, χ_odd) 表示虚拟偶、奇宇称块的维数，其和才是 χ。

direct 计算中的 depth 是把无限边界构造出的 replica 网络向长长度推进、检验渐近收敛的参数，不是把原来的 iPEPS 改成有限方格系统。Gaussian ring 的 L 则是另一条环境初态构造路线的有限参考环周长，两者不能混用。

另一路已探索：exact Gaussian boundary → 有限 ring 参考 → 中间 MPS 截断 → 一般 MPS 的 fidelity per site 优化 → 恢复原始费米子表示 → 原生边界方程优化。该路线仍是辅助环境构造，不能仅凭 fidelity 高就接受最终熵。

用户当前优先级：**先把 χ=4、6、8 算清楚，检查能否从 χ=12 得到更好的小 χ 初态；暂停更大 χ 的新计算。** 本次交接没有提交作业。此前剩余小 χ 作业被当前账号取消，恢复计算的确认尚未收到。

## 2. 当前哪些结果可信

必须区分“有限 χ 收缩通过已有数值检查”和“准确逼近目标态”。前者不自动推出后者。

| 参数与 χ | 当前数值 | 判断 |
|---|---:|---|
| Gapped w=0.25，χ=32 | 0.0164299369 | 与 Gaussian 参考 0.0164301443 接近，有 benchmark 支持；未认证所有输出位数 |
| Gapped w=0.25，χ=40 | 0.0164328001 | 与同一参考相差约 2.66×10⁻⁶；这是已完成的历史结果，当前不续跑 χ=40 |
| Gapless χ=4 | 0.4347199926 | 原生边界残差 <10⁻¹²，已有长度、相位检查通过；物理准确性未认证 |
| Gapless χ=6 | 0.5178839905 | 同上 |
| Gapless χ=8 | 0.5616249950 | 同上 |
| Gapless χ=12 | 0.8332802879 | 诊断值；原生残差约 2.76×10⁻⁶，未通过环境收敛 |
| Gapless χ=16、24 | 约 0.79277、0.80377 | 诊断实部；环境、长度或相位检查仍有失败 |
| Gapless χ=32，polish64 分支 | 0.7962634686 | 原生残差已到 8.07×10⁻¹⁴，但 replica 相位误差约 3.14×10⁻⁵，熵仍不接受 |

χ≥12 的 gapless 熵均不能作为已验收点。χ=32 即使实部的 512→1024 长度漂移很小，也不能绕过相位检查。

已有 Gaussian 关联函数核对提供了具体疑点：w=1 时横向非零距离的 normal correlator 应为零，而旧 χ=4、6、8 的最近邻绝对误差分别约 0.001914、0.003156、0.003410，没有随 χ 改善。r=31 的 anomalous correlator 相对误差约为 99.1%、88.0%、75.1%；但 r 已超过这些环境的 ξ，不能只凭这项长距离误差就断定错误分支。

χ=12 在 r=31 的 anomalous 相对误差约 3.83%，这一项更好；其密度关联和边界残差仍未全部过关，不能据此宣布 χ=12 正确。

旧 χ=4、6、8 的 S–ln ξ_physical 三点斜率约 0.154699，Gaussian 的相应 ln L 参考约 0.155019。**这只是探索性吻合，不能用来证明小 χ 环境、熵或渐近标度正确。**

## 3. χ=12 是怎么得到的，小 χ 实验发现了什么

图上 χ=12 的源文件是：

`data/stable_boundary_20260912/real_native_chart/chi12/diagnostic_boundary.jls`

它来自旧 χ=8 扩维后的 Schur 环境跟踪、残差优化和实参数表示转换，不是从 χ=10 的异常分支直接取来的。实表示处理改善了 replica 相位，但原生残差仍停在约 2.76×10⁻⁶。

随后从这个 χ=12 向下截断：

- balanced 分支：χ=4→(2,2)，χ=6→(3,3)，χ=8→(4,4)。优化后的关联函数回到了旧 χ=4、6、8 的值，差异约 10⁻¹²。
- global Schmidt 排序分支：χ=6→(2,4)，χ=8→(3,5)。优化最后残差约 9.14×10⁻⁴、2.84×10⁻⁴，均未收敛。取消前保存的完整两点 RDM 的 Gaussian 最大元误差约 0.003928、0.003652，比对应旧解更差；完整长距离测量尚未完成。
- global 初态的物理 transfer 有两个不同相位、相同模长的主本征值，不能随意选一个归一化环境。优化后这个主谱简并消失；不能把初态的谱问题直接套到终态。

**上述 downward 实验有一个尚待修正的初态处理问题。** 本地 MPSKit 的 SvdCut 在同时投影两个虚拟腿后，某个构造路径只恢复右规范；Newton 参数化却假设输入 AL 已左正交。于是 x=0 时的 `A * invsqrt(A' * A)` 也可能改变物理张量。

已测到 χ=4 的一次实表示投影使有限 L=3、4 的归一化波函数射线发生最大约 0.0051 的变化。源代码审查支持上述机制，但保存初态的直接 AL/AR 正交性测量和完整规范化修复测试尚未运行。

因此，“优化后回到旧解”这个终态观察仍成立，**但它还不是严格保持物理初态的多初态比较，更不能证明旧解最优。**

下一步只做 χ=4：完整 LR 规范化 → 检查波函数保持、费米子表示转换和零 Newton 步 → 再测初态与优化终态的关联函数。通过后才推广 χ=6、8。

## 4. 路径索引

所有下文命令从以下目录运行：

```bash
cd /ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps
```

| 路径（相对上述目录） | 内容 |
|---|---|
| `src/` | fPEPS 计算实现 |
| `benchmark/` | 隔离的诊断、环境优化、符号和 Gaussian 检查脚本 |
| `jobs/run_cpu.sh` | Julia 的 Slurm 包装脚本 |
| `jobs/run_python.sh` | Python 测量/绘图的 Slurm 包装脚本 |
| `data/direct_chi_curve/` | 已有图和曲线数据；继续使用这个图目录 |
| `data/stable_boundary_20260912/README.md` | 长篇历史实验记录；顶端是状态说明 |
| `data/stable_boundary_20260912/active_gaussian_ring_blas/jobs.json` | 历史作业号与 `submission_commands` 精确参数 |
| `data/stable_boundary_20260912/active_gaussian_ring_blas/small_chi_from_plotted12/` | 当前小 χ 实验，以下简称 B |
| `B/canonicalization_review.md`、`.json` | 规范化问题的源码证据、限制和脚本哈希 |
| `B/canonicalization_check_plan.json` | 已准备、未提交的单 χ=4 检查计划 |
| `B/seeds/chi4/diagnostic_boundary.jls` | 下一步的原始截断初态 |
| `B/native/`、`B/seeds_physical/`、`B/native_physical/` | 第一轮 balanced 结果，保留作历史诊断 |
| `B/global_rank/partial_rho2_audit.json` | 取消前保存的 global 分支两点 RDM 核对，明确是部分结果 |

B 是本文的路径缩写，不是磁盘上名为 B 的目录。长历史 README 中有已过时的运行状态，以 Slurm 实际状态及对应 completion audit 为准。

Gaussian 代码在 `/ix/zdai/kangw/PEPS3EE/Gaussian`；原有 LMPS/VUMPS 项目在 `../lmps/vumps`；符号笔记在 `../../notes/fpeps_signs_zh`。

## 5. Slurm 提交约定

数值计算和绘图均通过 Slurm，**不要在登录节点直接执行 `bash jobs/run_cpu.sh ...`、Julia 优化或 Python 大矩阵计算**。登录节点可查看文本、哈希、队列和提交作业。

包装脚本已经设置：cluster `htc`，partition `preempt`，`--no-requeue`，1 node/1 task；Julia 使用 `../lmps/vumps` 项目与现有 depot。无需重新安装环境。日志路径固定为：

```text
logs/<job-name>_<job-id>.out
logs/<job-name>_<job-id>.err
```

以下是一轮**新的 χ=4 修正实验的命令模板，尚未跑过**。采用新的结果子目录，防止覆盖历史初态与已核对结果；图仍放在原来的 `direct_chi_curve`。每个步骤先查看终态与科学检查，再执行下一步，不要把整篇命令一次性粘贴执行。

先在同一个 Bash 会话设置路径和提交函数：

```bash
cd /ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps
mkdir -p logs
branch=data/stable_boundary_20260912/active_gaussian_ring_blas/small_chi_from_plotted12
run_root=$(mktemp -d "$branch/resume_chi4_XXXXXXXX")
printf '%s\n' "$run_root"

# 保存实际命令和 Slurm 返回值；函数本身不提交任何作业。
submit_small() {
    local submitted
    printf '%q ' sbatch --parsable --cluster=htc --partition=preempt \
        --no-requeue --exclude=htc-n77,htc-n79,htc-1024-n0 \
        --cpus-per-task=2 --mem=12G --time=00:20:00 "$@" \
        >> "$run_root/submissions.sh"
    printf '\n' >> "$run_root/submissions.sh"
    submitted=$(sbatch --parsable --cluster=htc --partition=preempt \
        --no-requeue --exclude=htc-n77,htc-n79,htc-1024-n0 \
        --cpus-per-task=2 --mem=12G --time=00:20:00 "$@") || return
    printf '%s\n' "$submitted" >> "$run_root/jobs.txt"
    printf '%s\n' "$submitted"
}
```

`--parsable` 可能返回 `作业号;htc`；传给 `sacct` 或 dependency 时只取分号前的数字。若重开终端，重新设置 `branch`、`run_root`（使用刚才已经创建的目录）和函数，不要误建一套新的结果目录。

### 5.1 第一项：χ=4 完整规范化检查

```bash
canonical_job=$(submit_small --job-name=fp-chi4-canonicalization-check \
    jobs/run_cpu.sh benchmark/recanonicalize_small_native_seed.jl \
    "$branch/seeds/chi4/diagnostic_boundary.jls" \
    "$run_root/canonicalized/chi4")
canonical_id=${canonical_job%%;*}
printf '%s\n' "$canonical_id"
```

这项不做优化、不计算熵。成功应写出：

```text
<run_root>/canonicalized/chi4/diagnostic_boundary.jls
<run_root>/canonicalized/chi4/canonicalization.toml
```

报告必须 `complete=true`，并检查：虚拟宇称空间不变；修复后 AL、AR 正交性和中心一致性误差 <10⁻¹⁰；原始/恢复张量的表示往返误差 <10⁻¹⁰；有限 L=3、4、两种 seam 的波函数射线误差 <10⁻¹⁰；零 Newton 步的张量变化 <10⁻¹⁰。**这些有限环检查不是无限链波函数等价的完整证明。** `native_converged=false` 本身不意味着规范化失败，因为初态还没有优化。

旧计划中的默认输出是 `B/canonicalized/chi4`；这里显式改用新的 `run_root/canonicalized/chi4`，因此不要去旧路径找新报告。

### 5.2 查看状态、日志和失败原因

```bash
squeue -M htc -u "$USER"
sacct -M htc -j "$canonical_id" \
    --format=JobID,JobName%36,State%30,ExitCode,Elapsed,MaxRSS
tail -n 60 "logs/fp-chi4-canonicalization-check_${canonical_id}.out"
tail -n 60 "logs/fp-chi4-canonicalization-check_${canonical_id}.err"
cat "$run_root/canonicalized/chi4/canonicalization.toml"
```

队列中消失不等于成功，必须看 `sacct` 的 `COMPLETED`、`ExitCode=0:0` 和最终报告。`sstat` 不接受 `-M htc`。`TIMEOUT`、`CANCELLED`、导数检查失败、谱检查失败是不同原因；保留各自日志和检查点，不要把残留 CSV 当作完整测量。

如果确实要取消某个本轮作业，使用 `scancel -M htc <具体作业号>`，不要按整个账号批量取消。取消或失败后，先核对终态再决定是否重提；`--no-requeue` 不会自动恢复。

### 5.3 规范化通过后：准备优化清单

下面只读已完成报告并生成一个单 χ=4 的 `seeds.toml`，可在登录节点执行。它让现有 batch 优化器继续执行原生环境完整谱检查。

```bash
python3 - "$run_root" <<'PY'
import hashlib, json, sys
from pathlib import Path
from pip._vendor import tomli
r = Path(sys.argv[1])
folder = r / 'canonicalized'
report = tomli.loads((folder/'chi4/canonicalization.toml').read_text())
assert report['complete'] and report['chi'] == 4
assert report['zero_newton_step_tensor_change'] < 1e-10
source = folder/'chi4/diagnostic_boundary.jls'
digest = hashlib.sha256(source.read_bytes()).hexdigest()
plan = folder/'seeds.toml'
with plan.open('x') as f:
    f.write('complete = true\n\n[[cases]]\nchi = 4\nprepared = true\n')
    f.write('seed = '+json.dumps(str(source))+'\n')
    f.write('seed_sha256 = '+json.dumps(digest)+'\n')
PY
```

然后提交优化，保留第一轮 balanced 实验的数值参数：

```bash
native_job=$(submit_small --job-name=fp-chi4-canonical-native \
    --cpus-per-task=4 --mem=16G --time=00:45:00 \
    --export=ALL,FPEPS_NEWTON_COORDINATES=unweighted_tangent,FPEPS_NEWTON_CENTER_POWER=0,FPEPS_NEWTON_PARALLEL=1,FPEPS_NEWTON_SEARCH=best,FPEPS_NEWTON_FD_START=3e-5,FPEPS_NEWTON_JAC_TOL=1e-5,FPEPS_NEWTON_FD_INTERLEAVE=1,FPEPS_NEWTON_INNER_TOL=1e-15,FPEPS_NEWTON_REG_INTERLEAVE=1,FPEPS_NEWTON_FD_ORDER=4,FPEPS_NEWTON_NATIVE_TOL=1e-13 \
    jobs/run_cpu.sh benchmark/optimize_small_chi12_seeds.jl \
    "$run_root/canonicalized" "$run_root/native" 40)
native_id=${native_job%%;*}
```

检查 `native/batch.toml` 的每个 case、`native/chi4/result.toml`、`seed_spectrum.toml` 和迭代日志。batch 脚本会捕获单 case 异常并保存诊断终态，所以 **Slurm COMPLETED 和 batch 的 complete=true 都不等于 native_converged=true**。必须查看 case 的 `completed`、`error`、`native_converged` 与实际残差。导数精度检查失败时不能放宽门槛把它算作成功。

### 5.4 同样测量规范化初态和优化终态

需要对比两个阶段，不能只测优化终态。下面先设 `stage=canonicalized`；完成后改成 `stage=native` 再执行这一节。

```bash
stage=canonicalized
orient_job=$(submit_small --job-name="fp-chi4-${stage}-orient" \
    jobs/run_cpu.sh benchmark/transport_diagnostic_pair.jl \
    "$run_root/$stage/chi4/diagnostic_boundary.jls" \
    "$run_root/${stage}_oriented/chi4" all)
orient_id=${orient_job%%;*}
```

这里的源是 **south、原始完整两模费米子边界**。`transport_diagnostic_pair.jl` 会保存 north/south/rotated 三个方向和各自真实收敛标记。不要用只接受 north 的 `transport_restored_active_pair.jl`，也不要再做 active-vacuum 恢复，否则会混淆表示。

确认 orientation 完成后提交：

```bash
measure_job=$(submit_small --job-name="fp-chi4-${stage}-physical" \
    jobs/run_cpu.sh benchmark/measure_small_branch.jl \
    "$run_root/${stage}_oriented/chi4" \
    "$run_root/${stage}_physical/chi4")
measure_id=${measure_job%%;*}
```

输出包括一至三点 RDM 的实虚部 CSV、`modes.toml`、`correlations.toml`。脚本检查完整物理谱与归一化环境，再计算 normal/anomalous/nn 的 r=1…128 复数关联。**只有 CSV 而没有最终 `correlations.toml`，仍是未完成测量。**

两阶段完整测量后，提交 Gaussian RDM 比较：

```bash
gaussian_job=$(submit_small --job-name=fp-chi4-canonical-gaussian \
    jobs/run_python.sh benchmark/compare_active_strip_rdm.py \
    "$run_root/rdm_comparison" \
    "$run_root/canonicalized_physical/chi4" \
    "$run_root/native_physical/chi4")
```

结果为 `rdm_comparison/gaussian_strip_comparison.json`。其中 local purification entropy 是小区域 RDM 的检查，**不是**无限 direct 的 \(\widetilde S\)。关联函数的独立共轭/Wick 检查可对每阶段分别提交：

```bash
wick_job=$(submit_small --job-name="fp-chi4-${stage}-wick" \
    jobs/run_cpu.sh benchmark/current_correlation_adjoint_wick.jl \
    "$run_root/${stage}_physical/chi4/correlations.toml" \
    "$run_root/${stage}_adjoint_wick/chi4")
```

此处 `stage` 仍应分别取 canonicalized 和 native。Wick 偏差衡量有限 χ 结果偏离 Gaussian 的程度；小偏差不单独证明所有费米子符号正确。

### 5.5 记录三个 ξ

物理 ξ 与模式残差已经包含在每阶段的 `modes.toml` 中。其余两个可以显式提交：

```bash
paired_job=$(submit_small --job-name="fp-chi4-${stage}-pairedxi" \
    jobs/run_cpu.sh benchmark/bare_paired_boundary_xi.jl \
    "$run_root/${stage}_paired_xi.toml" \
    "$run_root/${stage}_oriented/chi4")
self_job=$(submit_small --job-name="fp-chi4-${stage}-selfxi" \
    jobs/run_cpu.sh benchmark/boundary_sector_xi_audit.jl \
    "$run_root/${stage}_self_xi.toml" \
    "$run_root/${stage}_oriented/chi4/diagnostic_north.jls")
```

| 名称 | 定义与限制 |
|---|---|
| ξ_boundary / self | 同一个边界 MPS 的自重叠 transfer；旧图使用 even sector |
| ξ_paired | 实际 north/south 两边界的 mixed transfer，不夹中间 PEPS 列；保存 even/odd 谱，旧比较取 odd 衰减长度 |
| ξ_physical | north–PEPS 列–south 的物理传播通道；必须结合具体算符的非零 residue 选模式，旧比较使用 fermion 模式 |

一般按 \(\xi=-a/\ln|\lambda/\lambda_0|\) 计算，并记录单位胞步长 a、宇称、算符耦合和本征对残差。不同 transfer/sector 的 ξ 不能直接当作同一个量。主模简并、非衰减且耦合的模式不能被悄悄略过后宣称有限 ξ。

### 5.6 何时计算 direct 熵

先完成上面的规范化、环境与物理 benchmark。若仅作诊断也必须保留未通过标记；进入可信曲线前需要全部检查。以下只对优化后的三个方向运行，逐项完成后再下一项：

```bash
geometry_job=$(submit_small --job-name=fp-chi4-canonical-geometry \
    jobs/run_cpu.sh benchmark/build_verified_active_geometry.jl \
    "$run_root/native_oriented/chi4")
```

这个入口要求三个方向原生收敛，并构造完整 tail/seam 谱。确认完成后：

```bash
sign_job=$(submit_small --job-name=fp-chi4-canonical-signs \
    --time=01:00:00 \
    jobs/run_cpu.sh benchmark/checkpoint_signed_geometry_audit.jl \
    "$run_root/native_oriented/chi4" 2)
```

检查 `signed_geometry_audit.toml` 的 complete、passed 和同一 geometry 哈希；应覆盖 depth=1、2 的 Z1/Z2A/Z2B/Z2C/Z4 全部十项。完成后才能提交长度与相位检查：

```bash
entropy_job=$(submit_small --job-name=fp-chi4-canonical-length \
    --time=01:00:00 \
    jobs/run_cpu.sh benchmark/measure_saved_audited_geometry.jl \
    "$run_root/native_oriented/chi4" 1024 2)
```

检查 `entropy.toml`：各 replica 的归一化相位误差 ≤10⁻⁸；最近三档的熵漂移和各 log sector 线性尾拟合残差 ≤10⁻⁷；端点残差 ≤10⁻⁶。完整符号比较误差要求 <10⁻¹⁰。仅组合相位很小，或只取实部、绝对值后平滑，都不能替代各 sector 的检查。

脚本资源是小 χ 起点，不保证每个新输入都在时限内完成。若超时，应根据最后保存进度决定续测方式，不能把超时当作物理失败，也不能直接重跑覆盖历史输出。

## 6. 画图、数据验收与继续 χ=6、8

已有图位于 `data/direct_chi_curve/`：`current_stilde_lnxi.*`、`current_physical_correlations.*`、`small_chi_from12_comparison.*`。第一轮小 χ 对比图可以用下面命令重画，但会覆盖同名历史派生图，使用前应保存快照：

```bash
plot_job=$(submit_small --job-name=fp-small12-historical-plot \
    jobs/run_python.sh benchmark/compare_small_chi_from12.py)
```

**该绘图脚本当前固定读取旧的 seeds_physical/native_physical 目录，不会自动读取本轮 run_root。** 因此重画它不是画修正后的结果。新数据完整后需显式扩展绘图输入与来源哈希，并在现有 `direct_chi_curve` 下输出修正版；不要把旧图标成新实验，也不要把新结果复制覆盖旧测量来迁就绘图脚本。

验收时至少保留：源 tensor 哈希、脚本/参数、作业号及终态、三方向残差、完整谱和所选主模残差、Gaussian RDM/关联误差、三个 ξ 的定义与 residue、完整符号和长度/相位报告。normal 精确值为零时报告绝对误差；connected nn 使用该环境自己测出的密度平方作扣除，不默认代入 1/4。

保护 `data/direct_chi_curve/points.csv`、`points.toml` 和原 Gaussian 参考。现有 `physical_benchmark_certified=false` 必须如实保留；没有通过检查的点继续标诊断，不能混入可信拟合。

χ=4 修正检查通过并分析完之后，才把同样过程应用到 χ=6、8：对应 source 为 `B/seeds/chi6`、`B/seeds/chi8`，规范化脚本已允许这三个 χ。重新生成各自哈希清单；不要直接复用上面写死 χ=4 的 `seeds.toml`。更大 χ 和其它分支暂不恢复。

## 7. 关键证据文件

- 当前数值与可信标记：`data/direct_chi_curve/points.csv`、`current_stilde_lnxi.json`。
- 旧态 2688 个复数关联核对：`data/stable_boundary_20260912/active_gaussian_ring_blas/current_physical_correlations/mean_corrected/completion_audit.json`。
- χ=12 来源与第一轮小 χ 比较：`B/plan.json`、`B/native_plan.json`、`B/balanced_endpoint_audit.json`、`B/rdm_comparison/gaussian_strip_comparison.json`。
- 初态表示问题：`B/chi4_truncation_chart_audit/chart_audit.toml`、`B/canonicalization_review.json`。
- 已准备未运行的检查脚本：`benchmark/recanonicalize_small_native_seed.jl`；写本文时 SHA256 为 `d03b8d6b285f369a1efb52378be39189bb0ba83b2c3e26c84153dd11a7138ecb`。
- 当前 χ=4 源 SHA256：`0be3e3eaef175595a7e65ff53cf9a64e8e8dedbf7ff20c8f26883fc6f84aa2c4`。
- 取消与部分结果：`B/external_cancellation.json`、`B/global_rank/partial_rho2_audit.json`。

本文中的命令已按现有脚本接口核对；修正后的整条数值流程尚未执行，不能把文档中的步骤当作已经验证成功的结果。
