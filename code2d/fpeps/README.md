# Fermionic iPEPS：带符号的区域环境与 S_tilde

整理后的入口见 [BARE_ENTRYPOINTS.md](BARE_ENTRYPOINTS.md)，测试入口见
[test/README.md](test/README.md)。下文保留研究历史；已归档的旧测试路径不再是
默认入口，历史非 preempt 提交示例也不得直接使用。

本目录独立实现 `PEPS/fPEPS_renyi_2.pdf` 中的严格 D=2 Gaussian fPEPS。
局域张量、虚键和 environment 使用 TensorKit `FermionParity`；没有修改原有 RK/TFIM 模块。

**当前计算入口（2026-09-21）：无限边界环境与 bare/direct 熵。**
双边环境的实际算法、验收状态与提交方式见
[独立双边求解记录](benchmark/independent_bivumps/README.md)，技术推导见
[独立中文 note](../../notes/fpeps_mpbp_technical_zh/main.pdf)。
目前集中处理 gapless `w=1, chi=16` 的收敛；新的最大残差求解完成 24 步仍未收敛，
割线修正完成 12 次更新仍未收敛，且末尾报告报错；保存张量已独立复核。
两条轨迹的 Gaussian 行权重误差均增大，尚无新的可信熵点。目录名中的 bivumps 不代表已经实现
标准 biVUMPS 更新；当前使用两条独立 MPS 的中心残差方程。
熵使用 bare/direct，不把 grown-tail 或 endmap 结果替代进来。
现有曲线保存在 `data/direct_chi_curve`；未通过验收的环境不替换曲线数据。

以下保留较早的有限窗口符号 benchmark 与接口说明。该路线使用
固定物理张量 → 有限窗口区域 boundary MPS → 显式转角 center →
occupation parity-sector 求和 → six-R 收缩。分别检查窗口 L 和总环境维数 χ 的极限。
局域张量是一格点平移不变的 iPEPS；单个有限窗口测量尚不是无限系统极限。

本次作业与 χ=2,4,8 的数据见 [验收记录](VALIDATION.md)。
同一有隙态的有限 Gaussian 尺寸扫描、能隙证明和 iPEPS 对照见
[gapped benchmark](benchmark/README.md)。
旧的 stationary LMPS 路径与 Gaussian 有明显差异，不能作为物理熵结果使用。
新的真实区域路径已在 L=2,4 不截断时逐 sector 与 Gaussian 达到 1e-14 内一致；
大 L/χ 审计的最新状态见 [审计记录](benchmark/AUDIT_20260910.md)。
已通过的 w=1/8 benchmark 见 [自动验收](data/gapped_convergence_audit/acceptance.json)
及 [收敛图](data/gapped_convergence_audit/convergence.png)。

## 有限窗口物理熵

```bash
# L=16；总 χ=16（even+odd），物理 D=2。
sbatch jobs/run_cpu.sh bin/run_finite_stilde.jl 16 16 data/my_finite_stilde
```

该入口默认 `FPEPS_BOND_WEIGHT=0.125,0.125`、`FPEPS_GAUGE=balanced`，
是具有严格有隙 parent 的 D=2 态。balanced 将每条键权重均分到两端。
可用 `FPEPS_BOND_WEIGHT` 改变权重，用 `FPEPS_GAUGE=outgoing` 检查等价虚腿 gauge
下 χ→∞ 的一致性。`FPEPS_ROTATE_180=true` 检查另一方向。
`CHI=0` 关闭截断，只适合小窗口。输出目录必须不存在。
可选第四、第五个参数是 `WIDTH` 和上半区分割位置 `CUT`；高度仍为 L，且 L 为偶数。
默认方形等分。原始临界 w=1 的 2×2 虚真空闭合是零态，不能归一化；
可用非零的 3×2 闭合作小系统对照：

```bash
FPEPS_BOND_WEIGHT=1 FPEPS_GAUGE=outgoing \
  sbatch jobs/run_cpu.sh bin/run_finite_stilde.jl 2 0 data/my_critical_3x2 3 1
```

```julia
include("src/FermionicPEPS.jl")
using .FermionicPEPS
peps = ksvc_ipeps(;bond_weight=(0.125,0.125),bond_gauge=:balanced)
r = measure_finite_fpeps_stilde(peps;L=16,chi=16)
r.stilde
r.diagnostics
```

`bond_gauge=:outgoing` 与 balanced 在精确收缩时是同一态，截断误差可以不同。
构造器的历史默认 `bond_weight=(1,1)`、outgoing gauge 没有改变。

分区为 A 左上象限、B 右上象限、C 下半平面，外边界为虚腿真空，共一个三分交点。
Julia 接口也接受 `width`、`cut`，支持 A/B 不等宽的矩形窗口。
保留每个区域真正的 MPS center，不把它替换为单位 cap。旋转后按 dual strand 处理
并行 ket/bra 接口的 pivotal twist；与 occupation replica 的 parity strings 分开。
输出 `run.toml`、区域和各 sector checkpoint、`result.jls`、`sectors_scaled.csv` 和
`stilde.csv`。sector 保存复数值与 logscale，避免面积归一化溢出。
每行的 SVD 平方范数损失是截断诊断，不是熵误差上界。
单点状态始终标为 `finite_window_measured`，不自动声称 L/χ 已收敛。

## 旧 stationary LMPS 诊断入口

在本目录运行（数值计算全部经 Slurm）：

```bash
sbatch jobs/run_cpu.sh bin/run_stilde.jl 2 2 data/my_stilde

# 使用前一次保存的三个区域环境，改变 seam 长度。
FPEPS_DEPTHS=16,32,64,128 \
  sbatch jobs/run_cpu.sh bin/run_stilde.jl 2 2 data/my_longer_scan data/my_stilde/boundaries
```

输出目录必须不存在。`2 2` 表示 even/odd 各两个保留态，总 χ=4；不是物理 D，物理 D 固定为 2。
输出含 `run.toml`（约定、源码/Manifest 哈希、分支尝试、收敛状态）、`prepared.jls`、
三个 `boundaries/A.jls,B.jls,C.jls`、`depth_scan.csv`、逐长度的完整复数 sector checkpoint
和 `result.jls`。checkpoint 使用相同固定 Julia/包环境重载。

默认 A、B 共用同一 north 固定点分支，但 A 的左 rail 和 B 的右 rail 不相等；C 单独求 south 环境。
这不假设张量旋转/反射对称。`FPEPS_INDEPENDENT_NORTH=true` 可以检查独立 north 初始化。
随机初始化可能选到不兼容的 graded boundary sector；密度和单副本本征值一致不足以识别它。
程序要求三个单副本接缝都有超过环境误差的谱间隙；新环境最多重试 `FPEPS_BOUNDARY_TRIALS=8`
个有记录的种子。用户提供的 checkpoint 不会被替换，不明确的 sector 会报错。

可配置 `FPEPS_SEED`、`FPEPS_BOUNDARY_TOLERANCE`（默认 1e-8）、
`FPEPS_BOUNDARY_MAXITER`（1000）、`FPEPS_DEPTHS`（8,16,32,64）及 `FPEPS_TOLERANCE`（1e-7）。
`FPEPS_ACCEPT_UNCONVERGED=true` 只允许保存未通过长度极限的结果，并明确标记；
不会绕过边界、sector、相位或数值抵消检查。
`FPEPS_BOND_WEIGHT` 接受一个各向同性实数或两个逗号分隔的水平/竖直权重（默认 `1,1`）；
`FPEPS_ROTATE_180=true` 对物理张量做 typed 180° 旋转。两者均记录于 `run.toml`，
重载环境时仍检查是否属于同一个物理张量。

Julia 接口：

```julia
include("src/FermionicPEPS.jl")
using .FermionicPEPS
run = solve_fpeps_stilde(; chi=(2,2),
    measurement_kwargs=(; depths=(8,16,32,64), tolerance=1e-7))
run.result.stilde
run.result.diagnostics

# 显式保留所选择的环境和 cap，再做不同长度的测量。
p = prepare_fpeps_stilde(; chi=(2,2))
r = measure_lmps_stilde(p.seams,p.caps; depths=(16,32,64,128))
```

`measure_lmps_stilde(...;audit_depth=1)` 还可逐项比较完整四副本与 cycle 分解；
完整四副本端点有八条 χ 腿，仅用于小 χ 验证。
若主空间简并，可在 Julia 接口显式传入经过物理选态的 `caps`；程序不猜测简并空间中的线性组合。

## 验收范围

- 原始定义为 `S_tilde = -log(Z4*Z1^2/(Z2A*Z2B*Z2C))`，自然对数。
  四副本 occupation sewing 是 16 个 graded parity-string 网络的带符号和。
  所有复数相位、传播范数和 parity term 都保留，没有用 `log(abs(Z4))` 代替符号检查。
- `physical_sewing_probe.jl` 从三个区域的 ket maps 独立构造双层 rails，与完整物理态的 occupation
  收缩逐项比较；覆盖随机三模式、非 Gaussian 六模式、严格及变形的 3×2 fPEPS。
  这验证物理 parity pull-through 和中心 cap，而不只是在同一网络中比较两条求和路径。
- `sixr_probe.jl` 比较五个 sector 的完整 seam 与两周期分解，包含全部 16 个四副本 term。
  物理对照还检查各区域 parity block 内一般酉 gauge 变换。
- `measurement_probe.jl` 检查 product 环境、长度收敛、增量传播与重新传播的一致性、未收敛拒绝。
  接近零的 term 不删除；端点残差按它对完整复数 sector 的贡献计权。
- 长度极限同时检查最后三个 S_tilde、五个原始 log|Z| 的线性尾部和加权端点残差。
  单副本环境诊断、metric 可逆性、cap 谱、相位和抵消门槛分别检查。
  小 seam 在固定 even intertwiner 空间中求完整谱；矩阵的每个 action 都经过 typed contraction，
  不把物理 graded tensor 当作无符号数组收缩。大空间继续使用 Krylov。

```bash
sbatch jobs/run_cpu.sh test/physical_sewing_probe.jl
sbatch jobs/run_cpu.sh test/sixr_probe.jl
sbatch jobs/run_cpu.sh test/measurement_probe.jl
sbatch jobs/run_cpu.sh ../lmps/tests/runtests.jl
sbatch jobs/run_cpu.sh ../lmps/vumps/test/runtests.jl
python test/check_legacy_sources.py
```

`measurement_probe.jl` 的集成部分使用仓库中的 `data/model_seams/chi4.jls`，
其余新符号测试自行构造小系统。

有限系统的 Gaussian/Fock 对照保证符号规则，不保证某个截断 VUMPS 环境的精度。
应随 χ 检查熵、norm 本征值、方向关联和分支，并独立延长 seam。
`refs/multientropy_PEPS.pdf` 中的 `6 log χ` 证明还有实数及空间对称性假设，不能直接当作本模型的验收条件。

## 运行

使用 `../lmps/vumps/Project.toml` / `Manifest.toml` 中已有的固定版本：
Julia 1.12、TensorKit 0.17.1、PEPSKit 0.8.1、MPSKit 0.13.13。
不需要修改或更新旧环境。Slurm 脚本给本目录配置独立的第一层 Julia depot。

保存四方向单副本 environment 与局域 LMPS：

```bash
sbatch jobs/run_cpu.sh bin/run_boundary.jl 2 2 data/my_boundary
sbatch jobs/run_cpu.sh test/checkpoint_probe.jl data/my_boundary
```

第二条命令应在第一条成功结束后运行。输出目录必须不存在；每个方向保存带 parity 空间的 `.jls`
和可读的 `.toml`，`run.toml` 记录模型约定、软件版本、源代码及 Manifest 哈希。
这些 Julia Serialization checkpoint 供相同固定环境使用，不是跨版本交换格式。
`11212371` 已完成 χ=(2,2) 的四方向保存；`11212500` 已通过重载、parity 空间、物理张量、
密度及局域 LMPS 的再计算对照。

```bash
cd /ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps
sbatch jobs/run_cpu.sh test/local_probe.jl
sbatch jobs/run_cpu.sh test/parent_probe.jl
sbatch jobs/run_cpu.sh test/replica_probe.jl
sbatch jobs/run_cpu.sh test/boundary_probe.jl
sbatch jobs/run_cpu.sh test/lmps_probe.jl
sbatch jobs/run_cpu.sh test/torus_probe.jl
```

在相同 Julia environment 下的接口：

```julia
include("src/FermionicPEPS.jl")
using .FermionicPEPS

A = ksvc_tensor()                    # physical ← (north,east,south,west)
peps = ksvc_ipeps()                   # 已知 A，不做物理基态优化
b = solve_boundary(peps; chi=(4,4))  # 保留 even/odd 各 4 个边界态，总 χ=8
boundary_number(b)
c = solve_ctm(peps; chi=(4,4))       # 独立的固定态 CTMRG environment
ctm_number(c)
objects = lmps_objects(b)            # 包含显式 fermionic cap 的局域 LMPS 对象

psi = finite_fock_state(3,2)         # 全局 CAR 投影，开边界虚腿真空
sectors = occupation_sectors(psi, [:A,:B,:C,:A,:B,:C])
reference = gaussian_sectors(majorana_covariance(psi), [:A,:B,:C,:A,:B,:C])
```

`chi=(even,odd)` 是分 sector 的简并维数。已知有限 D 的严格物理态不意味着有限 χ 的严格 environment；
`solve_boundary` 求的是双层 transfer operator 的固定点，并检查 Galerkin、左右 environment 和 canonical center 的残差。

## 费米约定

- 全局模式顺序为逐格点 `[c,α,β,γ,δ]`，格点 x 最快、再 y。
- 定义 Q 的单项式顺序为 `c† δ β γ α`；转成 `physical ← α⊗β⊗γ⊗δ` 时包含 `(-1)^(r*u)`。
- Bell bond 顺序是 β→α（水平方向）、δ→γ（向下）。吸收 bond 后，PEPS domain 为 `(V,V',V',V)`。
- 内部不将 graded tensor 转成无标签数组。`finite_tensor_state` 仅在所有虚腿收缩完后导出物理 Fock 系数作为对照。
- 物理 Fock 重排用 graded permutation；定义 Rényi 观测量的 occupation replica sewing 用普通置换。
  `occupation_permute` 在 graded permutation 后补偿其 Koszul 符号，两者不能互换。
- 四副本采用 `σA=id, σB=(12)(34), σC=(13)(24)`。
  保留复数 Z4，并在取对数前检查相位与正性，不用 `log(abs(Z4))` 掩盖符号错误。

## Q 与 Hamiltonian 的字典

直接按上述 CAR 顺序投影的态记为 `:projector`。它的 parent Hamiltonian 的 diagonal hopping
相对于 PDF 中写下的 Hamiltonian 反号。这里不把两者默认为同一约定，也没有靠改动测试目标隐藏差异：

| 约定 | 物理 Fock 变换 | hopping scale | pairing scale |
|---|---|---:|---:|
| `:projector` | 原始 Q 投影 | −1 | +1 |
| `:paper` | `particle_hole_fock(psi)` | +1 | +1 |
| `:gaussian` | `particle_hole_fock(psi;phase=im)` | +1/2 | −1/2 |

scale 相对于 `ksvc_hamiltonian_action` 中显式写下的 PDF Hamiltonian：
`2i c†r c†r+y − 2i c†r c†r+x − c†r cr+x+y − c†r cr+x−y + h.c.`。
最后一行与 `Gaussian/code/cirac2_model.py` 的 Hamiltonian 约定一致。
`parent_gaussian_reference` 独立构造 BdG 矩阵、基态能量和 covariance；存在零模时报告退化，
不任意填充零模生成唯一 covariance。

当前 `ksvc_ipeps()` 明确返回原始 `:projector` 态；这里的字典不表示已经实现带奇局域物理映射的另一种一格点 iPEPS 容器。

## 验证证据与限制

- `11212324`：所有有限开边界 Q 投影与 TensorKit 网络系数一致；包括严格模型的非零 3×2 patch。
  另用不对称 bond 权重 `(0.7,0.8)` 检查 2D orientation，最大系数误差 `2.9e-16`。
  occupation 五个 sector 与独立 Gaussian Pfaffian 参考一致，最大 `stilde` 差约 `6e-15`。
  变形仅用于测试，不替换默认严格模型。
- 默认严格模型的 2×2 虚腿真空闭合**恰好为零**，两条独立实现均如此。该态不能归一化；测试明确跳过其归一化观测量。
- `11212319`：3/4 模式随机 even 态的 typed 四副本与直接 occupation contraction 一致，包含非 Gaussian 态。
- `11212338`：3×3 周期态在三种 convention 下均通过直接 CAR 本征态检查、BdG 基态能量和完整 covariance 对照。
  基态能量分别 −18、−18、−9；这三个小系统没有零模。
- `11212364`：3×3 torus 的 typed 张量收缩与全局 CAR 系数完全一致；PBC 非零。
  该尺寸的另外三个 ±1 虚键 twist 扇区均给零态，两种实现一致。这不是三个可归一化的参考基态。
- `11212309`：四方向单副本 VUMPS χ=(2,2) 均收敛，密度为 1/2；同 χ CTMRG 也收敛。
  小 χ 两种 environment 的 norm 特征值仍有约 1.4% 差异，不能把密度吻合当作环境精度的完整证明。
- `11212356`：四方向局域 LMPS 的 B、G、center placement 与 PEPSKit 一致，误差 <4e-16；
  `B≈κ AL G` 残差 <1e-8。metric 使用有向 cap，不能改成 `L*R`。
- 单副本 LMPS 检查与新增的物理多副本闭合检查是不同的验收层次，不能互相替代。
- `11212370`：occupation sewing 相对于 graded sewing 的符号被精确编译成 16 个 parity-string sector
  （权重 ±1/4），其 typed 求和在 3/4 模式随机 even 态上与直接 occupation Z4 一致；三个 Z2 也一致。
  后续 `sixr_probe.jl` 和 `physical_sewing_probe.jl` 补上有限 rail 网络的分解与物理 sewing 检查。
- `11212357`、`11212358`：原有 `lmps/tests/runtests.jl` 和 `lmps/vumps/test/runtests.jl` 全部通过。
  后者覆盖 RK 适配、TFIM 单位胞/方向构造和 LTR 映射，不声称覆盖完整 TFIM 优化生产作业。

旧代码源文件和环境文件的初始 SHA-256 保存在 `legacy_source_sha256.json`。
研究符号方案见 `../../notes/fpeps_signs_zh/main.pdf`；该 PDF 描述设计阶段，不代替本目录的运行证据。
