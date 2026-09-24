# fPEPS S̃ 收缩不稳定性：实现审查与根因诊断

**2026-09-17 范围纠正：本文历史熵和“direct”闭合诊断对应 grown-tail / grown-turn，
不是当前要求的 bare/direct。不能由这些诊断断言 bare 的相位或符号失败。
新的 bare χ=4 S̃=0.439613897919，符号、相位、长度检查通过；物理精度未认证。
最新方法和 benchmark 见[实现 note](fpeps_signs_zh/implementation.md)。**

日期：2026-09-13。对象：`PEPS/code2d/fpeps`，gapless w=1 KSVC D=2 张量，无限 iPEPS 边界 MPS + LMPS direct 路线。
本文只做审查与诊断，没有提交任何生产作业，没有修改任何源码或数据。
配套：[交接文档](fpeps_calculation_handoff_zh.md)、[稳定化实验记录](../code2d/fpeps/data/stable_boundary_20260912/README.md)。

## 0. 结论

1. **实现的代数部分没有发现错误。** replica 符号、graded seam、six-R 闭合、有限区域 sewing 都有独立 oracle 到 1e-15 量级的验证（第 1 节列表）。当前的不稳定不是某个符号或收缩 bug，而是三个结构性问题叠加，任何一个不解决，"再抛光 native 残差"都没有用。
2. **根因一（不动点不唯一）**：w=1 时双层行转移算符有一个精确的 onsite 守恒 Z2（团队记录里的 neutral fermion，准粒子比值恰为 −1）。无限 χ 的不动点是 2^L 重简并的两类扇区 ±λ，1-site VUMPS 的目标方程本身不适定。χ10 Schur 分支、cap 依赖、seam 简并等症状都能归到它（第 2 节对照表）；但第 6.2 节的实验表明，扇区已纯到 1e-7 的候选态（χ12/16/24/32）的残余相位偏差不是扇区问题。χ=4/6/8 能收敛是因为碰巧落在纯扇区（neutral charge +1 零方差，job 11229766）。
3. **根因二（逆度规放大）**：`src/rails.jl` 的 `vR = R*inv(G)`，而 `lmps_objects` 自己检查的恒等式就是 G = λ_C·C（`package_metric_residual` 比的是 `MPSKit.C_hamiltonian`），所以 inv(G) = C⁻¹/λ。临界边界的最小 Schmidt 值随 χ 迅速变小，环境弱模上 1e-7..1e-8 的绝对误差被放大成 O(1e-2..1) 的 rail 误差（`rail_vs_boundary_residual` 0.07..0.99），进而给出 1e-5..1e-4 的 replica 相位误差。（第 6 节实验：去掉 inv(G) 后候选态的 S̃ 移动 1.7e-3..2.4e-2，与该残差同量级；但 polish64 的 3.1e-5 相位偏差不是它造成的，见 6.3。）
4. **根因三（验收指标反了）**：native Galerkin ≤1e-12 衡量的是"1-site 均匀 MPS 是 VUMPS 方程的精确驻点"，不是物理精度。对非厄米临界转移算符这个驻点在给定 χ 下可以不存在（χ12 的 2.76e-6 是 ½‖R‖² 的严格局部极小，Hessian 最小本征值 2e-9 > 0）。被拒绝的 χ12/16/24/32 的物理观测量比被接受的 χ4/6/8 好 10–100 倍。
5. **gap-closing 绕路已被封死**：w 族在 w_c=√2−1 处以 z=4 闭合，0.414<w<1 整段无能隙（8 个 QBT 节点），σ_y 质量态不是任何有限 D 的严格 GfPEPS（`MASS_REPRESENTABILITY_20260911.md`）。必须在 w=1 处解决。
6. **可行方案**（第 4 节）：去掉 inv(G)、把 canonical center 放在交点（有限区域代码已经是这个形式）；在所有对象上固定同一个 neutral 扇区；边界改用 exact Gaussian + Fishman–White 压缩作主环境，VUMPS 只做可选抛光；验收改为物理门槛；目标量改为 S̃ 对 ln ξ 的斜率。

## 1. 实现审查

### 1.1 已被独立验证的部分（不需要再怀疑）

| 组件 | 验证方式 | 精度 | 证据 |
|---|---|---|---|
| `ksvc_tensor` / `ksvc_projector` | 全局 CAR oracle、BdG 对角化、3×3 PBC 能量 | 精确 | fock.jl、gaussian.jl、signs note |
| `occupation_permute`、16 项 parity-string 展开、精确 4 项 character | 随机复数 even 态、非 Gaussian 六模式态 | 1e-14 | 11217997、11218021 |
| `_spatial_boundary_bra`、右端点 twist | 随机张量完整线性映射恒等式，四种旋转 | 4e-16，48/48 | 11216615、11217053、11217162 |
| `lmps_objects` 局域恒等式 B≈κ·AL·G、G=H_C(C) | 对照 PEPSKit 自身 C/AC 作用 | 2e-16 | implementation.md |
| 四副本交点分块求和 | 随机换序/分块 26 项，χ32 五扇区重闭合 | 5.9e-15 | implementation.md |
| `measure_finite_fpeps_stilde`（有限窗口、真实 center 在交点） | L=2,4 精确 Gaussian，50 项物理 occupation 对照 | <5e-15 | 11213353..74 |
| 有限窗口 gapped w=1/8 χ=16/24 | 同窗口 Gaussian | 3.4e-9 | CHI_SCAN_20260910.md |
| 无限 direct gapped w=1/4 χ32 | Gaussian APBC torus/OBC 极限 | 2.1e-7 | points.csv |
| period-2 rails、periodic replica、重复 χ4 端到端 | 复现一站点 S̃ | 1.1e-13 | 11223353 |

结论：从"给定三个边界 MPS + caps"到"五个扇区的复数值"这一段是对的。问题全在"边界 MPS 与 rails 怎么来"和"怎么判定可信"。

### 1.2 设计层面需要意识到的三点

**(a) 同一 MPS 的 sandwich 环境被当作物理竖直 rail。** `lmps_objects` 取的是 `boundary.environments.GLs/GRs`，即 ⟨n|T|n⟩（上、下都是北边界 MPS，下侧共轭）的左右固定点。这对 TFIM 那种上下镜像对称的张量是物理水平通道；KSVC 张量没有上下镜像（色散 Δ(k)=4(sin kx−sin ky) 在 ky→−ky 下不变号也不保持，只有 180° 旋转+规范、对角/反对角反射），所以 GL 不是 ⟨s|T|n⟩ 的物理左固定点。这就是 direct 与 mixed 分支的差别（gapped：direct S̃ 差 2e-7 但面积项斜率差 3.9e-5；mixed 面积项对但常数差 2e-4）。它不阻碍取 S̃–ln ξ 斜率，但意味着 direct 路线即使 χ→∞ 也不是精确的，不能拿"χ 收敛"当"物理收敛"。

**(b) `vR = R*inv(G)`。** 见第 0 节第 3 条与第 2.2 节。这是可以立即改掉的。

**(c) `solve_seam_caps` 用一副本 seam 的主本征向量做远端 cap，并要求它孤立。** 在根因一存在时它必然不孤立（seam AB 代数重数 2 / 3，gap 1e-15），于是 cap 成了"在简并子空间里的任意选择"，S̃ 随 cap 变化（0.5546 / 0.549 / 0.4347 在同一物理 ξ 下）。这不是 cap 代码的错，是扇区没固定。

## 2. 根因与症状对照

### 2.1 根因一：w=1 的精确 onsite Z2（neutral fermion）

已有证据（团队自己的数据，我只是把它们连起来）：

- `gaussian_boundary_ring/dominance_audit.json`：w=1 时 L=3/4/8/16 的精确行转移主模数重数 2^L，比值全部是 −1.000000000000；w=0.25 无。
- `local_gaussian_circuit/neutral_operator.json`：存在唯一的 onsite Hermitian 双线性 q（q²=I）与完整 L=4 Fock 行转移在每个站点对易，误差 2.8e-16；w=0.25 时不存在。
- `factorization/result.json`：T = P_neutral ⊗ T_active 到 4e-16。
- 精确 Gaussian 参考的 q 期望值为 1/√2，即物理（真空起始）不动点本身是两类扇区的叠加。

物理来源：`GAP_CLOSING_20260911.md` 给出 w=1 时 q,p 有公共因子 cos(kx/2)cos(ky/2)，即 kx=π 或 ky=π 的整条线是 fPEPS 映射的零权重（dark）方向；它们是 w<1 时 8 个 QBT 节点里那 6 个在 w→1 时跑到 k=π 的残余。ky=π、每列一个、沿 y 以 (−1)^y 交替——这正是行转移里"每站点一个、本征值比 −1"的 neutral 模式。所以它**不是**单层张量的规范对称（我的独立检查：Q 单层没有任何 Majorana 乘积对称，见第 3 节），而是最大纠缠虚键（w=1 时 Bell 键在 γ¹⊗γ² 下不变，w=0.25 不变性只有 0.47）与 Gaussian 映射零模共同产生的**双层**守恒量。它携带零物理权重，但 replica 交点项对它的扇区选择敏感——这解释了 cap 依赖，也说明不同扇区不是"同一物理答案的不同表示"，而是"不同的无穷远边界条件"。

症状对照：

| 观察到的现象 | 机制 |
|---|---|
| χ10 环境主本征空间二维、劈裂 1e-15 | 两类扇区 ±λ 模数相同 |
| self-transfer 奇扇区本征值 −1 | (−1)^y 的 neutral 串序 |
| self-even ξ = 2295 (χ12)、3.24e5 (χ32) | 非衰减的扇区叠加分量 |
| VUMPS 残差 1e-5 平台后跳到 1e-2、λ 出现虚部 | 局部本征求解在近简并本征向量间切换 |
| seam AB/BC 主本征值代数重数 2 (χ16)、3 (χ36)，AC 唯一 | replica seam 交换 copy 的 ket，混合各 copy 的 neutral 扇区 |
| identity cap / odd-weight cap 同值，positive-block cap 变值且相位失败 | 简并子空间里的 cap 选择 |
| 旧 χ4 与 active4×neutral4 的 χ16 在同一物理 ξ 下 S̃ 差 0.12 | 扇区不同 |
| χ4/6/8 能收敛 | 落在纯扇区（neutral charge +1，零方差） |
| gapped w=0.25 完全没有这些问题 | w≠1 无 dark 线 |

### 2.2 根因二：inv(G) 的放大

从已保存的 `geometry_checks.toml`/`entropy.toml` 汇总（gapless w=1）：

| 分支 | native/BG 残差 | G 相对最小奇异值 σ_min | rail_vs_boundary_residual |
|---|---:|---:|---:|
| χ12 plateau | 2.76e-6 | 2.5e-4 | 7.6e-4 |
| native16 (fit) | 4.9e-4 | 3.8e-6 | 0.98 |
| native16_real | 1.3e-4 | 8.2e-5 | 0.088 |
| native32_real (fit) | 1.6e-4 | 1.3e-8 | 0.995 |
| native32_tangent_fine | 7.3e-7 | 4.3e-7 | 0.072 |
| native32_inner15 | 6.6e-9 | 7.0e-7 | 4.9e-4 |
| native32_fourth_best16 | 2.7e-9 | 6.7e-7 | 2.5e-4 |
| native32_fourth_polish64 | 8.0e-14 | 6.7e-7 | 7.3e-9 |

规律：rail 误差 ≈ (环境弱模绝对误差)/σ_min。第 6.1 节的实验表明它进入 S̃ 的量级是 0.3–2 倍 rail 误差；但 replica **相位**偏差与它无关（polish64 的 rail 误差 7e-9，相位仍 3.1e-5，去掉 inv(G) 后逐位不变）。临界边界 σ_min 随 χ 按幂律下降，所以这个放大在 χ≥16 后不可能靠精度堆回来。

代数上 inv(G) 完全不必要：沿每条 rail，vR = C·GR·C⁻¹ 的 C⁻¹ 与下一段的 C 望远镜抵消，整条 rail 等价于用 GR 做 rail、把 C 放在交点一端、把 C⁻¹ 吸收进远端 cap（cap 本来就是作为主本征向量求出来的，不需要显式构造 C⁻¹）。这正是 `finite_regions.jl` 里 `mps_junction_geometry` + `_mps_junction_close(centers=…)` 的形式，而且那条路已经过 1e-15 的精确验证。

### 2.3 根因三：验收指标

| 状态 | native 残差 | NN 关联对 Gaussian 误差 | 物理 ξ | 当前标签 |
|---|---:|---:|---:|---|
| χ4 | 1e-12 | 2.7e-3 / 1.9e-3 | 1.49 | 接受 |
| χ6 | 1e-12 | 3.2e-3 | 3.26 | 接受 |
| χ8 | 1e-12 | 2.6e-4..3.4e-3 | 7.4 | 接受 |
| χ12 plateau | 2.76e-6 | 3.3e-5 | 35.5 | 拒绝 |
| χ16 Newton | 5.7e-6 | 4.7e-5 | 30.4 | 拒绝 |
| χ24 Newton | 4.6e-6 | 2.0e-5 | 29.2 | 拒绝 |
| χ32 tangent | 7.3e-7 | 1.7e-5 | 27.8 | 拒绝 |

被接受的三点 ξ<1.5 个格距，本来就不在任何渐近区；它们给出的斜率 0.1547 与 Gaussian 0.155 的吻合不能承载结论（交接文档已如实标注）。native 残差应记录，不应作为门槛。

## 3. 本次独立检查（job 11234909、11234911，登录节点未运行数值）

脚本：`kangw/tmp/claude-162088/.../scratchpad/ksvc_virtual_symmetry_check.py`、`ksvc_double_layer_symmetry_check.py`，纯 numpy 32×32，约定逐条照抄 `state.jl`/`fock.jl`。

- Bell 键 |00⟩+w|11⟩ 在 Majorana 对下的射线重叠：w=1 时 γ¹⊗γ² 与 γ²⊗γ¹ 给 1.0000（相位 −i），γ¹⊗γ¹、γ²⊗γ² 给 0；w=0.25 时最大只有 0.4706。即"虚键在 Majorana 对下不变"是 w=1 独有的。
- 单层 site map Q（2×16）：在"每个虚模式至多一个 Majorana"的 80 种乘积里，不存在 M·U = V·M（V 任意 2×2）的解。所以 neutral Z2 不是 site 张量的虚拟规范对称，KSVC w=1 不是 Z2^f-injective 那种情形。
- 双层、允许 ket/bra 取不同 Majorana 类型的检查：见 job 11234911 输出（本文末尾附）。

## 4. 建议的修法（按优先级，前两项是必要条件）

**A. 无逆闭合（改 `fermionic_rails` 一处，低风险，先做）。** 竖直 rails 改用 (GL, GR)，在交点插入各区域各 copy 的 canonical center（`finite_graded_sixr(...; centers=…)` 已有接口，`corner_centered_lmps_probe.jl` 有四方向版本），远端 cap 在同一 frame 里求主本征向量。验收：χ8 gapless 复现 0.5616249949580379 到 1e-12；gapped χ32 复现 0.0164299369 且 χ40 不再比 χ32 差。这一项与扇区问题无关，单独就能去掉表 2.2 里那列放大。

**B. 在所有对象上固定同一个 neutral 扇区。** north、south、rotated 三个边界、三个 seam cap，以及五个 replica 扇区必须处在同一 q 扇区。可行做法有两种：(i) 团队已有的 active 约化 + 带符号重建（11229387 已证明重建后 native 残差回到 1e-13）；(ii) 直接在双层 MPO 的竖直腿对上乘投影 (1+q)/2 求边界，再把边界还原成两腿表示。判据不是 S̃，而是：neutral charge 零方差审计通过、seam AB/BC 主本征值从重数 2/3 变为唯一（这是可以直接量的）。注意 replica seam 会交换 copy 的 ket，per-copy 投影不能在 seam 内部做，只能在边界层面做完再检查 seam 谱。

**C. 边界求解器：主环境不用 1-site 非厄米 VUMPS。** 这个态是 Gaussian，无限平面 active 边界已经有 L256/M256、采样协方差误差 3.5e-8 的精确参考；Fishman–White 压缩到目标 χ 的 MPS 就是该 χ 下的最优正定、宇称一致的环境，不需要满足 VUMPS 驻点条件。之后若要抛光，用 power method（行转移作用 + 保持 q 扇区的 SVD 截断）而不是 Krylov 本征求解；每步记录残差但不作门槛。若坚持 VUMPS，至少改 2 或 4 站点元胞（±π/2 节点使边界关联按周期 4 振荡，1-site 元胞只能用复本征值表示它）。

**D. 验收改物理门槛。** 密度 = 1/2、横向 normal 关联的绝对误差（精确值 0）、anomalous 关联与 Gaussian 的误差三者随 χ 单调下降；各扇区 replica 相位 ≤1e-8；三档长度漂移 ≤1e-7；native 残差只记录。

**E. 目标量。** S̃(χ) 在 w=1 处按 ln ξ(χ) 发散，可比较的是斜率（Gaussian 单交点 c(π,π/2,π/2)=0.155）。三种 ξ 都记录，主对照用 paired odd 或 physical fermion（已有文献核对）。要跨一个 decade 的 ξ，需要 χ≈64；只有 A+B+C 一起做才够得着。

**不建议再投入的方向**：更高精度的 Newton/有限差分、BigFloat、damping、更严的 native 门槛、更多 cap 试探——它们都在根因一、二之外打转，过去两天的记录已经说明这一点。

## 5. 附：job 11234911 输出

双层检查：对单站点双层张量 E = M†M（物理腿已缩并，16×16），遍历 ket 层与 bra 层各自取"每个虚模式至多一个 Majorana"的乘积（支撑相同、类型可不同，共 80×80 里 6560 对），没有任何一对满足 U_ket† E U_bra = E（相对误差 <1e-10 的解数为 0）。E 的奇异值为 [8, 8, 0, …]，即秩 2，与物理维数一致。

结论：neutral Z2 既不是单层 site 张量的虚拟对称，也不是单站点双层张量的对称；它只在把 w=1 的水平 Bell 键缩并成整行之后才出现，与 `neutral_operator.json` 在完整 L=4 行上找到 q、而单站点找不到的情况一致，也与 Gaussian 里 ky=π 整条 dark 线（需要整行动量）的图像一致。因此"投影掉 neutral 模式"只能在行转移 / 边界层面做（active 约化），不能通过修改局域 PEPS 张量实现。

## 6. 实验结果（2026-09-13，"你试试吧"之后）

所有输出在 `code2d/fpeps/data/inverse_free_20260913/`（`README.md`、`submissions.sh`、`jobs.txt`、`summary.csv`），
脚本 `benchmark/sector_support_tools.jl`、`sector_support_audit.jl`、`inverse_free_direct.jl`、`summarize_inverse_free.py`。
没有修改任何生产代码或已接受数据；所有 S̃ 仍是诊断值。

### 6.1 无逆闭合（方案 A）已验证

把三个区域的水平 rail 从 H=B·inv(G) 换成边界张量 AL 本身（谱半径归一），turn、cap、交点、character、长度/相位门槛全部不变：

| 情况 | χ | S̃ 无逆 | 旧 direct | 移动 | 相位误差 | 旧 H-vs-AL | ξ_self |
|---|---:|---:|---:|---:|---:|---:|---:|
| χ8（接受点） | 8 | 0.5616249949580379 | 0.5616249949580379 | +2e-13 | 2e-12 | 1.7e-10 | 1.49 |
| χ8 奇规范对照 | 8 | 0.5616249949582652 | 同上 | +2e-13 | 1e-12 | — | 1.49 |
| gapped w=1/4 χ32 | 32 | 0.0164299368975 | 0.0164299368759 | +2e-11 | 1e-9 | 4e-6 | 0.64 |
| χ12 real chart | 12 | 0.8315635 | 0.8332803 | −1.7e-3 | 3e-10 | 7.6e-4 | 3.88 |
| χ16 fresh Newton | 16 | 0.7849375 | 0.7927663 | −7.8e-3 | 3.0e-7 | 5.8e-3 | 5.03 |
| χ24 fresh Newton | 24 | 0.7911498 | 0.8037667 | −1.3e-2 | 2.3e-5 | 4.1e-2 | 5.75 |
| χ32 isolated | 32 | 0.73596（depth 32） | 0.7595745 | −2.4e-2 | 1.0e-4 | 6.5e-2 | 4.86 |
| χ32 polish64 | 32 | 0.7962634690 | 0.7962634686 | +4e-10 | 3.1e-5 | 7.3e-9 | 6.30 |
| χ32 polish64 奇规范对照 | 32 | 0.7962634690 | 同 polish64 | 0 | 3.1e-5 | — | 6.30 |
| gapped w=1/4 χ40 | 40 | 0.0164328001（depth 16，作业 11235043 未完） | 0.0164328001 | ~0 | 2e-10 | 2.3e-5 | 0.64 |

结论：无逆构造与原构造在精确不动点处逐位一致（χ8、gapped χ32/χ40、polish64）。gapped χ40 的 0.0164328（比 χ32 的 0.0164299 离 Gaussian 0.0164301 更远）不是 inv(G) 造成的，而是该 χ40 VUMPS 不动点本身的性质，与临界情形"不同分支给出不同 S̃"是同一现象。在非精确不动点处 S̃ 的移动与旧 H-vs-AL 残差同量级（0.3–2 倍）。这证实 inv(G) 确实把环境弱模误差带进了 S̃，量级 1e-3 到 2e-2。建议把 `fermionic_rails`/`compressed_direct_rails` 里的 H 换成 AL。

### 6.2 扇区审计（与图表无关的 SVD 方法）

`sector_support_audit_v2.toml`：边界张量在 (ket,bra) 腿上的第二奇异值与第一的比值，以及相对 χ8 扇区的泄漏。

| 边界 | sv2/sv1 | 泄漏 |
|---|---:|---:|
| χ4 / χ6 / χ8（接受） | ≤3e-14 | ≤3e-14 |
| χ10 Schur（失败分支） | 0.21 | 0.17 |
| χ12 real chart | 6.5e-8 | 4.7e-8 |
| χ16 fresh Newton | 2.6e-7 | 2.3e-7 |
| χ24 fresh Newton | 1.8e-4 | 1.7e-4 |
| χ32 isolated | 4.8e-3 | 4.4e-3 |
| χ32 polish64 | 3.7e-16 | 1.2e-15 |
| gapped w=1/4 χ32 | 0.85 | （无此扇区）|

w=1 的所有可用边界都精确落在同一个 2 维扇区，gapped 边界满秩，χ10 失败分支泄漏 17%。这是 neutral Z2 的直接证据，并给出一个新的边界验收指标：泄漏 >1e-3 的边界不能用。

但是：把 χ16、χ24、χ32-isolated 投影回 χ8 扇区后，S̃ 与相位误差几乎逐位不变（χ16：完全相同；χ24：S̃ 差 1e-9；isolated32：差 2e-6）。所以候选态里 1e-7..4e-3 的泄漏对 replica 网络不可见，残余相位失败**不是**扇区问题。

### 6.3 残余相位偏差的定位

把四个扇区的相位偏差解为每条 seam 的 2-cycle 端点相位偏移（Z2A=δAB+δAC，Z2B=δAB+δBC，Z2C=δAC+δBC，Z4=2Σδ，最后一式在所有情况下成立到 1e-7）：

| 情况 | δAB | δAC | δBC |
|---|---:|---:|---:|
| χ16 | +1.6e-7 | +1e-8 | −2e-8 |
| χ24 | −9.0e-6 | −8e-8 | −2.2e-6 |
| χ32 polish64 | +1.4e-5 | +2e-7 | +1.4e-6 |
| χ32 isolated | +7.2e-5 | +2.6e-5 | −2.5e-5 |

偏差集中在 AB seam（北边界的 tail-turn 配东边界的 rail），与深度无关（端点常数），随 ξ_self 单调增大。已排除：inv(G)（polish64 的 H-vs-AL 只有 7e-9）、扇区泄漏（投影不变）、奇虚规范（χ8 与 polish64 的奇规范对照逐位相同）、边界算符正定性（polish64 在有限环上 Hermitian/PSD 到 5e-14，比 χ8 还好）、native 残差（polish64 是 8e-14 的精确不动点）。剩下的解释：1-site VUMPS 的不动点在两副本交换通道里不是正定的——非厄米问题没有变分原理保证 σ=M†M 结构，这个偏差随 χ 增长，与 native 残差无关。

### 6.4 更重要的发现：S̃ 的分支依赖

χ≥12 的候选态局域观测量精度相近（NN 误差 1e-5 量级，密度 1/2 到 1e-12），但无逆 S̃ 分别是 0.8316（χ12）、0.7849（χ16）、0.7911（χ24）、0.7963（χ32 polish64）、0.736（χ32 isolated），且它们的物理 ξ 都停在 30 左右不随 χ 增长。不同求解分支在同一 χ 下给出相差 0.01–0.06 的 S̃。这比 1e-5 的相位问题大三个量级，是提取 ln ξ 斜率的真正障碍：VUMPS 不动点不构成一个规范的 χ 族。

### 6.5 结论更新

1. 立即可做：生产 rails 改为无逆闭合（一行改动，已验证）；用 `sector_support_tools.jl` 的泄漏作为新边界的门槛。
2. 相位门槛 1e-8 在 χ≥16 不可能满足，且偏差来源不是可修的 bug；应把相位偏差当作该点的不确定度记录（1e-5 对斜率无害），而不是拒绝点的理由。
3. 真正的瓶颈是边界族不规范。需要一个不依赖非厄米本征求解分支的构造：从精确 Gaussian active 边界直接压缩到目标 χ（不做 VUMPS 抛光），或保持正定结构的 power method。在这之前，任何 S̃–ln ξ 斜率都不可信。

## 7. 尝试无分支的边界族：截断 power 迭代（负结果）

`benchmark/power_boundary.jl`，输出 `data/power_boundary_20260913/`。从 χ8 接受态出发，每步"作用一行（direct 构造里同一个 grown 张量 D）→ 规范化 → 对 canonical center 做 SVD 截回 χ"，可选每步把 (ket,bra) 腿投影回 χ8 扇区。

| 设置 | 结果 |
|---|---|
| χ8，不投影，1500 步 | 不收敛到不动点：每步保真度亏损恒为 3.9e-7（极限环），native 残差 1.06e-4，λ=3.59215309（VUMPS 3.59214230）；北、南落到同一轨道；扇区泄漏 1e-15 |
| χ12，不投影，1500 步 | 漂出扇区：泄漏升到 0.36，λ 漂到 3.5958，残差 1e-3 |
| χ8，每步投影 | 11 步收敛到不动点（保真度 1−5e-14），但 native 残差 1.1e-5，λ 与 VUMPS 差 3e-6：是另一个态 |
| χ12，每步投影，600 步 | 不收敛：保真度 0.99996/步，λ 漂到 3.619，残差 5e-3 |
| χ8 power 族的无逆测量 | 在 "direct full-tail eigenvalue mismatch" 门槛失败：非不动点的环境通道主模数简并，tail 无法定义 |

结论：对这个临界转移算符，1-site 均匀 MPS 空间里"作用一行 + 截断"的映射在 χ≥12 没有吸引不动点（要么漂出 neutral 扇区，要么在扇区内游走），χ8 也只有极限环或与 VUMPS 不同的驻点。它不能提供规范的 χ 族。未投影的 χ16/24/32 作业（11237444–11237455）会漂扇区，其依赖测量会在同一门槛失败，结果不可用。

综合第 6、7 节：inv(G) 与扇区判据两项已解决，但"边界族不规范"这一核心障碍在今天试过的两条无分支路线上都没有解决。剩下的可行方向是从精确 Gaussian active 边界直接构造固定 χ 的均匀 MPS（不经 VUMPS/power 抛光），这需要单独设计，不是一次试验能完成的。

### 6.6 补充：用物理通道 ξ 作横轴时各分支是一致的

对 ξ_self（边界自转移）作图时 χ16/χ24 的 ξ 比 χ12 大而 S̃ 反而小，不合理。改用 north–south 夹列物理通道的 ξ（各点 observables.toml 的 physical_even_channel_xi，χ8 取 critical_physical_audit 的 even_transfer_channel_xi）后，全部无逆点按 ξ 单调：

| 点 | ξ_phys | S̃ 无逆 |
|---|---:|---:|
| χ8 | 7.44 | 0.5616 |
| χ32 isolated | 20.86 | 0.7519 |
| χ24 | 29.17 | 0.7911 |
| χ16 | 30.40 | 0.7849 |
| χ32 polish64 | 30.91 | 0.7963 |
| χ12 plateau | 35.53 | 0.8316 |

六点对 ln ξ_phys 的最小二乘斜率 0.166，最大残差 0.016；从 χ8 出发的两点斜率 0.159–0.185；Gaussian 单交点值 0.155。说明"分支差异"主要是各分支达到的有效截断长度不同，而不是同一截断下的随机散射；χ 本身不是好的排序变量，χ12 plateau 分支的物理 ξ 比 χ16/24/32 的都大。ξ_self 不是 replica 计算感受到的截断。

## 8. 双侧（MP-BP）截断的独立测试（2026-09-15）

依据 Woolls et al., arXiv:2609.05598（Matrix Product Belief Propagation）。输出在 `code2d/fpeps/data/biorthogonal_20260915/`（README、jobs.txt），脚本 `benchmark/bethe_row_eigenvalue.jl`、`biorthogonal_boundary.jl`、`biorthogonal_vumps.jl`、`make_rotated_seed_chi8.jl`、`transport_biorthogonal_rotated.jl`、`measure_biorthogonal_triple.jl`。全部新增文件、新目录；生产代码与已接受数据未动。

### 8.1 论文要点与本问题的对应
常规 bMPS 用边界自己的范数截断（正交投影 |R⟩⟨R|/⟨R|R⟩），对非厄米转移算符 δZ ∝ ε，且不规范不变；Bethe 估计 Z_B=∏⟨L|T|R⟩/∏⟨L|R⟩（斜投影 |R⟩⟨L|/⟨L|R⟩）给 δZ ∝ ε²；MP-BP 要求 L、R 是 Z_B 的驻点，即 T|R⟩∝|R⟩ 在 **L 的切平面**上成立（Eq.12）。无限系统等价于 Baxter 的非厄米 CTM，用 eig-CTMRG（角转移矩阵积 Λ 的左右不变子空间做斜投影，block-Krylov，balanced-Schur 规范）求解；作者明确说 naive 逐步更新常进入混沌极限环、不动点是鞍点、收敛不保证。KSVC 行转移厄米性偏差为 O(1)，是强非厄米情形。

### 8.2 两侧估计量检验（bethe_row_eigenvalue.jl）
用现有南北边界对算 ⟨s|T|n⟩/⟨s|n⟩ 与精确行本征值比较：gapped χ32/χ40 从 1e-9/3e-11 降到 3e-15/4e-15（ε² 兑现）；临界 χ8..32 只改善 3.6–17 倍（6.8e-4→3.6e-5 量级），因为临界边界自身 ε≈1e-2，ε² 正是观测值。估计量不是瓶颈，边界的截断方式才是。

### 8.3 双侧 power 截断（biorthogonal_boundary.jl）
每步长一行、以对侧未 grown 的边界为检验空间，用混合通道 r·l 的主左右不变子空间做斜投影截断（双侧 grown 的版本不可用：其混合通道主间隙 1e-7）。χ8：NN 误差 1e-4（VUMPS χ8 为 2.6e-3）；χ12/16：NN 误差 1e-5 量级，ξ_phys 19.5/20.7。但不收敛到不动点：χ8/12 慢漂（每步保真度 0.99987/0.9999991），χ16 进入两态循环，裸通道 ⟨L|R⟩ 间隙塌到 1e-10。

### 8.4 双侧 VUMPS（biorthogonal_vumps.jl）
把 Eq.(12) 写成广义本征问题 H^{LR}x = z N^{LR}x（AC 与 C 各一个；H 用 ⟨L|T|R⟩ 的混合环境，N 用 ⟨L|R⟩ 的），南北交替。v1（无阻尼、主 6 本征值窗口内按重叠选）：χ12 收敛（扫描间变化 1e-6，NN 误差 4.7e-6，两侧 λ 误差 4.8e-5，ξ_phys 22.5），χ16 收敛（1e-7，NN 1.5e-5，λ 误差 5e-6，ξ_phys 69.8），但前 20–40 次扫描在近简并本征值簇间乱跳，χ8 未稳住。加阻尼 α=0.5 后 χ8 干净收敛（9e-10），但 χ12 以上发散；"全有限分支延续 + power 步扩维"的变体更差。旋转框架的东/西对在所有变体下都不稳定。结论：pencil 在 χ≥12 病态（N 有近奇异方向、主本征值成簇），要稳定必须实现论文的 eig-CTMRG 机制，不是调参能解决的。

### 8.5 双侧边界不能直接接现有 direct LMPS 网络
把 v1 χ12 的南北对加传输得到的旋转边界接无逆 direct 测量：实部 log-ratio 收敛到 0.68243（漂移 1e-4），但相位偏差 3e-2 到 2.7e-1 且随深度增长（本征值相位不一致）。原因：direct 网络的 turn 来自自度量 tail ⟨n|T|n⟩，它假定边界是自度量驻点；双侧边界的自度量 Galerkin 残差 4e-3，H-vs-AL 0.2。要用双侧边界，replica 网络的 tail/turn 也必须是双侧的（混合 tail 或 CTM 角张量），这正是论文"环境与估计量都要两侧"的要求。

### 8.6 小结
两侧估计量的机制在本问题上成立；双侧截断在同 χ 下把局域观测量精度提高 5–20 倍；但要得到规范的 χ 族与 S̃，需要完整实现 eig-CTMRG（斜投影角张量 + 双侧 replica 闭合），这不是一次试验能完成的，且论文本身也警告收敛不保证。

## 9. 结果总图（2026-09-15）

`code2d/fpeps/data/biorthogonal_20260915/figures/session_summary_20260915.png`（脚本 `benchmark/plot_session_20260915.py`）。
A 面板：inverse-free S̃ 对 ln ξ_phys（六个诊断点，最小二乘斜率 0.166，Gaussian 单接头 0.155 参考线）；
B 面板：行转移本征值相对误差，单侧 ⟨n|T|n⟩/⟨n|n⟩ 对双侧 ⟨s|T|n⟩/⟨s|n⟩（临界圆点、gapped 方点）；
C 面板：最近邻密度关联误差，VUMPS 自度量 对 双正交 power step 对 双正交 VUMPS。全部为诊断值，无一被接受。

## 10. 图解说明（2026-09-15）

为什么双正交 (L, R) 不能直接接现有 direct LMPS 网络：见独立笔记 `fpeps_biorthogonal_lmps_zh.md`（图 1–4 在 `figs_fpeps_biorthogonal_lmps/`）。要点：每行相位两种边界都很小（≤6e-5），坏的是闭合对象（turn/cap）所依赖的自度量驻点条件，残差 4.5e-3 对 2.8e-6，五个 Z 的相位从 depth=4 起就偏差 0.04–0.27 且不随深度衰减。

## 11. inverse-free 的证明 note（2026-09-15）

`note_inverse_free_zh/main.pdf`（6 页，含张量图：图 1 基本张量 A_L/T/D/L/R，图 2 原路线 B=LTR、G=LR、H=BG⁻¹ 与 inverse-free 的 H=A_L、turn，图 3 六轨网络里六个 rail 的位置）：B = H_AC(AC)、G = H_C(C) 是 lmps_objects 审计过的恒等式，VUMPS 不动点处 B = κ·AL·G（κ = λ_AC/λ_C），逐区域标量在 S̃ 中相消；有限残差下两条路线之差 ≲ ε_G·cond(G)，cond(G) 为 Schmidt 值之比。含 summary.csv 数据表与 ε→0、病态-但-收敛两个检验。

## 12. 方法范围更正（2026-09-17）

> **方法范围更正（2026-09-17）。** 本文所有 S̃ 数值与网络图都建立在 **grown-tail** 路线上（`compressed_direct_rails`：一条边界与它自己的共轭夹一行 T，沿边界方向收缩）。用户要求的默认路线是 **bare/direct**（两列 `G=fp(M_L|M_R)`、三列 `B=fp(M_L|a|M_R)`，沿 seam 方向收缩，实现见 `benchmark/bare_direct_core.jl`，记录见 `data/bare_direct_20260917/README.md` 与 `data/direct_chi_curve/METHOD_STATUS.md`）。因此本文关于 inv(G) 放大、inverse-free、以及“双正交边界接不进 LMPS”的结论只对 grown-tail 网络成立，不能搬到 bare/direct。与路线无关、仍然有效的只有关于边界 MPS 本身的部分：中性 Z2 扇区审计、VUMPS 分支依赖、ξ_phys（它就是 bare 两列通道的关联长度）、两侧估计量 ⟨s|T|n⟩/⟨s|n⟩（它就是 bare 路线的 λ_T/λ_0）。

具体到本 note：第 2.2 节（inv(G) 放大）、第 6.1 节（无逆闭合）、第 8.5 节（双侧边界不能接 direct 网络）、第 9–11 节的图与证明，全部是 grown-tail 网络的性质。bare/direct 的首批结果（`data/direct_chi_curve/bare_direct_chi_scan.csv`）：χ4 0.43961、χ6 0.52426、χ8 0.57819 通过收缩检查；χ12 0.83778、χ16 0.85562 为诊断值（长度/相位门槛未过，输入边界未原生收敛）。同一批边界上 grown-tail 给 χ8 0.56162、χ12 0.8333、χ16 0.7928，两条路线在 χ16 相差 0.06。

## 13. 双正交边界接 bare/direct 路线的测试（2026-09-17，负结果）

数据 `code2d/fpeps/data/bare_biorthogonal_20260917/`（README、jobs.txt）。测量用未改动的 `benchmark/bare_direct_scan_point.jl`，输入是 09-15 的双正交 v1 χ12、χ16 三元组（只补了驱动要读的记账字段，态不变，fidelity 偏差 1e-15）。基线是同一驱动下的 VUMPS 边界（`data/bare_direct_20260917/chi12`、`chi16`）。

| 边界对 | ln(λ_T/λ_0) − 精确值 | cond(G) | replica 相位误差随深度 | Re S̃（诊断） |
|---|---|---|---|---|
| VUMPS χ12 | +3.6e-5 | 2211 | 3e-9，平 | 0.83778 |
| 双正交 χ12 | +5.1e-5 | 2431 | 0.04 到 0.81，不衰减 | 0.69110 |
| VUMPS χ16 | +1.2e-4 | 4270 | 7e-7，平 | 0.85562 |
| 双正交 χ16 | +8.3e-7 | 9032 | 0.45 升到 1.98，随深度增长 | 0.90332 |

结论：(1) bare 梯子的两侧行权重在双正交对上与 VUMPS 一样好甚至更好，固定点、cap、端点残差都收敛；(2) replica 相位在 bare 路线上照样不过，所以第 8.5 节与 `fpeps_biorthogonal_lmps_zh.md` 给出的原因（grown-tail 的自闭合）不是真正原因，“bare 的两侧闭合能接住双正交对”的设想被否定；(3) 实测到的、与规范无关的差别是实性：VUMPS 对的 bare G 通道所有本征值之比都是实数（1e-17），B 通道有精确的 ±i 对（real chart）；双正交对的 G 通道比值偏离实轴 2e-3 到 3e-2（χ12）、2e-2 到 4e-2（χ16），±i 对也破了。我的双正交求解器用复 pencil、按重叠挑本征向量，解跑出了 real chart；边界对不实，五个 Z_k 就不再是“公共每位点相位 × 实数”，相位门槛测的正是这个。要让双正交族可用，必须在 real chart 内求解（实 pencil、实本征向量或共轭对的实组合），这一步没有做。
