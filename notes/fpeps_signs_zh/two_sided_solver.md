
## 术语纠正：当前实现不是严格的双正交 VUMPS

当前生产代码不是图示意义的标准 bi-VUMPS。它用一个参数化张量 `a` 生成 `S=native(a)` 和 `R=move(S)`，在每个 trial 点分别求左右 cap 的主固定点，再最小化完整 joint residual。这里的 `R=move(S)` 是模型方向约束，不是独立求出的左右 transfer-MPS 固定点。

标准 bi-VUMPS 应该对同一非 Hermitian transfer MPO 独立求

```text
T |R> = lambda |R>       <L| T = lambda <L|
```

然后分别 canonicalize，计算中心重叠 `E = Q_L† Q_R = U Sigma V†`，在每个 fermionic parity block 做

```text
Qhat_L = Q_L U Sigma^(-1/2)
Qhat_R = Q_R V Sigma^(-1/2)
Qhat_L† Qhat_R = I
```

当前代码没有执行这一步独立左右 MPS 的 mixed-gauge SVD。因此现有结果应称为“耦合边界残差求解器”，不能称为严格的 bi-VUMPS；现有 bare/direct 和 Gaussian 对照记录仍然有效，但不能赋予 bi-VUMPS 的物理解释。
# 固定 χ 的双边环境：方程、求解算法与验收

更新：2026-09-18。本文描述仓库**当前实验实现**，对象是严格 D=2 KSVC fPEPS 的 w=1 分支。相对于本文所在目录，代码根目录为 `../../code2d/fpeps`。PDF 版本见 [main.pdf](main.pdf) 的第 8.5 节。

**最新已完成结果。** 下文保留了早期扫描和方法演变，不能把早期端点当成当前通过点。当前用于比较的结果如下；所有 S̃ 均为 bare/direct，有限 χ 检查通过不等于无限 χ 物理精度认证。

| χ／分支 | 原方向联合残差 | ξ_MPS | ξ_pair | ξ_physical N/F | S̃ | 状态 |
|---|---:|---:|---:|---:|---:|---|
| 4 | 1.32e-14 | 1.170877 | 4.261893 | 4.788326 | 0.4903883073 | 双方向方程及收缩通过 |
| 6，(4,2) | 3.14e-10 | 1.177982 | 3.719791 | 4.248679 | 0.5344557052 | 独立驻点；检查通过，ρ₂误差稍大于χ4 |
| 8 | 5.27e-10 | 2.676483 | 10.821803 | 11.347482 | 0.6366027957 | 双方向方程及收缩通过 |
| 12 | 7.88e-12 | 6.100225 | 25.498075 | 26.031724 | 0.7652622556 | 双方向方程及收缩通过 |
| 16，selected_precision_trust16_from8 | 8.74e-12 | 5.935184 | 23.810412 | 24.335984 | — | 原模型上下联合方程通过；ρ₂误差3.92e-5，实际旋转模型待完成 |
| 16，refined_factor_alt16 | 5.29e-5 | 15.207702 | 70.589325 | 71.161584 | 0.9114471196 | 仅诊断：收缩通过，环境未通过 |

χ6不接入原χ4/8/12分支的斜率拟合。χ16另有较小残差的分支，不能把其ξ与上表诊断熵混配；高精度续算的中间迭代也不作为终态。当前作业、每步检查与提交命令见[实验日志](../../code2d/fpeps/data/selfconsistent_pair_20260917/README.md)，全部分支的[CSV总表](../../code2d/fpeps/data/direct_chi_curve/selfconsistent_pair.csv)将未通过环境的熵放在单独的 `stilde_bare_diagnostic` 列。

**χ16原方向已收敛，物理精度仍需另验。** 作业11343802在7个接受步后通过原始完整联合残差门槛。保存的同一组 ComplexF64 canonical 张量，用128位原方程复评得到残差4.7825e-13；没有重新正则化。北、南单边 ξ_MPS 分别为5.935183819264623、5.935183819264632；ξ_pair=23.810411880401265。normal/anomalous 的物理 ξ=24.335983727061745，connected density 的 ξ=4.985337441038739，三档模权重阈值1e-10/1e-8/1e-6均稳定，64个距离的模重构误差小于1e-13。这些 ξ 略小于χ12，不能由联合方程收敛推断物理精度随χ改善。Gaussian对照已完成：两点RDM最大误差3.92050e-5（Gaussian积分网格漂移6.16e-11），略大于χ12的3.63918e-5；r=31处normal绝对误差1.81232e-4、anomalous相对误差5.83454%、connected density相对误差41.3901%。有限χ误差仍存在，联合残差收敛没有自动改善这些误差。实际旋转模型须独立完成后才测新bare熵。

本次使用 `selected_precision_weighted_trust.jl`：每轮在完整Float64响应矩阵上逐步以128位响应修正软方向；所有128个参数方向始终保留。实际更新若未通过相对于新算128位响应的1e-8门槛，修正方向数自动从16增加到32、64或全部；末步16与32都失败，64才通过。所有接受步仍要求原两档四阶差分、真实舍入候选的正下降和完整joint<1e-9终止条件。它没有截断小奇异值，也没有将χ固定为修正方向数。显式旋转模型入口另用 `audit_model_precise_weighted_merit.jl`、`audit_model_precise_weighted_step.jl`、`model_precise_cached_weighted_trust.jl`，输入自己的实际旋转PEPS检查点；完整提交参数见实验日志的 `model_precision_submissions.json`。

**可以称为“双边 VUMPS 驻点方程求解”。目前实际使用的是带模型方向约束的解析梯度／Hessian 信赖域求根，不是原来的逐边 VUMPS 更新。** 双正交截断用于生成较大 χ 的初态；随后在目标 χ 重新求解。三个步骤不能混称为一次“截断”。

先看整个流程：**小 χ 父态 → 双方吸收一行 PEPS → 双边混合度量截断生成目标 χ 初态 → 在目标 χ 重新求解上下耦合驻点 → Gaussian／关联函数检查 → 实际旋转模型另行求解 → bare/direct 熵测量。** 全部边界都是无限、平移不变 MPS；父态只提供初值，目标 χ 的环境需要重新优化。

**χ24 的初态例外与当前进度。** 从新收敛 χ16 做双边长行和谱投影时，原切口检查失败，尚未由这条路线得到 χ24 种子。每个宇称块在目标切口附近的两个本征值约为 −8.44e-9、−3.77e-9；局部相对间隔虽约为0.553，但非正规度量的投影对输入敏感：相对1e-12的矩阵扰动使投影变化约0.12%–0.45%，超过预设1e-3。纯对角相似变换平衡也没有通过同一坐标下的扰动检查。这里失败的是初态谱截断的稳定性检查，不能称为 χ24 优化不收敛，更不能只改阈值就当作通过。

隔离的 `prepare_embedded_joint24.jl` 因此提供另一种初值：将 χ16 按宇称等距嵌入 χ24 的(12,12)虚空间，仅在旧虚空间块外加入1e-3扰动，再规范化并重建上下关系。这**不是双正交谱截断**，后续仍须在完整目标 χ24 参数空间求解同一双边驻点方程。11345054保存的两侧与父态每格 fidelity 均约0.9999997254，重建 chart 的 fidelity 误差小于7e-14；但初始 joint=0.0239161、ξ_pair=37.3894，仅是零步快照，没有 χ24 结果或熵。ξ 对新引入弱模敏感，不能用初始 ξ 变大来评价改善。自己的高精度响应和完整 Jacobian 检查分别由11345456、11345458执行，χ4入口控制为11345455；完整命令在 `wide_model_precision_submissions.json`。原小χ入口、符号收缩及1e-9最终门槛保持不变。

**实际旋转模型与大 χ 收缩的独立检查。** 实际旋转 χ16 的完整128位 Jacobian 求解和自适应精度求解，前12个接受步的联合残差最大相对差为1.30e-8、ξ_pair最大绝对差为1.78e-9；但两者最终残差都约5.97e-6，仍未通过。原方向已经收敛不代替实际旋转方向收敛，也不能用旧分支的熵补上新点。

旧的未收敛 χ24 环境还揭示了另一个问题：在 bare/direct 深度4，冻结同一组四副本端点、只改变最后闭合的 sliced 边，单项复数结果的最大相对差为3.44e-7。仅将已算出的标量项改为128位求和，full/character 偏差仍约5.68e-8；因此最终标量累加不足以解释这项偏差。随机复数 graded 网络的直接/优化/sliced 收缩26项对照全部通过，但这不能保证这些病态端点的数值精度。保持所有宇称和指标、逐项可精确逆转的二进制成对缩放虽然降低范数包络，却未改善 χ24 的两种闭合结果；该实验没有加入生产 bare 路线。这些诊断使用的是旧未收敛环境，不能据此宣布新 χ24 驻点或 χ32 收缩失败。

**二副本传播误差的高精度定位。** 作业11346975对旧 χ24、bare 深度4的同一组 Float64 几何张量和远端 cap，逐元素精确提升后重算传播、正范数归一化及最终张量闭合。完整带符号表示与 character 表示的相对差在128/192位分别为2.07620e-31 / 5.26863e-51；两档精度的 signed/character 结果分别相差7.89525e-32 / 1.28715e-31。signed 结果相对原 Float64 改变2.52973e-9。只提升最后闭合的精度时，差异仍约9.50e-10。因此这里确认了传播/归一化浮点误差的影响，没有改变符号、环境、G/B 或 cap，也没有放宽门槛。这只验证二副本 Z2C 的深度4数值：四副本高精度收缩尚未验证，旧环境仍未收敛，没有新增熵点。来源和独立十进制复算见 `two_replica_propagation24_audit/local_verification.json`。

**χ24 求解准备检查已通过。** 常规分区作业11346769完成全部288个参数方向的128位 Jacobian/SVD，无奇异方向截断；重构误差1.73e-37，实际方向误差1.04e-25，两档四阶差分误差6.34e-8 / 3.96e-9。只读候选降低了加权优化目标，但完整 joint 暂时增大，不能当成收敛结果。自己的自适应响应检查11346770已启动，后续求解仍由该检查控制。确切结果见 `embedded_precision24_fullJ_v3/local_verification.json`。

**作业恢复。** `PREEMPTED` 和 `TIMEOUT` 是调度终止状态，不是数值门槛失败。必须先由 `sacct` 确认终态并由整个 `squeue` 确认父作业已退出，才读取其检查点。`restore_model_precise_checkpoint.jl` 直接恢复保存的八个 canonical 张量，要求零改动并复现对应日志行的 joint 与 ξ；恢复报告明确标为 `snapshot_only`，不能当作优化终态。续算使用原模型模板和恢复检查点，重新检查自己的响应与 Jacobian。近期恢复链显式指定 `--partition=htc`，覆盖作业脚本默认的 `preempt` 分区；精确命令见实验目录 `regular_partition_recovery_submissions.json`，后续作业进度以实验 README 为准。

| 名称 | 在当前实现中的含义 |
|---|---|
| 双边 | 上、下边界一起决定有效环境和重叠度量，并检查两侧方程 |
| VUMPS 型方程 | 固定 χ 的混合正交中心张量／中心键驻点条件 |
| 当前主迭代 | 使用模型方向约束的解析导数信赖域求根 |
| 双正交截断 | 为较大 χ 生成初态；没有直接优化 ξ_pair |
| bare/direct | 环境求解后，以 G、B 固定点测量 S̃ |

方程和实际迭代见第 1–4 节，截断见第 5 节，验收与结果见第 6–7 节，可复制的提交命令见第 8 节。

## 1. 要求的两个边界是什么

用一对无限、单胞平移不变 MPS 表示上下边界，记作 R 和 S。概念上，我们近似行转移算子的右、左固定环境：

\[
T|R\rangle\simeq\Lambda|R\rangle,\qquad
\langle L|T\simeq\Lambda\langle L|.
\]

这里的 \(\langle L|\) 来自 S 的**空间边界 bra**。实际代码调用 `_spatial_boundary_bra(S)`，包括方向、共轭和费米子 twist；不能将它换成忽略腿次序的普通逐元素共轭。`boundary_1.jls`、`boundary_3.jls` 分别保存实际 north、south 边界。

用长为 n 的环定义每格行本征值商，有助于理解我们优化的量：

\[
q(R,S)=\lim_{n\to\infty}
\left[\frac{\langle L_n|T_n|R_n\rangle}
{\langle L_n|R_n\rangle}\right]^{1/n}
=\frac{\mu_T}{\mu_0}.
\]

\(\mu_T\) 是上下 MPS **夹一行 PEPS** 所形成的水平混合通道主本征值；\(\mu_0\) 是上下 MPS 直接重叠通道的主本征值。实际直接求无限通道固定点，不构造这个有限环。当前实数表示要求 q 为正，并求 \(f=\log q\) 的驻点。

固定 χ 时求的是 MPS 流形上的投影固定点条件，不是要求完整 Hilbert 空间的 \(TR-qR\) 严格为零。驻点可能不唯一；小残差既不证明主分支，也不证明无限 χ 的精度。

## 2. “双边”体现在有效环境和度量

给定当前整对 (R,S)，重新计算四个水平 cap：夹 PEPS 的 \(l_T,r_T\)，以及直接重叠的 \(l_0,r_0\)。R 写成混合正交形式：

\[
A_C=A_L C=C A_R.
\]

留出一个中心张量或中心键，分别得到作用算子 \(H_{AC},H_C\) 和混合重叠度量 \(N_{AC},N_C\)。图中的所有费米子交换仍由原有 graded contraction 执行。

示意地，省略固定的物理腿恒等映射：

\[
N_{AC}(X)=l_0 Xr_0,\qquad N_C(Y)=l_0 Yr_0.
\]

H 则包含实际 PEPS 行及 \(l_T,r_T\)。在两个方向都检查：

\[
H_{AC}(A_C)=z_{AC}N_{AC}(A_C),\qquad
H_C(C)=z_C N_C(C),\qquad
z_{AC}=qz_C.
\]

最后一个因子很重要：AC 方程比 C 方程多一个局部 PEPS 张量，不能直接要求两个 z 相等。south 方程用实际反方向 transfer，并交换 R、S 的角色构造。

因此，R 的方程依赖 S，S 的方程也依赖 R；双方的自重叠规范化并不能把这个混合度量自动变成单位阵。各自的 self-Galerkin 残差很小，不等于这组耦合方程已经收敛。

对应实现：[selfconsistent_pair_core.jl](../../code2d/fpeps/benchmark/selfconsistent_pair_core.jl) 的 `joint_operators` 和 `joint_audit`。

## 3. 当前上下变量是否完全独立

**方程检查两个方向，但当前主求解器并没有独立优化两套任意复数张量。** 在当前 w=1 模型、物理基和方向约定下，已对局域张量及收缩检查了上下方向的变换关系。求解器据此使用

\[
S=\mathrm{native}(a),\qquad R=\mathrm{move}(S),\qquad a^\dagger a=I,
\]

其中 a 在固定的中性物理子空间及实数表示内变化。`move` 在已核查的 mode 基中实施由

\[
W=\begin{pmatrix}0&1\\-1&0\end{pmatrix},\qquad u=W^\dagger
\]

诱导的 Fock 空间变换，再转回原有张量约定。实数虚键表示由 Takagi 规范构造。这是当前模型的约束，不是对任意非厄米 transfer 都适用的假设，也不说明 T 是 Hermitian。

代码中的旧 χ=8 `anchor` **只确定局域物理支持基**；它不把新环境虚键限制在 χ=8。给定目标虚空间和宇称多重度后，目标 χ 内允许的实数参数都会参加优化。但模型方向关系、中性支持、实数表示和宇称空间仍是限制，不能描述成“任意两套复数 MPS 的无约束优化”。

约束流形上的梯度为零本身不充分；因此验收时重新规范化整对边界，检查原始复数 north/south 的 AC、C 方程，而不只检查约束梯度。通过该检查也不等于证明其他分支不存在。

另外，这个上下关系不是原模型到 90° 旋转模型的环境映射。bare/direct 所需的旋转环境，使用实际 `rotl90(ksvc_ipeps())` **另行求解**。

对应实现：[analytic_joint_hessian.jl](../../code2d/fpeps/benchmark/analytic_joint_hessian.jl) 的 `analytic_variational_chart`。所依赖的审计记录在 `data/selfconsistent_pair_20260917/` 下的 `direction_constraint_audit.toml`、`variation_audit.toml` 及各自的 `rotated_` 版本。

## 4. 当前怎么迭代：对驻点方程作信赖域求根

每个目标 χ 的迭代如下。

1. 从当前 a 重建 R、S，求新的混合通道 cap 和 q；求导同时包含上下方向贡献。
2. 令 Q 为 a 的正交补，\(a^\dagger Q=0\)。计算切向梯度 \(g=Q^\dagger\nabla_a f\)。
3. 计算 cap 对张量变化的响应，得到解析 Hessian K。对于一个 cap 的 \(Mx=\mu x\)，解带规范固定的线性方程：

   \[
   \begin{pmatrix}M-\mu I&-x\\x^\dagger&0\end{pmatrix}
   \begin{pmatrix}dx\\d\mu\end{pmatrix}
   =\begin{pmatrix}-(dM)x\\0\end{pmatrix}.
   \]

   所以改变 a 时，环境也参与求导。不是把旧 cap 冻结后仅解一次局部本征问题。Hessian 包含正交约束的曲率项。
4. 在信赖半径 Δ 内求 \(\min_{\|\delta\|\leq\Delta}\|g+K\delta\|^2\)，使用完整 Hessian 谱，不按其本征值删去更新方向。
5. 用极分解形式回到规范化流形：

   \[
   a_{\rm trial}=(a+Q\delta)(I+\delta^\dagger\delta)^{-1/2}.
   \]

   重新求 trial cap、实际梯度，以实际与预测的梯度范数平方下降比决定接受和调节 Δ。代码要求下降比大于 0.1。
6. 接受后检查**完整双边残差**；满足门槛才结束。最后从保存态重新计算，核对它确实对应最后一次接受的状态。

这是一种驻点求根方法，并非保证 q 每步增大的变分最大化，也不是 maximize fidelity per site。不能承诺每一步完整联合残差或 ξ 都单调。解析导数另外用实际标量 \(\log q\) 的自适应中心差分／Richardson 外推核查；差分用于核查，不是用于生成主更新方向。

**2026-09-18 的补充审计：步子模型和接受范数需要区别。** 上述 K 是标量 f 在极分解坐标中的 Hessian；实际接受步子用的是新状态的投影梯度范数。令 \(g_a=\nabla_a f\)，其等价残差为

\[
F(a)=(I-aa^\dagger)g_a,\qquad
dF=(I-aa^\dagger)dg_a-da(a^\dagger g_a)-a(da^\dagger g_a).
\]

对于 \(da=QV\)，真实水平响应包含 \(-V(a^\dagger g_a)\)，而标量 Hessian 使用 \(-V\operatorname{sym}(a^\dagger g_a)\)。若反对称部分非零，两者的 merit 预测不同；不能从“标量 Hessian 审计通过”推断它就是接受范数的完整 Jacobian。在 χ=6 停滞点，独立残差差分确认了这个区别。另一个针对最后被拒绝小步的检查，得到旧预测 −6.95e-9、真实导数 +4.18e-9，差分支持后者；该首次检查的第二个方向差分未分辨，后续自适应 Richardson 核查（11316455）已通过，并保存所有步长记录。

因此另保留实验脚本 `projected_merit_trust.jl`：以真实 dF 构造矩形 Jacobian J，求 \(\min_{\|\delta\|\le\Delta}\|F+J\delta\|^2\)，接受标准和原完整联合门槛不变。χ=4 控制达到 1.20e-14，但 χ=6 仍停在 1.34e-3 附近，**这个修正没有单独解决 χ=6，也不证明它是所有旧停滞的唯一原因**。χ=16 的同初态比较也已完成：第 40 步的原方法／修正方法 joint 分别为 1.64046e-3／1.64839e-3，未见明显加速。原路线和结果均保留；χ=8、12 的独立联合方程验收不因此失效。

另用完整实数 Stiefel 切空间（包括 QV 与 aK，K 为实反对称矩阵）作小 χ 控制，新残差为 `ga-a*sym(a†ga)`，保留规范冗余。`full_stiefel_merit_trust.jl` 的导数核查及 χ=4 控制通过，但 χ=6 仍停在 joint≈1.44e-3；尚无证据说明这个改法能解决大 χ。它是独立实验，不替换上面的主算法。

旧的“解 AC/C 局部广义本征问题 → regauge → 阻尼更新上下边界”的实现仍保留在 `solve_selfconsistent_pair.jl`。当前结果来自 [analytic_dense_trust_adaptive.jl](../../code2d/fpeps/benchmark/analytic_dense_trust_adaptive.jl)，这两种迭代算法应分别标注。

所谓“内层”是给定当前边界求 cap、局部线性响应等问题；“外层”是更新边界后仍满足耦合固定点方程。内层本征残差小不能替代外层残差。

## 5. 双正交截断用在哪里；是否按 ξ 截断

当前较有效的初态生成方式是：从已核查的较小 χ 父态出发，**上下边界各吸收实际的一行 PEPS**，再用双方共同决定的度量投影到目标维度。父态只作初始猜测，之后在目标 χ 完整重新优化。

若双方增长后的混合通道有 m 个简并主模，使用整个孤立主空间。把左右主模归一成 \(\operatorname{tr}(r_j\ell_i)=\delta_{ij}\)，构造

\[
D_N=\frac1m\sum_i r_i\ell_i,\qquad
D_{S,\mathrm{bra}}=\frac1m\sum_i\ell_i r_i.
\]

这两个量未假设为正定的厄米密度矩阵。代码检查它们在主空间内换基不变。对度量的保留谱子空间，用左右 Schur 向量 V、U 构造

\[
J=(U^\dagger V)^{-1}U^\dagger,\qquad JV=I,\qquad P=VJ.
\]

因此 P 一般是斜投影，不是 \(VV^\dagger\)。检查 Gram 条件数、\(P^2=P\)、与度量的对易关系，并拒绝切开未分辨的简并或共轭本征对。共轭配对是这里的非厄米谱检查，**不是说 SVD 奇异值必须成复共轭对**。

当前这条初态路线先保留总维度 \(2\chi_{\rm target}\)，中间投影的两个宇称扇区等额保留。然后用已核查的交换代数分解出两份等价的 graded 副本，并核查奇虚规范下的等价性。**拆分后的**宇称多重度不是再强制设为相等；实际 4→6 得到 (3,3)，4→8 得到 (4,4)。候选配对按与父态双方 fidelity 的乘积选择，不按 ξ 或最终熵选择。不能将这种副本结构普遍解释成任意 fMPS 的 Majorana 简并。

这叫“由双边混合度量决定截断”，**目前没有直接最小化 ξ_pair 的误差或保证保留全部慢模**。度量的截断谱与计算 ξ_pair 的混合转移谱也不是同一个谱。ξ_pair 只作诊断，算法不保证它随 χ 增大。

对应实现：[upward_block_seed.jl](../../code2d/fpeps/benchmark/upward_block_seed.jl)、`audit_leading_block_metric.jl`、`block_metric_power.jl`、`split_block_components.jl`。这是一条有明确维度和谱间隙检查的实验初态路线，还不是通用任意大 χ 截断器。

## 6. 验收分三层

**方程层。** `joint_residual` 取以下相对误差的最大值：north/south 的 AC 与 C 原始广义本征残差、去混合度量后的残差、\(z_{AC}=qz_C\) 一致性、两个方向的 q 一致性。当前通过门槛为 \(10^{-9}\)。另外要求 cap 残差 \(<10^{-10}\)、主模模间隙 \(>10^{-8}\)、混合度量条件数 \(<10^{10}\)、逆作用检查 \(<10^{-9}\)。不能用单个小梯度替代这些检查。

**物理层。** 检查 1/2/3-site RDM 的迹、厄米性、正定性及 Gaussian 误差；测量保留复相位的 normal、anomalous、connected-density 关联。既看短程，也结合当前有限 χ 的物理衰减长度解释远距离误差。临界态在 \(r\gg\xi\) 的误差包含有限纠缠效应，不能仅据一个远距离点判定错误分支。

**熵层。** 实际原方向和旋转方向分别求环境，再用原有 **bare/direct：G 为两列、B 为三列固定点，H=BG^{-1}** 的网络测量。另查费米子符号对照、相位、端点残差、随收缩长度稳定性。优化初态时的 row-growth 不会把熵定义改成 grown-tail；此处不使用 grown-tail 或 endmap 熵。

因此 `complete=true` 只表示该任务执行完成；`joint_converged=true` 只表示方程层通过；`accepted_entropy=false` 表示尚未认证熵。三个标志不可互相替代。

ξ 的定义也分开保存：单边 \(\xi_{\rm MPS,N/S}\) 用各自 self 通道；\(\xi_{\rm pair}\) 用实际两边的直接混合通道，不夹 PEPS；算符相关的 \(\xi_{\rm physical}\) 从含 PEPS 列的实际关联函数通道及非零 residue 提取。一般不相等，也没有一个能单独证明环境正确。

## 7. 扫描记录与 χ4/8/12 主分支

下面五点是早期同一主扫描算法的端点，**全部未收敛**，列 ξ 仅便于诊断，不代表可信熵点。

| χ | 完整联合残差 | ξ_pair | 数据子目录 |
|---:|---:|---:|---|
| 6 | 1.33926e-3 | 13.37245 | `midchi_dense6` |
| 8 | 6.63812e-3 | 10.39797 | `midchi_dense8` |
| 12 | 8.83930e-3 | 10.71140 | `midchi_dense12` |
| 16 | 1.00742e-3 | 20.98738 | `midchi_dense16` |
| 24 | 2.07294e-1 | 31.01642 | `midchi_dense24` |

另从已核查 χ=4 双边增长：χ=6 的 20 步端点残差 1.31316e-3，仍未收敛；χ=8 经 20 步、续 60 步、再续 15 步后通过原有门槛：

\[
\epsilon_{\rm joint}=5.2717177477\times10^{-10},\qquad
\xi_{\rm pair}=10.8218030113.
\]

数据为 `data/selfconsistent_pair_20260917/upward_block4to8/solve_resume2`，job **11316040**；已核对输入 SHA、最终报告及末次接受态残差一致。这与旧 χ=8、ξ_pair≈241.34 的驻点是不同记录，不能混用旧观测量或旧熵。新点最终态的物理／Gaussian 测量 **11316152**、算符模式测量 **11316153** 均已完成，结果见下文。

实际旋转 χ=8 的 60 步端点残差为 1.06924e-4；续跑 **11316154** 已在第 36 步通过联合门槛，残差 4.8993461402e-10，ξ_pair=10.8218030043。输入 SHA 和保存态残差一致性已核查。**11316153** 也已完成：新原方向最终态的 normal/anomalous 物理衰减长度为 11.3474822773，density 为 2.3482498890，均对所测 residue cutoff 稳定；r≤64 的模展开重构误差小于 8.3e-15。这是数值通道核查，不代替 Gaussian 精度比较。

后续 **11316162** 已完成新 χ=8 的 bare/direct 测量：\(\widetilde S=0.6366027956792095\)，长度漂移 6.99e-11，最终相位误差 1.14e-10，端点残差 1.86e-13；完整 signed 与 character 的浅层对照误差不超过 6.06e-13。输入来自上述原方向和实际旋转方向最终态，SHA 核对通过。输出为 `upward_block4to8/bare`。

这是**联合方程和 bare 收缩都通过的有限 χ 数据点**，物理精度仍需随 χ 检验，`accepted_entropy` 仍为 false。最终态 Gaussian 测量 **11316152** 已完成：1/2/3-site RDM 物理检查通过，2-site 最大元误差 1.59502e-4；anomalous 在 r=1、7、31 的相对误差约 0.0878%、1.63%、20.76%；normal 的 Gaussian 值为零，r=7 实测幅度 1.26142e-3。新分支的二点 Wick 关系最大偏差为 3.77e-11，旧 ξ_pair≈241.34 分支为 3.32e-4；这个诊断不单独证明整个近似态是 Gaussian。比较图仍在原 `direct_chi_curve/joint_branch_correlations.{png,pdf}`。

| χ | ξ_MPS（完整宇称谱） | ξ_pair | bare S̃ | 方程／收缩状态 |
|---:|---:|---:|---:|---|
| 4 | 1.1708770101 | 4.2618928289 | 0.4903883073404 | 两者通过，有限 χ |
| 8 | 2.6764831294 | 10.8218030113 | 0.6366027956792 | 两者通过，有限 χ |
| 12 | 6.1002248428 | 25.4980754180 | 0.7652622555597 | 两者通过，有限 χ |

补测最终 χ=4 的物理模式后（11316188），normal/anomalous 的 ξ 为 4.7883258552，density 的 ξ 为 0.9915878609。χ=12 对应为 26.0317238202 和 5.1652614217，均对所测 residue cutoff 稳定。三点图与机器可读表在 `direct_chi_curve/joint_bare_smallchi.{png,pdf,csv,json}`。同一组熵对不同 ln ξ 的**相邻割线斜率**为：

| ξ 定义 | χ=4→8 | χ=8→12 |
|---|---:|---:|
| self（完整谱） | 0.1768544 | 0.1561739 |
| pair | 0.1569078 | 0.1501207 |
| physical normal/anomalous | 0.1694621 | 0.1549516 |
| physical density | 0.1695992 | 0.1632143 |

已有 Gaussian 每结点 ln L 拟合约 0.1550–0.1553；当前点数和尺度范围不足以判断渐近一致性，也不能按是否接近 Gaussian 来选择 ξ 定义。JSON 另保存三点有限窗口回归，不作渐近认证。没有混用旧 grown-tail 熵或旧环境的关联长度。

χ=12 的原方向续跑 **11316208** 和实际旋转续跑 **11316362** 分别达到联合残差 7.88e-12、7.96e-12；最终态测量为 **11316365/66**。两点 RDM Gaussian 最大元误差为 3.63918e-5；**11316385** 的 bare 长度漂移 2.92e-10，最终相位误差 1.23e-10，端点残差 1.07e-12。原方向输出 `upward_block8to12/solve_resume`，bare 输出 `upward_block8to12/bare`。χ=12 的二点 Wick 偏差约 6.33e-5；一般有限 χ MPS 不必保持 Gaussian 性，不能把单个 Wick 诊断当作物理认证或否决条件。

更大 χ 应逐级推进，不能用一次 χ=32 尝试代替中间扫描。χ=8→16 已生成初态并完成同初态的两种步子比较，尚未收敛；原方法续跑 80 步后 joint=4.73049e-4，正在继续求解。另从已通过的 χ=12 做相邻扩维到 16（**11316459**）。新 `upward_large_parent_seed.jl` 仅把父态上限扩至 16、目标上限扩至 32，原有完整谱尺寸、宇称、投影与副本等价检查不变，原初态脚本保留。

这次 12→16 的原始 south 副本分解未过泄漏门槛；直接交换子 SVD 后仍为 1.31e-8，原 1e-8 门槛保持不变。随后 **11316575** 从已通过检查的共同度量投影 north 分量出发，用已验证的模型内部逆方向关系构造新的 south 伙伴。该**新初态**的完整检查通过：方向往返误差 6.51e-16，父态 fidelity 乘积 0.99999224，初始 joint=1.30584e-3、ξ_pair=26.65858895；目标 χ=16 的首段 60 步（11316591）已结束但未收敛：末态 joint=0.12649，梯度=8.36e-5，混合度量条件数约 1.42e6；最佳 joint 也只有 1.16e-3。该端点不用于 bare 熵测量。这不是接受原先失败的 south 分解，也不是原模型到旋转模型的环境变换。

## 8. 怎样提交与续跑

所有数值在 Slurm 上执行。以下命令从 fPEPS 根目录运行；输出目录须使用新名字，不能覆盖已有结果。

```bash
cd /ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps
EXCLUDE=htc-n77,htc-n79,htc-1024-n0,htc-n45,htc-n92
RUN=data/selfconsistent_pair_20260917/my_upward4to8

# 生成 χ=8 双边初态；χ=6 将最后一个参数改为 6，并换 RUN。
sbatch --job-name=fp-pair-seed --exclude="$EXCLUDE" \
  jobs/run_cpu.sh benchmark/upward_block_seed.jl \
  data/selfconsistent_pair_20260917/real_root4 "$RUN" 8
```

任务结束后检查 `$RUN/enrichment.toml` 的 **`seed_ready=true`**，不能只看作业退出码或 `complete=true`。确认后：

```bash
sbatch --job-name=fp-pair-solve --time=01:00:00 --exclude="$EXCLUDE" \
  jobs/run_cpu.sh benchmark/analytic_dense_trust_adaptive.jl \
  "$RUN/seed" "$RUN/solve" 20

# 前一次完成且尚未收敛时才续跑；仍传最初的 seed，保证 checkpoint hash 相符。
sbatch --job-name=fp-pair-resume --time=01:00:00 --exclude="$EXCLUDE" \
  jobs/run_cpu.sh benchmark/analytic_dense_trust_adaptive.jl \
  "$RUN/seed" "$RUN/solve_resume" 60 "$RUN/solve/iterate_pair.jls"

# 再需续跑时，使用新的输出目录及上一次的 checkpoint。
sbatch --job-name=fp-pair-resume2 --time=01:00:00 --exclude="$EXCLUDE" \
  jobs/run_cpu.sh benchmark/analytic_dense_trust_adaptive.jl \
  "$RUN/seed" "$RUN/solve_resume2" 100 "$RUN/solve_resume/iterate_pair.jls"
```

不要将这三条在无依赖条件下同时提交。当前原方向成功记录使用上述 20/60/100 的分段上限，最后一段在 15 步通过。每次启动会重置信赖半径，因此一口气跑相同步数不保证逐步轨迹完全相同。

令 `FINAL` 指向实际要测量的最终目录，作业完成后再提交：

```bash
FINAL="$RUN/solve_resume2"
sbatch --job-name=fp-pair-measure --time=00:45:00 --exclude="$EXCLUDE" \
  jobs/run_python.sh benchmark/followup_midchi.py "$FINAL" 8
sbatch --job-name=fp-pair-modes --time=00:30:00 --exclude="$EXCLUDE" \
  jobs/run_cpu.sh benchmark/measure_joint_physical_modes.jl "$FINAL"
```

第一条包括 ξ_MPS／ξ_pair、RDM、关联函数及 Gaussian 对照；第二条计算含 PEPS 的物理算符衰减模（当前 bounded 实现 χ≤16）。测量任务也允许分析未收敛端点，但会保存未收敛标记。

旋转方向使用 `upward_rotated_block_seed.jl`，父态为 `real_rotated4`；求解和续跑改用 `analytic_dense_trust_rotated.jl`，其余参数结构相同。它使用独立输出目录，不能拿原方向 checkpoint 代替。bare 提交与通过的 χ=4 控制记录见[完整计算日志](../../code2d/fpeps/data/selfconsistent_pair_20260917/README.md)。

原方向和实际旋转方向都通过联合门槛后，才准备 bare 输入。以下路径需替换为**本次同一 χ 的最终输出目录**；原方向和旋转方向都必须来自已完成的求解：

```bash
NATIVE_FINAL="$FINAL"
ROTATED_FINAL=data/selfconsistent_pair_20260917/my_rotated4to8/solve_resume
STAGE="$RUN/bare_input"

# 仅核查元数据、SHA 并建立输入链接，不执行张量收缩。
python benchmark/stage_joint_bare.py "$NATIVE_FINAL" "$ROTATED_FINAL" "$STAGE"

# 当前该扫描入口支持 χ=4、6、8、12、16；实际收缩在计算节点执行。
sbatch --job-name=fp-pair-bare --time=01:00:00 --exclude="$EXCLUDE" \
  jobs/run_cpu.sh benchmark/bare_direct_scan_point.jl "$STAGE" "$RUN/bare"
```

检查 `bare_result.toml` 的 `contraction_passed`，并保留长度、相位和端点误差；它不等于无限 χ 精度认证。较大 χ 使用单独的 preflight／预计算入口，不能只修改此处维度上限。

查看状态用 `sacct -M htc -j JOBID --format=JobID,State,Elapsed,ExitCode -X`；日志在 `logs/作业名_JOBID.out`、`.err`。进度看输出目录的 `progress.toml`，最终看 `report.toml`，物理任务看 `followup_status.json`。不能把 Julia 启动期无输出当作计算失稳。


## 2026-09-18 补充：直接优化完整双边方程残差（实验中）

通过验收的 χ=4、8、12 bare/direct 点仍来自上文路线。为解决标量梯度变小而完整联合残差不降的问题，新增隔离入口 `canonical_joint_trust.jl`：在相同的模型方向约束参数空间中，直接最小化双方 AC/C 方程的规范化残差，以及中心本征值和两侧行商的一致性残差。它不是另一次独立的标准双边 VUMPS 迭代，也没有按 xi 排序截断。

对每侧 X=AC,C，残差使用 `[N_X^{-1}(H_X X)/z_X-X]/||X||`，其中 `z_X=<N_X X,H_X X>/<N_X X,N_X X>`。中心 C 取自通道正密度 rho 的正平方根，响应通过 `C dC+dC C=d(rho)` 求解；AC、AR、混合环境及逆度量的响应均包含在 Jacobian 中。信赖域按真实残差下降验收，每5步做额外差分检查。最终仍以原 `joint_audit` 的完整复数联合残差 <1e-9 验收。

χ4 加1e-3切向扰动后两步回到 joint=5.31e-11。χ16 从旧路线累计200步（joint=2.58e-4）的 checkpoint 出发，40步降到 **joint=5.80e-5，xi_pair=23.8981**，仍未收敛；没有可信的新 χ16 熵点。该端点后续测得 xi_MPS=5.9261914015，两点RDM Gaussian最大元误差3.9217978294e-5；完整记录见计算日志。χ6 平衡奇偶扇区在16步停于 joint=3.92e-4，也未通过。

复现这次 χ16 试验（使用新的输出目录，审计先完成）：

```bash
ROOT=data/selfconsistent_pair_20260917
EXCLUDE=htc-n77,htc-n79,htc-1024-n0,htc-n45,htc-n92,htc-n74
sbatch --job-name=fp-canonical-audit16 --exclude="$EXCLUDE" jobs/run_cpu.sh \
  benchmark/audit_canonical_joint_residual.jl "$ROOT/upward_block8to16/seed" \
  "$ROOT/my_canonical_audit16" "$ROOT/upward_block8to16/solve_resume2/iterate_pair.jls"
# 上述report.toml的passed为true且源文件/代码SHA匹配后再提交：
sbatch --job-name=fp-canonical-solve16 --exclude="$EXCLUDE" jobs/run_cpu.sh \
  benchmark/canonical_joint_trust.jl "$ROOT/upward_block8to16/seed" \
  "$ROOT/my_canonical16" "$ROOT/my_canonical_audit16/report.toml" 40 \
  "$ROOT/upward_block8to16/solve_resume2/iterate_pair.jls"
```

小步长的实际更新方向导数审计未通过。随后宽步长审计11317378通过：相邻两档Richardson误差为1.75e-6和3.90e-6，支持小步长存在消减误差的判断。独立入口 `canonical_joint_trust_fd_checked.jl` 在原线性检查失败时要求实际方向连续两档差分同时支持组装J和新方向响应（相对误差<1e-4），否则仍拒绝该步。11317445已完成40步，joint=6.0169071008e-6、xi_pair=23.7094550449，仍未通过；原求解器保留，最终联合残差、逆作用或费米子符号检查均未放宽。最新轨迹位于原 `data/direct_chi_curve/joint_chi16_convergence.png`；熵图仍只含本路线通过环境和bare收缩检查的 χ=4、8、12 点。该补充同步到 LaTeX 源；PDF 构建记录见完整计算日志。


## 2026-09-18：完整响应基与中心因子精度核查

后续完整响应基试验保留 Jacobian 的全部右奇异向量，在该正交基中重新计算解析导数，避免把近软方向仅作为大列向量的相消组合。χ4扰动控制通过；χ16的40步终态为联合残差 **5.9361748843e-7**，xi_MPS=5.9348458333，xi_pair=23.8244622130。两点RDM的Gaussian最大元误差3.92011e-5。这仍未达到1e-9联合门槛，不能加入bare熵图；原目录的收敛与关联函数图保留了未收敛标记。原始AC/C残差也尚未通过，不只是逆度量残差的问题。

独立诊断发现，新残差实现中先求自通道密度rho、再对角化开方，可能损失小Schmidt值的相对精度。在已完成的FD检查χ16端点（与上述40步响应基终态分开）上，最小Schmidt值约4.11e-6；两种rho仅差2.26e-15，但密度开方构造的AR右等距误差为1.74e-7，而直接使用MPSKit中心因子的SVD正因子重建AR后为5.21e-12。χ4对照没有这一明显精度损失。

隔离入口 `audit_canonical_factor_residual.jl` 用直接中心因子的SVD构造正C，保留全部奇异值；其Sylvester响应也使用这些未平方的奇异值，并检查重建AR的等距误差<1e-9。χ4扰动恢复通过（joint=5.31e-11），χ16一般方向导数审计通过，但实际软更新方向未通过连续两档差分检查，**尚未启动该版本的χ16求解**。进一步的环境特征向量迭代精修正在独立审计中。

这项发现针对新canonical残差cache的数值实现，不能据此解释全部原VUMPS失败。原始联合方程检查、已通过的χ4/8/12双边边界和bare结果、Gaussian以及RK/TFIM路线均保持不变。最新入口、完整提交命令和各任务状态见[完整日志](../../code2d/fpeps/data/selfconsistent_pair_20260917/README.md)。


## 2026-09-18：精修 χ=16 分支的驻点检查（最新完成结果）

采用直接中心因子和环境特征向量迭代精修后，χ12增长得到的另一χ16初态通过一般导数和实际更新方向审计。原方向35步后停在joint=5.2853820450e-5、xi_pair=70.5893249470；实际旋转模型独立求解39步后停在joint=5.2855963642e-5、xi_pair=70.5860955634。两者都没有通过联合门槛。

原方向端点的xi_MPS=15.2077019601；含PEPS柱、具有非零算符留数的物理衰减长度：normal/anomalous为71.1615844071，connected density为12.1348157549。两点RDM Gaussian最大元误差1.37193e-4。r=31的anomalous相对误差0.2571%，connected density相对误差81.81%；较大的xi不代表所有观测量更准确。这里没有bare/direct熵值；短条带局部purification测试中的熵不可替代它。

为诊断停滞，令F为完整AC/C规范化残差向量，J为它在当前切空间中的Jacobian。此端点的||F||=1.14304e-5、||J^T F||=1.35391e-10，但原始log(row quotient)的水平梯度范数为0.0596076776。沿其单位梯度方向的解析导数0.0596076775，h=1e-6的有限差分为0.0596077817。因此这个点不是原始商的驻点；不能把残差平方和优化停滞等同于双边环境收敛。小||J^T F||也不独自证明局部极小值：本次有限步长的merit差分仍有明显高阶项，原值均保留在诊断报告。

完整物理空间的检查给出：north/south AC残差在已保留rank2支撑之外的范数仅约3.8e-14，而AC残差本身约1.07e-5。因此此点的主要残差不是固定物理支撑遗漏造成的。χ4控制的joint=5.31e-11，原始商梯度约1.47e-10，与上述χ16明显不同。

独立诊断入口为 `benchmark/audit_refined_stationarity.jl`，输出在 `refined_stationarity16` 与 `refined_stationarity4_control`。χ6(4,2)的导数审计通过缩小有效扰动到h=1e-7达到相对误差4.63e-6、2.49e-5；无效的大步长记录为失败，不改变基点唯一性或1e-4精度要求。后续χ6求解与χ16原始商梯度重启仍是实验，最终都必须通过完整联合方程及独立旋转方向检查后才做bare/direct验收。最新任务和可复制提交命令见[计算日志](../../code2d/fpeps/data/selfconsistent_pair_20260917/README.md)。


## 2026-09-18：χ16 bare/direct 诊断与更新图

现已对 `refined_factor_alt16` 及独立实际旋转的 `refined_factor_rotated_alt16` 做完整带符号 bare/direct 收缩（11341060），不使用 grown tail 或 endmap。两侧环境尚未通过联合方程：native/rotated 残差分别为 5.2853820450e-5 / 5.2855963642e-5。这里允许测量未收敛环境，是为了把环境误差和 replica 收缩误差分开；正式验收入口和标准不变。

|收缩深度 d|诊断 S_tilde|相位误差|
|---|---|---|
|256|0.9115819898082691|6.1360e-9|
|512|0.9114477391444780|6.2025e-9|
|1024|0.9114471254702039|3.2396e-9|

512→1024 的相邻漂移为 6.13674e-7；生产检查使用最后三个深度，报告的最大相邻漂移仍为 1.34251e-4，高于 1e-7。最终端点残差 8.54686e-11，相位门槛 1e-8 通过，但 `contraction_passed=false`、`environment_gate_passed=false`、`accepted_entropy=false`。**0.9114471 只能作为该未收敛边界的有限深度诊断值，不能当作新的通过点或用于斜率拟合。** d 是无限边界上的收缩深度，不是有限物理系统的边长。

这个端点的 xi_MPS（含完整宇称扇区）=15.2077019601，xi_pair=70.5893249470，xi_physical(normal/anomalous)=71.1615844071。bare 后端附带输出的 xi_boundary_even=7.6108714787 只含偶扇区，不能替换完整 xi_MPS。其它优化分支的 xi 与本次熵不可混配。

图保存在原 `data/direct_chi_curve/joint_bare_chi16_diagnostic.{png,pdf}`：左上是 χ4/8/12 通过点与 χ16 诊断叉号，另三个面板显示长度与相位检查。原 `joint_bare_smallchi` 通过点数据不变。`joint_chi16_correlations` 也补入了 Gaussian L256 初态优化后的未收敛终态，不能拿 Gaussian 初态本身的 xi 代替它。

新求解试验 `balanced_checked_merit_trust.jl` 在固定虚拟规范平衡后，对每个实际更新方向做两档四阶差分核查，保留全部奇异方向并要求真实 merit 下降。χ4 的 1e-3 扰动控制两步恢复到 joint=2.17891e-10（11341072）；χ16 试验11341073尚在运行，不能从χ4控制推断其会收敛。初始χ16审计中最软 Hessian 方向仍失败；该新路线明确只验证实际采用的方向，不声称整个 Jacobian 审计通过。完整提交命令见计算日志。


## 2026-09-18：χ16 direct 延长检查通过，环境仍未通过

同一对原方向/独立实际旋转边界延长到d=4096（11341145），并复现原扫描所有重叠深度。d=2048/4096分别得到0.9114471197699459 / 0.9114471195789520。最后三档1024/2048/4096的最大相邻漂移为5.69980e-9，最后一档漂移1.90994e-10；linear tail residual=1.15415e-9，端点残差1.40009e-10，最终相位误差8.76556e-9。因此原生产门槛下 **contraction_passed=true**。完整带符号cycle sum仍用于熵，character只作浅层审计；原后端没有改变。

这更新了上一节深度1024时的长度失败结论，但没有更新环境：native/rotated joint仍为5.28538e-5 / 5.28560e-5。**环境未通过，accepted_entropy仍为false；0.9114471196是这组固定近似边界的稳定direct诊断值，不能据此认证χ16环境，也不加入通过点的拟合。** 同一 `joint_bare_chi16_diagnostic` 图已更新到4096，JSON/CSV记录完整深度及来源SHA；原4/8/12通过点曲线保留。

另一个新的加权求解试验把原始商梯度的水平残差F=(I-aa†)g替换为R=F C^{-1}，C来自直接MPS中心因子的正SVD因子。所有Schmidt值保留，C可逆时零点不变；导数完整包括dR=dF C^{-1}+F d(C^{-1})。这是优化范数的条件调整，不是按xi_pair截断，也不是把transfer matrix假定为Hermitian。最终完整联合方程门槛仍1e-9。

χ4/16的随机及实际更新方向均通过两档四阶差分<1e-5，脚本/源/检查点SHA匹配。χ4加1e-3扰动后的两步控制收敛到joint=4.53241e-11、xi_pair=4.2618928211（11341316）。χ16最多20步试验11341313及其Gaussian观测量followup11341317仍在运行/等待；它们来自scalar_from_refined16，**不是本节已经测得S_tilde的refined_factor_alt16边界，不能混配两条分支的S与xi**。入口 `factor_weighted_merit_trust.jl` 与可复制提交命令见计算日志。


## 2026-09-18：完整响应基续算终态及 Gaussian 初态对照

χ16 加权求解的20步终态之后，完整响应基续算40步已完成（11341506）。所有奇异方向保留，40个实际接受步均通过在线导数和真实merit下降检查；源、检查点、helper与最后行一致性已核对。终态 joint=5.14573850585e-4、xi_pair=55.8673349455、加权梯度范数0.00856832666，仍未通过1e-9联合门槛。后段已明显停滞，不能把优化范数下降等同于联合方程改善。原收敛图 `joint_chi16_convergence` 加入这段续算，重启处数据连续性检查保持。该分支没有新增bare熵；0.9114471196仍仅属于前述refined_factor_alt16及其独立实际旋转环境。

Gaussian L256 拟合初态现已做零步快照及相同观测量测量，避免把优化后结果误当初态。两点RDM误差定义为与Gaussian参考的最大矩阵元绝对差。

|χ16分支|联合残差|xi_MPS|xi_pair|xi_physical normal/anomalous|两点RDM误差|
|---|---:|---:|---:|---:|---:|
|有限L256 Gaussian原始拟合初态，0步|2.54497e-3|11.858952|76.585276|76.620003|5.25981e-4|
|同一初态经20步scalar优化|6.16033e-2|4.990879|18.705391|19.196607|5.33160e-5|

优化后局域RDM误差约缩小十倍，但联合残差增大、xi减小，r31 anomalous相对误差从3.7108%变为8.0105%。不能只凭大的xi或单一局域量评价环境。这里的L256只表示初态构造的有限环长，不是精确无限Gaussian环境。两个端点均未收敛，没有各自独立实际旋转求解及bare结果。前后完整关联函数图在原目录 `joint_gaussian_chi16_correlations.{png,pdf}`（11341965），源码/输入SHA已核对并目视检查。

χ6加权试验停在joint=2.01732e-5、xi_pair=4.20724，联合残差较初态更大，局域RDM误差几乎不变，仍无通过点。另一个移动正交切空间坐标的只读导数审计虽然通过，但χ16有限步长的真实merit变化几乎不变，因此没有据此再启动求解器。当前通过本路线联合方程、独立旋转及bare收缩检查的有限χ点仍只有4、8、12；这些检查也不等于无限χ物理极限认证。


## 2026-09-18：χ16最新续算停止原因与χ6新驻点

原目录 `direct_chi_curve/joint_chi16_convergence` 已补入χ8扩维分支的5步weighted续算。它从joint=5.93617e-7出发，终态joint=1.57924e-6、xi_pair=23.80965，未通过1e-9。此次停在导数检查，不能解释为已证明优化极小值。更细步长审计使原随机方向通过，但实际更新方向组装/fresh差1.19723e-8仍超过1e-8，且两档四阶差分要求未通过；依赖续算从未启动并已取消。两点RDM Gaussian误差3.92187e-5不替代联合收敛。现有0.9114471196仍只属于另一对refined_factor_alt16固定近似边界，长度和相位检查通过，环境检查未通过。不同分支不能混配S与xi。

χ6出现另一驻点：从已通过双边χ12投影得到(4even,2odd)初态，随后在目标χ6独立优化；原方向和实际旋转方向分别8步达到joint=3.13511e-10与3.13533e-10。独立旋转方向使用实际旋转PEPS，未由native边界直接运输。xi_MPS=1.1779819574、xi_pair=3.7197912081、xi_physical N/F=4.2486792497、xi_nn=1.5667834813。1/2/3-site RDM物理检查通过；两点RDM Gaussian误差9.37354e-4稍大于χ4的7.76806e-4，所以联合驻点通过并不说明该分支更准确。bare/direct作业11342433已提交，目前没有该新χ6的熵。详细审计、作业ID与可复制命令见[计算日志](../../code2d/fpeps/data/selfconsistent_pair_20260917/README.md)。


随后11342433已完成：χ6(4,2) bare/direct S_tilde=0.5344557051926131，最后长度漂移2.55795e-12，深度256的相位误差7.24909e-12，端点残差1.81161e-13，完整带符号收缩通过。原生单边converged标志仍为false，但本次独立检查的双边joint已通过；不改写旧标志。物理精度未认证，accepted_entropy=false。该点在joint_bare_chi16_diagnostic图中以独立空心菱形展示，不把它接入原4/8/12分支或斜率拟合；其Gaussian局域误差稍大于χ4的事实保留。


## 2026-09-18：128-bit精度检查与Gaussian续算

χ16五步weighted终态的实际更新方向在128-bit完整Jacobian下通过检查：组装/fresh相对差1.74580e-22，h=0.0016/0.0008两档四阶差分误差6.47619e-9/4.04766e-10。全部128个奇异方向保留，SVD重构误差4.58e-38；cap与隐式响应使用原始带符号方程残差精修。Float64的响应相对高精度约有2.6e-8误差，支持此前这一方向受数值精度限制，但不能据此解释全部VUMPS失败。

对目前最好χ16保存态（canonical_metric_basis16），只将AL/AR/C/AC逐元素提升到128-bit，不改变波函数，完整联合残差从5.93617488432e-7变为5.93617485555e-7。主导项仍为原始AC残差；因此环境确实尚未收敛，不能靠精度重评估通过1e-9门槛。完整Jacobian的只读候选也显示非线性步长限制：全步和1/4步使weighted merit变差，1/16步才改善。这些候选未保存成新的终态，没有对应熵。

隔离的precise_weighted_trust.jl将残差、Jacobian、SVD和回缩算术提升到128-bit；候选转回ComplexF64后重新评估真实merit，要求下降且预测/实际gain比通过，并逐步核查实际方向。最终仍使用原始完整joint<1e-9，保存边界与bare/direct后端格式不改。11343214为χ4的1e-3扰动控制；11343215只有自身完整Jacobian审计及此控制均通过才启动χ16最多12步。11343263/64依赖该求解终态，分别测物理模和xi/RDM/Gaussian关联。程序结束不等于环境或熵认证。完整提交命令见实验README。

Gaussian L256/M256 χ16 fidelity追加500步，loss从1.27969178812e-7到1.27368096194e-7，梯度4.22498e-7，仍未达到1e-9。新态导入及有限环费米子符号检查通过；零步联合残差2.22958e-3，xi_MPS=11.8685951781，xi_pair=64.5167962663。normal/anomalous的xi_physical=64.6014971278在算符留数阈值1e-10/1e-8/1e-6下稳定；密度xi_nn在1e-10下为11.7900971213，在1e-8/1e-6下为4.0532066637，需保留阈值依赖。两点RDM Gaussian误差从5.25981e-4增至5.70563e-4，fidelity改善并未改善这一局域基准。该新Gaussian分支没有bare熵，不能与另一分支的0.9114471196混配。
