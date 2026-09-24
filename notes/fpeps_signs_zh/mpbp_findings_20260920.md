# 2609.05598 对当前 fPEPS 计算的具体启发与已验证结果

来源：[Woolls 等，Matrix Product Belief Propagation](https://arxiv.org/pdf/2609.05598)。本记录对应 2026-09-20 的隔离实验，不替换原独立双边求解器或已选曲线。

最有用的收获是：**双边驻点要求保留左右不变子空间；它并不要求保留的模式始终是按模排序的前几个。重新取最大模可能把已有驻点带到另一分支。** 论文 §V B 的式 (36)–(47) 区分了驻点条件与选谱；§V E 也明确说明迭代未必收敛。

## 1. 先构造正确的四方向表示

论文式 (57) 给出从两个独立的相对边界构造横向边界的字典，其中含有裸 `LR` 重叠 cap 的逆。直接将 `LTR` 的物理 cap 当作横向 MPS，只保证原物理行的收缩，并未自动给出水平驻点。

西边在本代码的 graded 图表中为

\[
W_{\rm new}=h_W J_N g_W^{-1}J_S,
\quad C_{NW}=g_W,\quad C_{SW}=I,
\]

其中 `h_W` 是 `LTR` cap，`g_W` 是 `LR` cap，`J` 补偿内部 dual 腿的 twist。先逐元素核对完整物理 cap 的缝合不变，再检查两个方向的原始 AC/C 双边方程；不是用反射猜另一个边界。

控制 11358253、11358301 对既有 gapless χ=4、6、8、12、16 全部通过。上下原始 MPS 不变，优化步数为零。最大两方向残差分别约 8.14e-13、1.30e-12、5.16e-12、8.87e-12、1.10e-10。

## 2. 已经定位 χ=8 的一次更新为何偏离原解

完整 corner 循环谱显示，原 χ=8 驻点在 odd sector 保留的模序号是 **[1,2,3,5]**，而最大模规则选 **[1,2,3,4]**。序号是在该 parity 块内按本征值模排序；不是虚拟腿的 parity 维数。

第 4、5 模的归一化本征值分别约 1.72069e-4、1.33087e-4，因此它们不是等模简并。仅要求成组保留简并模不足以修复这一差异。

| 同一已验证 χ=8 初态、同一更新 | 更新后原始联合残差 | 四边最大 fidelity 误差 |
|---|---:|---:|
| 最大模子空间 | 9.62618e-3 | 2.9953e-5 |
| 匹配原驻点的 corner 子空间 | 5.07246e-12 | 3.11e-15 |

这是 11358300 与 11358384 的直接对照：只改子空间选择便恢复了原驻点。它解释的是这次一步偏离；不证明所有历史不收敛都只有这一原因，也不证明原驻点是全局最佳近似。

χ=16 也保留非最大模前缀，但其完整参考更新对照尚未通过数值识别门槛。失败记录保留，未放宽原始方程、物理或 bare 验收门槛。

## 3. χ=12 已完成独立算法的 bare 复现

从正确字典初始化后，完整 simultaneous eig-CTM 在三轮中持续通过原始双边方程，最终残差 1.0692e-11；物理和 bare 流程独立检查完成。

| 量 | 新路线结果 |
|---|---:|
| bare/direct S_tilde | 0.765262255316884 |
| xi_MPS(R) | 6.100224842745814 |
| xi_MPS(L) | 6.100224842626140 |
| xi_pair | 25.498075373674958 |
| xi_physical（normal 关联） | 26.031723775399634 |

熵与原独立双边 χ=12 解相差 +3.72893e-11。传播到深度 1024，长度及相位检查通过，没有 grown tensor 或 endmap。与 Gaussian 的 normal 关联最大误差仍约 1.48246e-3：这复现了一个有限 χ 解，不是精确 Gaussian 极限的认证。也不能将论文针对 Bethe 估计量的二阶误差结论直接当作当前 replica 熵的误差保证。

## 4. 当前进度与下一步

- 从该已收敛 χ=12 四方向环境增长到 χ=16 的作业 **11358413** 已结束，150 轮后原始联合残差为 **1.8098012835e-3**，未收敛。AC、C 的原始及去度量残差都在 1e-3 量级，内层 cap 残差约 8e-15；混合度量最小/最大奇异值比约 1.49e-3。因此这次失败是外层方程尚未解好，不能归因于内层 cap 没算准，也不是仅由去度量放大的残差。
- 这个未收敛环境的 xi_MPS 约 15.06、xi_pair 约 76.22，**只作初态诊断，不加入已验收 xi–chi 或熵曲线**。较大的 xi 本身不是正确性的证据。
- 独立双边校正最初提交为 HTC **11358554**：把上述 R、L 作为初态交给原有 `anderson_targeted.jl`，两边继续独立更新；CTM 不再继续迭代。新入口 `benchmark/eigctm/correct_pair_anderson.jl` 保留初始快照、重算原始残差并记录哈希；旧 χ=16 目录仅提供模型与 spatial-bra 图表，数值状态来自新快照。HTC 的 AssocGrpBillingMinutes 阻止作业启动；确认同一用户、同一 `zdai` 账户在 SMP 有可用额度后，以完全相同的源码和数值参数迁移为 SMP **24133992**。新提交成功后才取消未启动的旧作业，迁移记录为 `data/eigctm_20260920/pair_correction16_smp_migration.json`。该 SMP 作业随后已正常结束，尚无新 χ=16 熵。

独立双边校正作业 SMP **24133992** 已完成：100 轮后 `converged=false`，最佳导出迭代为 0，残差仍为 1.8098012787e-3，`accepted_entropy=false`。这说明当前校正 map 在该初态附近没有找到下降方向；不把这次结果当作失败的物理环境。
- 冻结驻点的参考谱对照已经实现；它还不是处理任意未收敛环境或增大 χ 的算法。
- 后续应研究如何连续跟踪左右不变子空间及其数值条件，再做 χ=16、24、32 的原始方程、关联函数与 bare 检查。是否提高物理精度由 Gaussian 对照判断，不能用 xi 是否单调来选结果。

技术推导和 tensor 图：`independent_bivumps.tex` / `independent_bivumps.pdf`。源码、说明及可复制的 Slurm 命令在 `PEPS/code2d/fpeps/benchmark/eigctm/README.md`；精确命令数组和源码哈希在 `data/eigctm_20260920/*submission*.json`。最新 χ=12 测量及 χ=16 增长命令为 `dictionary12_measure_and_grow16_submissions.json`。

已编译 PDF v40 收录到 χ=12 bare 复现；本节 χ=16 的最终残差与校正提交是其后的进度补充，不声称已在 v40 PDF 中。

## 5. 待验证：单切口选谱、四切口传递子空间

新增 `transported_projectors.jl` 在一个切口选择原驻点的谱子空间，然后让右基底沿 corner 的作用方向、左基底沿相反方向做 QR 传递，在各切口重新平衡双正交化。其目的首先是消除四个切口独立匹配极小本征值造成的数值不一致，**不是**自动解决不同 χ 的选谱，也不是论文的 periodic Schur 实现。

在当前 graded 图表中，开放 corner 的普通线性作用候选为 \(B_i=J_i C_i\)，其中 \(J_i\) 是该切口的图形单位元。闭环应满足

\[
B_i B_{i+1}B_{i+2}B_{i+3}=(-1)^p\Lambda_i
\]

（逐 parity 块），因此不变子空间相同。实现必须将这个关系与已验证的显式闭环逐块比对；不能仅凭形式推导接受符号。

控制作业 SMP **24133994** 对 χ=4、6、8、12、16 的已知驻点检查上述关系、传递过程中不丢秩、局部双正交与不变性、四条显式 graded intertwiner 方程、更新后原始双边残差及四边 fidelity。χ=4、6、8、12 全部通过，更新后的联合残差分别为 2.55e-13、6.26e-13、5.07e-12、8.56e-12；传递中的最小奇异值比虽在 χ=12 降到约 1.7e-3，仍未发生秩丢失。χ=16 在参考谱匹配阶段被拒绝：一个保留模式的相对匹配误差为 4.13e-8，超过预设 1e-8。没有通过放宽阈值来制造通过结果。命令与源码哈希在 `transported_reference_submission.json`。这条路线目前只验证了已知小/中 χ 驻点的子空间传递，尚未解决 χ=16 增长。

随后提交了单独的 exploratory 作业 24134047，把匹配阈值放宽到 1e-7，但保留周期 intertwiner 的 1e-9 严格门槛。它在更新前即得到四条残差
\[
(3.40\times10^{-12},\;3.48\times10^{-15},\;3.80\times10^{-15},\;3.45\times10^{-9}),
\]
因最后一条失败而拒绝更新。这说明 χ=16 的障碍不只是单个本征值的匹配容差；不能用放宽谱阈值解决。失败证据记录在 `progress_chi16_tolerant_transport.json`。

## 6. Block-Krylov 原型

按照文章 Sec. V D 的思路，新增 `krylov_projectors_exploratory.jl`：对当前切口的左右参考基底分别构造
\[
K_R=\{V_R,\Lambda V_R,\ldots,\Lambda^K V_R\},\qquad
K_L=\{V_L,\Lambda^\dagger V_L,\ldots,(\Lambda^\dagger)^K V_L\},
\]
做 block-QR 后，在左右 Krylov 基底上解 generalized Petrov--Galerkin 问题，再做交叉 SVD 双正交化。实现只输出 projectors 和 graded seam，不更新物理环境。

在已验证的 \(\chi=12\) 驻点上，\(K=1,2,3,4\) 全部通过严格的四切口 seam 检查；最大 seam 分别为
\[
2.08\times10^{-11},\;1.52\times10^{-11},\;5.74\times10^{-12},\;5.74\times10^{-12}.
\]
在 \(\chi=16\) 上，四个深度都失败，最好是 \(K=2\) 的 \(6.07\times10^{-9}\)，\(K=3,4\) 约为 \(9.67\times10^{-9}\)。因此在当前分支上，增加 Krylov 深度并不能达到 (10^{-9}) 的周期兼容性门槛；没有 staging 熵。详细数据在 `progress_krylov_chi12_vs_chi16.json`。

## 7. 周期左右子空间联合 refinement

在 graded seam 的实际方向
\[
C_iR_{i+1}=R_i c_i,\qquad C_i^\dagger Y_i=Y_{i+1}c_i^\dagger,
\qquad Y_i=L_i^\dagger
\]
下，新增 `joint_periodic_refinement.jl`，对四个切口同时做阻尼 QR 更新并逐切口重新双正交化。第一次版本的左基底方向已修正，并保留了失败记录。

\(\chi=16\) 上，\(\alpha=0.01,0.02,0.05,0.1,0.2\) 的最佳最大矩阵残差分别为
\[
7.89,\;5.87,\;4.60,\;4.54,\;4.56\times10^{-9}.
\]
其中 odd sector 可以降到约 \(3.8\times10^{-10}\)，但 even sector 的最低值仍约 \(4.54\times10^{-9}\)。因此 refinement 确实降低了部分周期残差，却没有通过严格 \(10^{-9}\) 门槛；没有做 TensorMap seam、物理关联或 bare 熵测量。数据记录在 `progress_joint_refinement_chi16.json`。
## 8. 文章思路的进一步冻结审计（χ=16）

文章的 periodic invariant-subspace 观点给出了一个直接诊断：不能把每个 cut 都重新按模排序后独立截断，而应先在一个 cut 选定左右不变子空间，再沿四个局部因子输运，并检查周期 intertwiner。为此新增 `periodic_exact_transport_audit.jl` 和 `subspace_condition_scan.jl`。这里的 raw overlap 只包含输运基底的尺度，不能把它叫作 principal cosine；经过 QR 后，χ=16 参考分支的实际 principal cosines 约为

\[
(0.10104,\;0.53801,\;0.10104,\;0.53801).
\]

因此此前“最小 principal cosine 为 \(10^{-9}\)”的说法是错误的，已更正。当前参考分支的主要问题是 even sector 最后一条周期矩阵 seam，矩阵诊断为 \(5.92\times10^{-9}\)，对应 graded TensorMap seam 为 \(3.45\times10^{-9}\)；χ=12 的同一诊断最大只有 \(9.20\times10^{-13}\)。

为排除“只差一个模式”的可能性，对每个 parity 块扫描一次 mode swap，并把候选重新装回完整 graded TensorMap。χ=16 even sector 将第 10 模换为第 8 模的候选 \([1,2,3,4,5,6,7,8]\) 确实把冻结 TensorMap seam 降到 \(4.61\times10^{-10}\)，低于局部 \(10^{-9}\) 门槛；但其输运 principal cosine 只有约 \(0.0129,0.0452\)，条件明显变差。更严格的一次 simultaneous growth 显示，旧环境的原始双方向残差为 \(1.10\times10^{-10}\)，reference 候选更新后为 \(6.33\times10^{-11}\)，而这个 even-swap 候选更新后跳到 \(5.00\times10^{-4}\)。因此 even swap 只能作为谱诊断，不能用来产生 \(S_{\rm tilde}\)；odd swap 与双 swap 在第 4 个 cut 的 projector algebra 已经失败。

这组结果把文章方法对本问题的作用边界说清楚了：周期输运和 block-Krylov 可以发现错误的局部谱选择，也能区分 graph seam 与普通矩阵残差，但它们本身不会把 χ=16 的近似环境变成固定点。任何候选要进入 bare/direct replica 流程，还必须同时通过原始两方向 AC/C 方程、四个 graded seam、一次 growth 后的 boundary residual、长度/相位检查和 Gaussian benchmark。本节实验均为 `accepted_entropy=false`，没有改写既有环境、曲线或熵数据。
## 9. reference branch continuation 的结果

为了区分“seam 只是一次性误差”与“继续 growth 会发散”，又做了 4 次 reference branch continuation。每一步都从当前环境的 corner cycle 重新提取 reference 左右不变子空间，沿 graded 四角输运，再做一次 simultaneous growth；每步都重新测原始 boundary residual。结果为

\[
\begin{array}{c|c|c}
\text{step} & \text{最大 graded TensorMap seam} & \text{boundary residual}\\
0 & \text{--} & 1.10\times10^{-10}\\
1 & 3.45\times10^{-9} & 6.32\times10^{-11}\\
2 & 5.77\times10^{-10} & 2.31\times10^{-10}\\
3 & 2.32\times10^{-10} & 5.00\times10^{-9}\\
4 & 4.29\times10^{-11} & 1.10\times10^{-7}
\end{array}
\]

所以 seam 的确可以被连续压低，但环境同时离开原始固定点；不能以 seam 单调下降替代 boundary 方程收敛。随后尝试的通用 unitary CTM gauge damping 也不适用：\(\alpha=0.25\) 第一步就把 boundary residual 推到约 \(6.8\)，其余阻尼作业已停止。以上 continuation 和 damping 都保持 accepted_entropy=false，没有进入 direct/bare 熵流程。
