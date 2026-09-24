# w=1 的 north/south 转移算子关系：局域代数证明

日期：2026-09-14。对象是当前 `PEPS/code2d/fpeps/src/state.jl` 中的严格 KSVC D=2 张量、`bond_weight=(1,1)`。本证明不使用有限 χ 固定点，也不改变已有边界或熵数据。

**结论：在代码的有向指标转换及统一两模基底中，south→north 的模式变换确实给出完整整行转移算子的酉等价关系。** 下述结论只覆盖 north/south；不能直接推广到当前 south→rotated 的 W 映射。

## 1. 基底与结论

先按 `benchmark/gaussian_boundary_oriented_chart.jl` 处理每个方向的对偶指标箭头，再用 `V⊗V'` 展开每条双层腿。所有 fermionic twist 保留。这里的“物理腿”是边界 MPS 的两模腿，不是原模型的物理格点。

令

\[
W=\begin{pmatrix}0&1\\-1&0\end{pmatrix},\qquad
u=W^\dagger=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]

其 Fock 空间作用为 \(g=\Gamma(u)\)。在本次导出文件的扁平基底
\((|00\rangle,|10\rangle,|01\rangle,|11\rangle)\) 中，

\[
g=\begin{pmatrix}
1&0&0&0\\
0&0&-1&0\\
0&1&0&0\\
0&0&0&1
\end{pmatrix}.
\]

注意这与 `act_on_mode_coefficients` 显式使用的 \((|00\rangle,|01\rangle,|10\rangle,|11\rangle)\) 基底相差两个单占据基向量的排列。二者给出同一个两模操作，不能不换基就直接比较 4×4 数组。

对周长 L 的行，记 \(\mathcal U_L=g^{\otimes L}\)。则相同闭合约定下有

\[
\boxed{T_N^{(L)}=\mathcal U_L T_S^{(L)}\mathcal U_L^\dagger},\qquad
T_N^{(L)}\mathcal U_L=\mathcal U_L T_S^{(L)}.
\]

该关系对普通环闭合和总宇称 twist 闭合都成立；开边界情况需要同时变换横向端点。两方向各自的指标表示已包含在 T 的定义里，此式不额外施加未说明的空间反序或复共轭。

## 2. 局域恒等式

把每个局域双层 MPO 写成全向外的八模式 Choi 张量 \(C_d\)，模式分组固定为：左虚拟两模、上/输出两模、下/输入两模、右虚拟两模。直接由当前 KSVC 张量得到

\[
\boxed{C_N=(g\otimes g\otimes g\otimes g)C_S.}
\]

这不是根据固定点猜测的等式：局域全部 256 个 Fock 系数已用高斯整数 \(\mathbb Z[i]\) 的逐项运算核对。以下给出可独立检查的较紧凑代数证书。

局域 Choi 张量的真空系数为 1，可写成

\[
|C_d\rangle=\exp\!\left(\tfrac12\sum_{ij}(G_d)_{ij}f_i^\dagger f_j^\dagger\right)|0\rangle.
\]

从模型的局域张量得到的反对称矩阵为

\[
G_S=\begin{pmatrix}
0&-1&1&-1&i&-i&-1&-i\\
1&0&-1&1&-i&i&i&-1\\
-1&1&0&1&-1&i&-i&i\\
1&-1&-1&0&-i&-1&i&-i\\
-i&i&1&i&0&-1&1&-1\\
i&-i&-i&1&1&0&-1&1\\
1&-i&i&-i&-1&1&0&1\\
i&1&-i&i&1&-1&-1&0
\end{pmatrix},
\]

\[
G_N=\begin{pmatrix}
0&-1&1&1&i&i&-1&-i\\
1&0&1&1&i&i&i&-1\\
-1&-1&0&1&-1&i&-i&-i\\
-1&-1&-1&0&-i&-1&-i&-i\\
-i&-i&1&i&0&-1&1&1\\
-i&-i&-i&1&1&0&1&1\\
1&-i&i&i&-1&-1&0&1\\
i&1&i&i&-1&-1&-1&0
\end{pmatrix}.
\]

令 \(R=\operatorname{diag}(u,u,u,u)\)。直接矩阵相乘得到

\[
G_N=R G_S R^{\mathsf T}.
\]

这立即推出上述局域 Choi 恒等式。验证程序还独立用 Pfaffian 展开恢复两方向的全部 512 个 Fock 系数，确认没有遗漏高阶项或真空整体因子。因此这里不是仅检查两点协方差后假设整个张量相同。

## 3. 为什么能推出任意长度的整行恒等式

g 是实正交、宇称为偶、保持粒子数的操作。横向相邻张量的 g 因而在 cup/cap 收缩中抵消。它同时保持

\[
I,\quad P=\operatorname{diag}(1,-1,-1,1),\quad
D=\operatorname{diag}(1,1,1,-1),\quad PD
\]

对应的双线性收缩形式：对这些 M 都有 \(g^{\mathsf T}Mg=M\)。其中 D 是 Fock 对偶反序的 \((-1)^{N(N-1)/2}\) 因子。g 为偶操作，因此移过 graded 收缩时不会产生额外的奇算符换序号。

逐格应用局域恒等式，内部横向变换全部消去，只剩输出腿上的 \(g^{\otimes L}\) 和输入腿上的共轭作用。由 Choi 张量转回算子，正好得到
\(T_N^{(L)}=\mathcal U_L T_S^{(L)}\mathcal U_L^\dagger\)。因 g 与 P 对易，环上的 parity seam 同样保持。

这一步是任意 L 的代数推导。有限 L 核对用于独立验证程序中的箭头、基底和闭合约定，而不是用几个小环代替证明。

## 4. 实际完成的检查

- 从当前原生 `ksvc_ipeps()` 重新导出局域张量和整行矩阵，Julia 作业 **11240784，COMPLETED，4 分 14 秒**。
- 精确检查作业 **11240789，COMPLETED，1 秒**。所有 CSV 数值均要求实、虚部本身就是整数，随后转成 Python 整数对运算，没有四舍五入、容差判断或本征向量拟合。
- 局域 north/south 全部 256 个系数完全相等；两方向总计 512 个系数的 Pfaffian 展开一致；\(G_N=RG_SR^{\mathsf T}\) 精确成立。
- 去掉模式交换的必要负号后，局域有 **56 个系数不一致**，符号负对照有效。
- 完整行矩阵在 L=2、3、4、两种 seam 下逐元素一致。三个闭合给出零算子，报告明确标记，不拿零等式作为有效证据；其余三个是非零算子的完整检查。
- 非零样本中的 \(T_S\) 并非 Hermitian，且在这一统一表示中 \(T_N\ne T_S^\dagger\)。因此此处证明的是两个方向的酉等价，不能把它改写成未经指标转换的普通矩阵伴随关系。

早期检查先使用旧 Gaussian 预翻腿导出，发现其横向局域 gauge 是 W；当前原生导出所需横向 gauge 则为 W†。直接照搬旧局域公式的作业 11240786 明确失败，随后根据原生完整张量修正。整行 north/south 关系在两种表示中一致，且最终证明使用的是当前原生导出。

## 5. 能说明什么，不能说明什么

若 \(T_S|r\rangle=\lambda|r\rangle\)，则 \(T_N\mathcal U|r\rangle=\lambda\mathcal U|r\rangle\)。因此，在正确转换指标表示之后，可以把 south 的精确本征环境搬到 north；不需要 transfer 本身 Hermitian。

但该关系不证明主本征空间唯一，也不证明一个已经找到的有限 χ 固定点是目标物理解，不保证截断后的物理关联准确、ξ 随 χ 单调或 replica 熵正确。特别是简并时，对应本征子空间内的边界选择仍需核对。

**south→rotated 必须单独处理。** 本次也导出了实际 `rotated_boundary.jls` 所定义的 PEPS 与方向，简单 \(\Gamma(W)^{\otimes L}\) 共轭在非零 L=2、3、4 整行矩阵上不成立。之前对 χ=8 固定点的数值匹配不能升级为完整转移算子恒等式；可能需要不同的物理变换、空间反序或额外关系，但本记录不对尚未证明的形式作结论。没有据本次 north/south 证明新增任何可信熵点。

## 6. 复现位置与提交

工作目录：`/ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps`。

- 证书：`data/stable_boundary_20260912/direction_intertwiner_w1/exact_identity.json`。
- 原生局域/整行 CSV 与源哈希：同目录 `native_export/`、`native_export/export.toml`。
- 精确检查器：`benchmark/check_direction_intertwiner_exact.py`。
- 导出器：`benchmark/export_direction_intertwiner.jl`。
- 作业参数：同目录 `export_plan.json`、`exact_plan.json`；早期虚拟 gauge 失败原因保存在 `old_virtual_gauge_failure.json`。

只重跑精确检查，可以使用已有、哈希已绑定的原生导出，结果另存文件：

```bash
cd /ix/zdai/kangw/PEPS3EE/PEPS/code2d/fpeps
sbatch --parsable --exclude=htc-n77,htc-n79,htc-1024-n0 \
  --job-name=fp-direction-exact-recheck --cpus-per-task=1 --mem=4G --time=00:05:00 \
  jobs/run_python.sh benchmark/check_direction_intertwiner_exact.py \
  data/stable_boundary_20260912/direction_intertwiner_w1/native_export \
  data/stable_boundary_20260912/direction_intertwiner_w1/exact_identity_recheck.json
```

若源代码或边界参考发生变化，检查器会因源哈希不同而拒绝复用证书。此时应重新导出到新的结果目录并重新检查，不能把旧证书自动套到新模型或其他参数。
