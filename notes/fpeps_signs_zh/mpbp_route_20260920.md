# MP-BP 对当前 fPEPS 环境求解的启发

阅读日期：2026-09-20。来源：[Woolls 等，Matrix Product Belief Propagation，arXiv:2609.05598v1](https://arxiv.org/pdf/2609.05598)。下文前半记录最初的方法评估；实现后的状态另见文末更新。

## 文献定位

重点是 §III、§V B–E 和 SM4：联合驻点可通过双边 CTM 不变子空间求解。SM4 记录了逐张量更新和通用 Newton 类求根的稳定性困难。§V D 提出 balanced-Schur 表示；§V E 明确不保证迭代收敛。本文没有给出我们这个 exact KSVC fPEPS 的 benchmark。

## 与本项目方程的对应：本地代码核对与推导

在有限周长 N 上，令

\[
Q_N(L,R)=\frac{\langle L_N|T_N|R_N\rangle}
                   {\langle L_N|R_N\rangle}.
\]

对两个独立变量分别变分，得到

\[
\langle\delta L_N|(T_N-Q_N)|R_N\rangle=0,
\qquad
\langle L_N|(T_N-Q_N)|\delta R_N\rangle=0.
\]

因此右残差由左态的切空间检验，反之亦然。分母不为零、主根分支固定时，无限均匀极限的每格商为

\[
q=\lim_{N\to\infty}Q_N^{1/N}=\mu_{LTR}/\mu_{LR}.
\]

`benchmark/independent_bivumps/core.jl` 的 `bv_environment` 正是构造这两个混合通道，并检查 AC/C 中心商与 q 的关系；`bv_residual` 同时检查左右广义本征方程。这在目标方程层面与上述联合驻点相符。它不说明我们现有 Newton 更新已经等同于论文的 CTM 更新，也不替代数值验证。

`bv_white` 对局部混合度量作 SVD，给出 `Y† N X = I`。这只改变局部方程的坐标；不是选择并更新 CTM 的保留子空间。已安装 PEPSKit 0.8.1 的一般 `HalfInfiniteProjector` / `FullInfiniteProjector` 使用 SVD，因此之前试过标准 CTMRG 并不等于已经试过这条路线。

## 建议的独立实验

保留现有源码和选定数据，新建 eig-CTMRG 实验。先用已有带符号的 PEPS/CTM 收缩构造四个 enlarged corner，令循环积为 Λ。以左右不变子空间构造满足

\[
\Lambda V_R=V_R K,\quad V_L\Lambda=K V_L,
\quad V_LV_R=I,\quad \Pi=V_RV_L
\]

的投影，并更新四个方向；这里 V_L 是行映射，已包含所需的 bra 约定。第一版在 D=2、χ=4 上使用小型稠密 Schur 分解，有利于直接检查投影、不变子空间及 graded 方向；验证后再用 block Krylov。

记录左右不变子空间残差、`||V_L V_R-I||`、`||Π²-Π||`、投影范数及保留/舍弃子空间的谱分离。先分别将左右子空间基正交化，再测交叠奇异值，避免把有量纲的局部度量奇异值误当作主角余弦。对本模型，先验证 antiunitary 对称如何作用到各 corner；只有映射已确认的共轭谱块才要求一起保留，不能假定任意复矩阵都有共轭成对谱。奇异值本身为实非负数。

控制顺序：gapped χ=4 → gapless χ=4 → χ=8、12、16。同一初态比较旧解与新解，分别记录方程残差、带复相位的 Gaussian 关联函数误差、ξ_MPS 左/右、ξ_pair、ξ_physical。若检查通过，再接原有 G=LR、B=LTR 的 bare/direct 熵流程。不会把论文对其他估计量的误差阶结论直接当作我们 replica 熵的误差保证。

## 对 χ=12→16 的含义

选定点的 ξ_physical 从 26.0317 到 24.3360；两个边界每格重叠约 0.999999643，χ=16 新增四个方向的普通单边 Schmidt 权重合计约 1.61e-8。详见原始诊断 `data/independent_bivumps_20260919/selected12_vs_selected16_diagnosis.md`。

对固定近似态作可逆内部 gauge 变换只会使相应 transfer matrix 相似变换，不能改变本征值比与 ξ。因此期望的改善必须来自新的保留子空间或收敛分支，不能只来自双正交化的重命名。这是本项目的诊断方向，不是文章已证明当前选定 χ=16 解错误。新方法也不保证 ξ 对 χ 单调。


## 实现后的更新（2026-09-20）

独立实现位于 `PEPS/code2d/fpeps/benchmark/eigctm/`，没有替换原双边求解器。
已验证 graded 闭环中环境腿的 twist、左右不变子空间投影、完整等模组选择，
以及四方向同时更新。它目前使用小型稠密循环 Schur，尚未实现文中完整的
periodic Schur / block-Arnoldi。控制及逐步失败记录都保留。

新路线复现了原 gapless chi=4、6 的 bare/direct 值，熵差约 1e-9；
gapped chi=6、8、10、12 通过物理及 bare 检查，但 Gaussian 熵误差仍分别约
2.96%、3.02%、0.895%、0.757%。环境驻点、收缩稳定和有限 chi 精度分开报告。

从 chi=6 扩张到 gapless chi=8，逐方向和同时更新都未稳定收敛。
同时更新在 150 轮后的原始联合残差约 0.115；没有生成可接受的熵点。
这符合论文 §V E 对迭代稳定性不作保证的限定，但不能据此认定本模型无解。
下一项隔离实验是 gauge 对齐后阻尼；三个已知驻点的控制已通过，
alpha=0.2/0.5 的物理试算仍须经过全部原始验收。
所有提交命令、源码哈希和运行结果在 `data/eigctm_20260920/`；
编译说明和技术细节汇入 `independent_bivumps.tex`。


另一个具体收获来自式 (57)：水平 boundary 需要将物理 LTR cap 与裸 LR
重叠 cap 的逆组合。新控制 11358253 对原 gapless chi=4、6、8 的独立
上下边界验证了该转换，零优化步后两方向联合残差均低于 5.2e-12，
原上下张量和完整物理行 cap 保持不变。这修正了 CTM 暖启动的构造；
不改变之前的 bare 熵。eig-CTM 是否保持该已知驻点另由 11358300 检查，
chi=12/16 的同一转换由 11358301 检查。


已验证更具体的原因：原 chi=8 驻点的 odd corner 子空间对应按模排序的
[1,2,3,5]，而最大模更新改选 [1,2,3,4]。后一更新把残差推到 9.6e-3；
只改为匹配原 corner 参考谱，更新后残差保持 5.1e-12，四边 fidelity
误差低于 3.2e-15（11358384）。这与论文区分“驻点子空间”和“最大模
选择”的论述一致。它还不证明原解全局最优，或构成可稳定增大 chi 的
完整算法；下一步应研究保留子空间的连续跟踪，并继续 Gaussian 验证。
