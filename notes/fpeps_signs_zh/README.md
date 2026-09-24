# 严格 fermionic PEPS 与 six-R 的符号方案

[中文 note](main.pdf) · [LaTeX 源码](main.tex) · [小规模检查结果](checks.json)

[双边求解方法与提交命令](two_sided_solver.md)：解释双边 VUMPS 驻点方程、
当前方向约束下的解析信赖域算法、双正交初态与 ξ 的区别，以及三层验收。
PDF 已加入对应的方法节；此方法说明优先于下面按时间保留的早期状态。

[实现阶段补充](implementation.md) 记录后续 CAR/BdG 检查、Hamiltonian 约定字典与有向 LMPS cap 的发现；
[实现目录](../../code2d/fpeps/README.md) 记录当前接口、作业证据和仍未验收的部分。

固定 χ 双边求解记录：χ=4 精修后的 bare S̃=0.4903883073405；
旧 χ=8（ξ_pair≈241.34）虽联合残差达到 9.04e-12，长距离密度关联仍未通过 Gaussian 核对。
新 χ=4→8 增长后精修分支（job 11316040）联合残差 5.27e-10、ξ_pair=10.82180301；
实际旋转环境也通过联合门槛（4.90e-10）。新 χ=8 的 bare S̃=0.6366027956792，
符号对照、相位、端点和长度检查通过；2-site Gaussian RDM 最大元误差 1.60e-4，
关联函数仍有有限 χ 误差。该有限 χ 数据不代表无限 χ 物理熵已认证。
χ=12 也已通过两个方向的联合检查及 bare 检查：S̃=0.7652622555597，
ξ_MPS=6.1002248428、ξ_pair=25.4980754180，2-site Gaussian RDM 误差 3.64e-5。
三点图及不同 ln ξ 的相邻割线已加入方法 note；χ=6、16 仍未验收。
这些点属于新的联合求解分支，不替换下述原生 VUMPS 数值。
详见[逐步记录与提交命令](../../code2d/fpeps/data/selfconsistent_pair_20260917/README.md)。

**2026-09-17：主路线已纠正为 bare/direct。χ=4 得到 S̃=0.439613897919，
完整符号、相位、长度检查通过；重构一致性和目标态精度尚未认证。
旧值 0.434719992643 属于 grown-tail。新旧及 Gaussian benchmark 见
[实现补充首节](implementation.md)；PDF 正文同步增加 bare 定义和数值结果。**

这是独立的研究设计 note，讨论 `PEPS/fPEPS_renyi_2.pdf` 中的严格 D=2 Gaussian fPEPS 如何接入现有 environment / LMPS / six-R 路线。它不是现有 RK/TFIM production 的方法认证，也没有修改 production 代码。

内容包括完整局域张量系数、区域 Fock 顺序、普通 replica permutation 与 FSWAP 的区别、双层公式的适用范围、cycle 分解与中心闭合、截断后的 parity string、边界条件和分阶段验证方案。

原 PDF 的验证范围为局域 CAR 代数与小规模 occupation/Gaussian 对照，其“尚未实现”描述对应设计阶段。
后续实现进展见上述补充文件：已加入带 parity-sector 求和的有限 χ LMPS/six-R 测量、
物理 ket sewing 对照和长度极限检查；尚未完成无限 χ 外推。设计正文中的“下一步”应结合实现补充阅读。

运行小规模检查及编译（在此目录）：

```bash
sbatch check_slurm.sh
sbatch build_slurm.sh
```

检查依赖 NumPy 及仓库内 Gaussian 参考模块；编译使用 XeLaTeX。数值检查 jobs：`11211749`、`11211764`（补充显式 SWAP 矩阵检查后重跑）。
