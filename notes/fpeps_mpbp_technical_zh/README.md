# fPEPS 收缩路线图解

从 [main.pdf](main.pdf) 阅读，源文件是 [main.tex](main.tex)。

当前 PDF 共 18 页。正文按“未知量 → 收缩图 → 方程 → 一轮更新 → 测量”展开。
文末第 16–18 页新增 eigCTMRG 实现说明（源文件 `eigctm_details.tex`）：
四角四边图、graded 闭环、左右 Schur 子空间与交叉 SVD、相邻切口相容性、
同时更新流程、edge 到 boundary MPS 的转换，以及未通过的历史候选。
构建 11425655 在 htc/preempt 完成，退出 0；输入与 PDF 哈希校验通过。
参数与命令在附录，完整旧日志保存在 [实验记录附册 v18](appendix_experiments_v18.pdf)。

| 页码 | 内容 |
|---|---|
| 1 | 主路线：两条独立无限 MPS 的双边中心残差求根 |
| 2 | 局域双层张量、两条无限 MPS、canonical 中心 |
| 3 | rho_A 自转移固定点、C_A 键中心、A_C 站点中心：定义、维数、Schmidt 系数和三幅图 |
| 4 | 两种 transfer channel，四个 cap 怎样得到；明确 mixed gauge 下四个固定点的左右 MPS 尾 |
| 5 | site/bond 的 H、N 四张开放张量网络图；明确 a=R.AC、c=R.C |
| 6 | 双边广义本征方程；中心度量白化与全局双正交规范的区别；明确当前规范 |
| 7–8 | 一轮更新、切空间、完整导数链、八组残差的定义 |
| 9 | Fock 交换、空间 bra 映射、replica 标签置换的符号 |
| 10 | bare/direct G、B、三分区接缝与熵 |
| 11 | 三种关联长度、parity 扇区、外围模和算符权重 |
| 12 | 各条历史路线的区别与尚未解决的问题 |
| 13–14 | 公式到代码的对应、门槛、保存结果、Slurm 提交命令 |

本文说明的是当前实际实现：固定 chi 的两个独立 MPS 的中心自洽方程，
用最大残差块信赖域方法选步。它没有实现标准 biVUMPS 的中心更新流程，
也没有在当前循环中更新 CTM 四边四角。当前 chi16 仍未通过全部环境验收。
新版还明确区分 mixed gauge 的四个环境：\(\ell_T\) 是
\(A_L\!-​O\!-̅B_L\) 左固定点，\(r_T\) 是
\(A_R\!-​O\!-̅B_R\) 右固定点；\(\ell_0,r_0\) 则删去中间的 \(O\)。

所有数值源文件保持不变。本轮只重写文档及编译脚本。
旧正文和 PDF 未删除：appendix_experiments_v18.tex / .pdf；
旧 README 历史状态保留为 appendix_readme_v18.md。

## 编译

全部通过 preempt，当前目录执行：

    sbatch --parsable --clusters=htc --partition=preempt build_slurm.sh

图由 TikZ 生成，不再为技术正文重跑历史数值图表。
v19 构建 11416385 已完成，htc/preempt，退出 0，33 秒，13 页。
v20 新增符号定义页；最终排版构建 11416648 已完成，htc/preempt，退出 0，33 秒。
v21 补充图中 $a,c,b,d$ 与代码对象的逐项对应，以及当前正定-$C$ mixed gauge；
v22 补充 mixed gauge 下 \(\ell_T,r_T,\ell_0,r_0\) 的固定点方程与代码对应；
最终排版构建 11418715 已完成，htc/preempt，退出 0，12 秒，15 页。
没有 overfull、缺字或未定义引用，build/SHA256SUMS_11418715 校验通过。
14 页，无 overfull、缺字或未定义引用；第 3 页两张图与说明已目视检查。
build/SHA256SUMS_11416648 的 compute/login 校验均通过。
初次构建 11416380 的 TikZ 样式名冲突已经修复；
11416381 为第一份通过的草稿，最终版增加了准确的残差与 replica 符号定义。
