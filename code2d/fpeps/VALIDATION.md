# fPEPS S_tilde 验收记录（2026-09-10）

本次新增测量接口、批处理入口、长度/相位/sector 门槛与物理 sewing 对照。
原有 RK-Ising、TFIM 和 Gaussian 源码未修改，VUMPS Project/Manifest 未更新。

**旧 stationary LMPS 未通过物理 benchmark。** 同一 w=0.25 的态在 Gaussian torus 上
每交点 S_tilde 收敛至约 0.0164301443，而当前 χ=8 LMPS 为 0.0073185220。
下表记录的 w=1 数值仍仅为所构造 LMPS 网络的长度极限，不能称为已验证的物理熵。
具体尺寸扫描、能隙证明及非厄米环境诊断见 [benchmark](benchmark/README.md)。

新有限窗口区域 MPS 的审计见 [当前记录](benchmark/AUDIT_20260910.md)。
L=2,4 不截断时完整物理 sector 与 Gaussian 一致，误差 <5e-15。
在 w=1/8、balanced gauge 下，L=8/12/16、χ=16 的熵误差 <3.5e-9，
全部归一化 sector 相对误差 <1.2e-7；L=12 增至 χ=24 后熵变化 <1.6e-9。
大窗口酉虚腿 gauge 的熵变化 <2e-15。此精度结论限于这组参数和 regulator；
非酉 outgoing gauge 与 w=1/4 的未达标点单独记录，没有放宽 sector 门槛。

- 50 项旋转、复杂虚腿 gauge、非 Gaussian 随机态和直积态测试通过（11213374）。
- 11 项公开 API / balanced 构造器对照通过（11213532），包括原始零 2×2 闭合的拒绝。
- 25 项矩形不等宽区域对照通过（11213431），包括原始临界 w=1 的非零 3×2 PEPS。
- 15 项弱键 CAR / BCS / typed PEPS 对照通过（11213399），包含新 w=1/8 点。
- 旧 fPEPS seam / six-R / measurement 回归和 22 个旧文件哈希检查通过（11213380）。
- 最新默认 CLI 的完整 4×4 Gaussian benchmark 通过（11213527），误差 <1.2e-14。
- w=1/8 的 OBC 与 APBC torus/4 极限在 L=24 相差 <1.5e-13（11213544）。

## 固定 χ 的结果

模型为严格 D=2 KSVC Q projector；观测量使用 ABC 区域 Fock 顺序及普通 occupation replica permutation，
自然对数。默认 A/B 采用同一个 north 固定点分支，各自使用相应左/右 rail；C 单独求 south。
每个 sector 都保留复数值、16-term parity 求和的实际抵消及有限传播的 logscale。

| even/odd χ | 总 χ | 最大 seam 长度 | S_tilde | 最后三点最大漂移 | 作业 / 数据 |
|---|---:|---:|---:|---:|---|
| (1,1) | 2 | 64 | 0.080210591464521 | 8.35e-10 | 11212794 / [chi2](data/stilde_chi2_final/run.toml) |
| (2,2) | 4 | 64 | 0.110039435718278 | 6.27e-9 | 11212795 / [chi4](data/stilde_chi4_final/run.toml) |
| (4,4) | 8 | 128 | 0.113166530039507 | 2.01e-10 | 11212792 / [chi8](data/stilde_chi8_final/run.toml) |

三个点均通过固定 χ 的长度门槛；五个 log|Z| 的线性尾部残差均 <1e-7。
最终相位误差分别约 5.93e-12、3.31e-9、7.50e-8，加权端点残差均 <4e-14，
最大 parity 抵消条件数约 2。三个 run.toml 的源码哈希对应当时的实现；
后续双侧 observable 和有限区域修正已改变当前 src，历史 checkpoint 未改写。

`11212804` 从独立 seed=1733 重新求总 χ=4 的环境和 cap，长度验收再次通过，
S_tilde 与 seed=1729 的结果相差 <1e-10；记录见 [独立种子复算](data/stilde_chi4_seedcheck/run.toml)。

总 χ=8 在默认长度 (8,16,32,64) 下最后三点漂移约 1.22e-6，被正确拒绝；
重载相同环境并延长至 (16,32,64,128) 后通过。总 χ=2 的 seed=1729 在 AC 接缝有简并主根，
被拒绝；seed=1730 的单副本谱通过后才进入熵测量。两种失败及重试均保留在日志/元数据中。

这是有限 χ 的 LMPS 近似及其长度极限，不是 χ→∞ 外推。
尤其三个点不足以确定临界增长律，不能把任意 Gaussian 有限分区的大小直接等同于 χ。
χ=8 的相位误差接近当前 1e-7 门槛；若继续延长 seam，应先提高边界求解精度。

## 独立检查

- `11212776`：`physical_sewing_probe.jl`，59 项通过。独立完整 ket 与区域双层 rails 逐 sector 比较，
  覆盖三模式三角网络、非 Gaussian 六模式态、严格和变形 KSVC 3×2 patch，
  以及各区域 parity block 内一般酉 gauge 变换。最大误差约 1e-14。
- `11212786`：`sixr_probe.jl` 通过。五种 sewing 在长度 0、1、2 的 full/cycle 对照；
  全部 16 个四副本 parity term 逐项对照；gauge 检查。差异约 1e-15。
- `11212787`：`local_probe.jl` 通过。局域 CAR/Fock 投影与 typed PEPS 系数一致；
  五个 sector 与独立 Gaussian 参考一致，S_tilde 差异 <6e-15。零范数严格 2×2 patch 不归一化。
- `11212769`：原 `lmps/tests/runtests.jl`，129 项通过。
- `11212799`：原 `lmps/vumps/test/runtests.jl`，50 项通过，覆盖 RK adapter、TFIM 单位胞/方向构造和 LTR 索引。
- `11212796`：`measurement_probe.jl`，16 项通过，覆盖相位/抵消拒绝、product 环境、长度门槛及增量传播。
- `test/check_legacy_sources.py`：22 个已记录的旧源码和环境文件全部保持原哈希。

旧功能回归的覆盖范围以上述测试为准，没有重新运行完整 TFIM 优化 campaign 或 GPU 扫描。

## 复现

参见 [README](README.md) 的命令和 Julia API。所有 Julia 作业均运行在 Slurm 计算节点。
生成数据保留 `prepared.jls`、A/B/C 环境、逐长度的原始复数 sector 及 `depth_scan.csv`。
详细符号与数值约定见 [实现补充](../../notes/fpeps_signs_zh/implementation.md)。
