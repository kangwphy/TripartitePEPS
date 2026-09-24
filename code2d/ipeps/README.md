# RK-Ising CTMRG reference

Main entries: `rk_ctmrg.jl`, `split_replica_ctmrg.jl`, `measure_xi_split.jl`.
Shared contractions are in `yvx_core.jl`; its small boundary-MPS support file
is now local in `support/rk_bmps.jl`. The finite-PEPS package is not required.

Data: `../../data/rk_ising/ctmrg/`. Comparison scripts use the consolidated
MC data under `../../data/rk_ising/mc/`. Historical plots/collectors are retained
but do not constitute new validation of their input points.

Use the pinned parent `code2d/Project.toml` environment and `../run_preempt.sh`.
Always submit computation/plotting with `--partition=preempt`.
