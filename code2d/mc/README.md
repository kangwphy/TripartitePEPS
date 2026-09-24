# RK-Ising Monte Carlo

Main estimator: `mc_tildeS.jl`; independent benchmark:
`mc_ising_benchmark_local.jl`. Existing environment-variable controls are
unchanged. Exact finite-size validation uses `support/cube.jl`, copied locally
so this directory no longer needs the separate MPS repository.

Campaigns, tables, error analyses and figures are under
`../../data/rk_ising/mc/`. Read each campaign's README for sampling and error
conventions; a completed file alone is not an equilibration certificate.

Use the pinned parent `code2d/Project.toml` and `../run_preempt.sh`; Python
analysis/plotting uses its `--python` option. All computation goes through
Slurm `--partition=preempt`, including short analyses and plots.
