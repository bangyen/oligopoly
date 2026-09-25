# Scripts

| Script | Purpose |
|--------|---------|
| `learning_benchmark.py` | Short-run convergence of each learning strategy in every market (`python -m scripts.learning_benchmark [--quick]`) |
| `long_horizon_learning.py` | Calvano-style Q-learning with deviation tests and a memoryless control (`python -m scripts.long_horizon_learning [--quick]`) |
| `ci/check_wheel.py` | CI check that the built wheel is self-contained (`--core` for an install without extras) |

Results from both experiment scripts are written up in
[`docs/learning_benchmarks.md`](../docs/learning_benchmarks.md).
