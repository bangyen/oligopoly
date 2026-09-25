# Learning Benchmarks

Do the learning strategies find the equilibrium? Each row is a symmetric
duopoly in which **both** firms use the strategy, played through the real
multi-round runner. Regenerate with:

```bash
python -m scripts.learning_benchmark          # ~3 minutes
python -m scripts.learning_benchmark --quick  # ~20 seconds
```

- **Nash gap** — mean |action − Nash action| / Nash action over the final
  rounds. 0% means the firms play the one-shot Nash equilibrium.
- **Collusion index Δ** — (profit − Nash profit) / (joint-monopoly profit −
  Nash profit). 0 is Nash-level profit, 1 is full monopoly, negative is below
  Nash (e.g. excess output or price wars).

## Results

Symmetric duopoly (costs [10.0, 10.0]), all firms using the strategy; 200 rounds, 5 seeds, metrics over the last 20 rounds (mean ± sd across seeds).

| Market | Strategy | Nash gap | Collusion index Δ |
|--------|----------|---------:|------------------:|
| Linear Cournot | `fictitious_play` | 1.0% ± 0.4% | -0.01 ± 0.02 |
| Linear Cournot | `q_learning` | 57.1% ± 5.2% | -3.25 ± 0.61 |
| Linear Cournot | `deep_q_learning` | 47.5% ± 13.0% | -1.39 ± 1.34 |
| Linear Cournot | `behavioral` | 23.0% ± 6.4% | -1.20 ± 0.71 |
| Linear Bertrand (winner-take-all) | `fictitious_play` | 7.2% ± 3.4% | +0.03 ± 0.01 |
| Linear Bertrand (winner-take-all) | `q_learning` | 71.2% ± 10.7% | +0.17 ± 0.05 |
| Linear Bertrand (winner-take-all) | `deep_q_learning` | 267.9% ± 60.9% | +0.57 ± 0.16 |
| Linear Bertrand (winner-take-all) | `behavioral` | 345.7% ± 50.3% | +0.75 ± 0.12 |
| Isoelastic Cournot | `fictitious_play` | 0.7% ± 0.3% | -0.00 ± 0.01 |
| Isoelastic Cournot | `q_learning` | 63.5% ± 5.5% | -1.52 ± 0.28 |
| Isoelastic Cournot | `deep_q_learning` | 43.6% ± 12.0% | -0.59 ± 0.64 |
| Isoelastic Cournot | `behavioral` | 23.8% ± 10.1% | -0.36 ± 0.28 |
| Isoelastic Bertrand | `fictitious_play` | 7.3% ± 4.2% | +0.22 ± 0.12 |
| Isoelastic Bertrand | `q_learning` | 89.0% ± 8.8% | +0.79 ± 0.05 |
| Isoelastic Bertrand | `deep_q_learning` | 76.9% ± 25.7% | +0.58 ± 0.42 |
| Isoelastic Bertrand | `behavioral` | 90.9% ± 10.7% | +0.88 ± 0.09 |
| CES Bertrand | `fictitious_play` | 0.7% ± 0.3% | -0.02 ± 0.05 |
| CES Bertrand | `q_learning` | 73.2% ± 10.1% | -2.47 ± 0.88 |
| CES Bertrand | `deep_q_learning` | 107.7% ± 52.7% | -3.38 ± 1.23 |
| CES Bertrand | `behavioral` | 44.9% ± 19.6% | -1.43 ± 0.78 |

## Reading the results

- **Fictitious play converges.** In both Cournot markets and CES Bertrand it
  ends within ~1% of the Nash actions with Δ ≈ 0. In homogeneous Bertrand it
  stays within ~7%: undercutting moves in discrete steps near marginal cost
  (and with isoelastic demand Δ ≈ +0.2).
- **Q-learning, deep Q-learning and behavioral firms do not converge within
  200 rounds.** Exploration is still large (tabular Q-learning decays ε by
  0.5% per round), so in Cournot and CES they over-produce or misprice and
  earn *less* than Nash (Δ < 0).
- **Homogeneous Bertrand is the exception**: the same learners sustain prices
  well above marginal cost (Δ ≈ 0.2–0.9). Nash profit is zero there, so any
  noisy pricing above cost that splits demand scores as "collusive". This is
  not evidence of learned reward-punishment collusion of the kind reported for
  long-horizon Q-learning (e.g. Calvano et al., 2020); testing that would need
  far longer runs and a deviation-response experiment.

The fast regression test `tests/benchmarks/test_learning_benchmark.py` pins the
convergence of fictitious play so it cannot silently regress.
