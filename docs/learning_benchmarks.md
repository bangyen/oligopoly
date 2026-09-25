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
  not evidence of learned reward-punishment collusion; see the long-horizon
  experiments below.

The fast regression test `tests/benchmarks/test_learning_benchmark.py` pins the
convergence of fictitious play so it cannot silently regress.

## Long-horizon Q-learning (algorithmic collusion)

`sim.experiments.algorithmic_collusion` follows the design of Calvano,
Calzolari, Denicolò & Pastorello (2020), *Artificial Intelligence, Algorithmic
Pricing, and Collusion* (AER), on every market in this repo. Each of two
symmetric firms runs tabular Q-learning over a 15-point action grid spanning the
Nash and joint-monopoly actions. Its state is both firms' previous actions
(one-period memory), so it *can* learn to punish. Exploration decays slowly
(ε = e^(-βt)), and a session ends once neither firm's greedy strategy has changed
for 100,000 periods. Runs bypass the database (millions of periods each).
Regenerate with:

```bash
python -m scripts.long_horizon_learning          # 10 seeds, ~6 minutes
python -m scripts.long_horizon_learning --quick  # 2 seeds, faster decay
```

After convergence, firm 0 is forced to play its static best response for one
period and play then continues from the learned strategies:

- **Deviation punished**: firm 0 earns less than before over the next five
  periods.
- **Deviation unprofitable**: its discounted profit over the response is below
  what it earns by staying on the collusive path.
- **Back on path**: play returns to the pre-deviation cycle.
- **Memoryless control**: the same learners with no state, which cannot
  condition on a rival's past price and so cannot punish.

Symmetric duopoly (costs [10.0, 10.0]), 15-point action grid, alpha=0.15, delta=0.95, beta=4e-06, convergence after 100,000 unchanged periods.

| Market | Converged | Periods (mean) | Δ (memory) | Δ (memoryless control) | Deviation punished | Deviation unprofitable | Back on path |
|--------|----------:|---------------:|-----------:|-----------------------:|-------------------:|-----------------------:|-------------:|
| Linear Cournot | 10/10 | 2,231,852 | +0.84 ± 0.09 | +0.10 ± 0.20 | 8/10 | 10/10 | 10/10 |
| Linear Bertrand (winner-take-all) | 10/10 | 1,730,483 | +0.89 ± 0.06 | +0.96 ± 0.03 | 10/10 | 9/10 | 10/10 |
| Isoelastic Cournot | 10/10 | 2,361,406 | +0.84 ± 0.07 | +0.06 ± 0.18 | 8/10 | 10/10 | 10/10 |
| Isoelastic Bertrand | 10/10 | 1,878,528 | +0.88 ± 0.04 | +0.97 ± 0.03 | 8/10 | 8/10 | 10/10 |
| CES Bertrand | 10/10 | 2,253,166 | +0.75 ± 0.15 | +0.12 ± 0.18 | 10/10 | 10/10 | 10/10 |

A typical response (CES Bertrand, seed 0): the deviation is met with a price
cut, a short price war, and a return to collusive prices within five periods.

**CES Bertrand**, seed 0: firm 0 deviates in period 1

| Period | Firm 0 action | Firm 1 action | Firm 0 profit |
|-------:|--------------:|--------------:|--------------:|
| 0 | 18.33 | 18.05 | 87.0 |
| 1 | 16.90 | 18.05 | 88.2 |
| 2 | 16.90 | 16.33 | 83.9 |
| 3 | 16.62 | 16.62 | 84.7 |
| 4 | 18.33 | 17.48 | 85.5 |
| 5 | 17.76 | 18.05 | 87.7 |
| 6 | 18.33 | 18.05 | 87.0 |
| 7 | 18.33 | 18.05 | 87.0 |
| 8 | 18.33 | 18.05 | 87.0 |
| 9 | 18.33 | 18.05 | 87.0 |
| 10 | 18.33 | 18.05 | 87.0 |


### Reading the results

- **Cournot and CES Bertrand: learned collusion.** With memory, learners settle
  at Δ ≈ 0.75–0.84. Deviations are punished, unprofitable and followed by a
  return to collusion. Without memory Δ falls to ≈ 0.1. The supra-competitive
  outcome depends on the ability to punish, which is the reward-punishment
  mechanism Calvano et al. report.
- **Homogeneous Bertrand: not attributable to punishment.** Memoryless
  learners reach Δ ≈ 0.96, as high as those with memory. With a constant
  learning rate, the value of undercutting is estimated while the rival still
  explores at random, and exploration dies out before that estimate is
  revisited, so play freezes at high prices. This artifact of Q-learning
  dynamics has been discussed by e.g. Asker, Fershtman & Pakes (2022). Deviation
  tests still show punishment here, but the memoryless control shows it is not
  needed to sustain the prices.
- Results are for symmetric duopolies with one-period memory on a 15-point
  grid, as in the original paper; other designs may differ.
