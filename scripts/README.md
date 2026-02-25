# Scripts

Demo and utility scripts for the oligopoly simulation. All scripts must be
run from the **repo root** using `python -m scripts.<name>` (the `scripts/`
folder is a package).

## Quick Reference

| Script | Purpose |
|--------|---------|
| `strategy_demo.py` | **Start here.** Demonstrates the three built-in strategies (Static, TitForTat, RandomWalk) in Cournot and Bertrand models. |
| `policy_demo.py` | Shows how tax, subsidy, and price-cap policy shocks affect simulation outcomes round-by-round. |
| `collusion_demo.py` | Demonstrates cartel formation, defection detection, regulator intervention, and event logging. |
| `epsilon_greedy_demo.py` | Runs ε-greedy Q-learning agents over 20 rounds and compares them to static/TitForTat baselines. |
| `enhanced_economics_demo.py` | Shows capacity constraints, fixed costs, economies of scale, and isoelastic demand in the enhanced simulation models. |
| `advanced_economics_demo.py` | Demonstrates Fictitious Play, Deep Q-Learning, and Behavioral strategies; product differentiation; and advanced API features. |
| `segmented_demand_demo.py` | Shows how to model markets with multiple consumer segments of different price elasticities. |
| `utils.py` | Shared helpers (formatting, DB setup, HHI/CS calculations) used internally by the other scripts — not a runnable demo. |

## Usage

```bash
# Recommended first run
python -m scripts.strategy_demo

# Policy shock walkthrough
python -m scripts.policy_demo

# Full collusion + regulator dynamics
python -m scripts.collusion_demo
```

> **Note:** `policy_demo.py` and `epsilon_greedy_demo.py` write a temporary
> SQLite database to `data/demo.db`. The `data/` directory will be created
> automatically if it does not exist.
