# Experiment Runner

> **Two-layer design:** This directory contains the **user-facing CLI and JSON configs**. The underlying simulation logic lives in [`src/sim/experiments/`](../src/sim/experiments/) — edit there to change how experiments run; edit here to change how they are invoked or configured.

The experiment runner provides functionality for running batch simulations with multiple configurations and seeds, exporting summary metrics to CSV files.

## Features

- **JSON Configuration**: Define experiment configurations in JSON format
- **Multiple Seeds**: Run each configuration with multiple random seeds for statistical robustness
- **Summary Metrics**: Calculate and export key economic indicators:
  - Average market price across rounds
  - Average Herfindahl-Hirschman Index (HHI) for market concentration
  - Average consumer surplus
  - Total and per-firm profits
- **CSV Export**: Results saved to timestamped CSV files in `/artifacts/`
- **Policy Support**: Include policy shocks (taxes, subsidies, price caps) in experiments
- **Segmented Demand**: Support for multi-segment consumer demand models

## Usage

### Command Line Interface

```bash
# Run experiments with 3 seeds per configuration
python experiments/cli.py experiments/sample_config.json --seeds 3

# Use custom artifacts directory and database
python experiments/cli.py experiments/sample_config.json --seeds 5 --artifacts results/ --db data/my_experiments.db

# Enable verbose output
python experiments/cli.py experiments/sample_config.json --seeds 3 --verbose
```

### Python API

```python
from sim.experiments.runner import ExperimentRunner, ExperimentConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup database
engine = create_engine("sqlite:///data/experiments.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

# Create experiment configurations
configs = [
    ExperimentConfig(
        config_id="baseline",
        model="cournot",
        rounds=10,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
    ),
    ExperimentConfig(
        config_id="with_policy",
        model="cournot", 
        rounds=10,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
        policies=[
            {"round_idx": 2, "policy_type": "TAX", "value": 0.1}
        ],
    ),
]

# Run experiments
runner = ExperimentRunner("artifacts")
csv_path = runner.run_experiment_batch(configs, seeds_per_config=3, db=db)
print(f"Results saved to: {csv_path}")
```

## Configuration Format

Experiments are defined in JSON format with the following structure:

```json
[
  {
    "config_id": "baseline_cournot",
    "model": "cournot",
    "rounds": 10,
    "params": {
      "a": 100.0,
      "b": 1.0
    },
    "firms": [
      {"cost": 10.0},
      {"cost": 15.0},
      {"cost": 20.0}
    ]
  },
  {
    "config_id": "segmented_bertrand",
    "model": "bertrand",
    "rounds": 10,
    "params": {
      "alpha": 100.0,
      "beta": 1.0
    },
    "firms": [
      {"cost": 10.0},
      {"cost": 15.0}
    ],
    "segments": [
      {"alpha": 200.0, "beta": 1.0, "weight": 0.6},
      {"alpha": 150.0, "beta": 0.8, "weight": 0.4}
    ],
    "policies": [
      {"round_idx": 2, "policy_type": "TAX", "value": 0.1},
      {"round_idx": 5, "policy_type": "SUBSIDY", "value": 2.0}
    ]
  }
]
```

### Configuration Fields

- **config_id**: Unique identifier for this configuration
- **model**: Simulation model ("cournot" or "bertrand")
- **rounds**: Number of simulation rounds
- **params**: Market parameters
  - For Cournot: `a` (price intercept), `b` (slope)
  - For Bertrand: `alpha` (demand intercept), `beta` (slope)
- **firms**: List of firm configurations with `cost` (marginal cost)
- **segments** (optional): Segmented demand configuration
  - `alpha`: Segment demand intercept
  - `beta`: Segment demand slope  
  - `weight`: Market weight (must sum to 1.0)
- **policies** (optional): Policy events
  - `round_idx`: Round when policy applies
  - `policy_type`: "TAX", "SUBSIDY", or "PRICE_CAP"
  - `value`: Policy parameter (tax rate, subsidy amount, price cap)

## Output Format

Results are exported to CSV files with the following columns:

- **run_id**: Unique identifier for each simulation run
- **config_id**: Configuration identifier
- **seed**: Random seed used
- **model**: Simulation model
- **rounds**: Number of rounds
- **avg_price**: Average market price across rounds
- **avg_hhi**: Average Herfindahl-Hirschman Index
- **avg_cs**: Average consumer surplus
- **total_profit**: Total profit across all firms and rounds
- **mean_profit_per_firm**: Average profit per firm
- **num_firms**: Number of firms
- **firm_N_profit**: Total profit for firm N across all rounds

## Examples

### Basic Cournot vs Bertrand Comparison

```json
[
  {
    "config_id": "cournot_baseline",
    "model": "cournot",
    "rounds": 10,
    "params": {"a": 100.0, "b": 1.0},
    "firms": [{"cost": 10.0}, {"cost": 15.0}]
  },
  {
    "config_id": "bertrand_baseline", 
    "model": "bertrand",
    "rounds": 10,
    "params": {"alpha": 100.0, "beta": 1.0},
    "firms": [{"cost": 10.0}, {"cost": 15.0}]
  }
]
```

### Policy Impact Analysis

```json
[
  {
    "config_id": "no_policy",
    "model": "cournot",
    "rounds": 10,
    "params": {"a": 100.0, "b": 1.0},
    "firms": [{"cost": 10.0}, {"cost": 15.0}]
  },
  {
    "config_id": "with_tax",
    "model": "cournot",
    "rounds": 10,
    "params": {"a": 100.0, "b": 1.0},
    "firms": [{"cost": 10.0}, {"cost": 15.0}],
    "policies": [
      {"round_idx": 2, "policy_type": "TAX", "value": 0.1}
    ]
  }
]
```

### Segmented Market Analysis

```json
[
  {
    "config_id": "segmented_market",
    "model": "cournot",
    "rounds": 10,
    "params": {"a": 100.0, "b": 1.0},
    "firms": [{"cost": 10.0}, {"cost": 15.0}],
    "segments": [
      {"alpha": 200.0, "beta": 1.0, "weight": 0.6},
      {"alpha": 150.0, "beta": 0.8, "weight": 0.4}
    ]
  }
]
```

## Testing

The experiment runner includes comprehensive tests:

```bash
# Run all experiment tests
python -m pytest tests/unit/experiments/ -v

# Run specific test categories
python -m pytest tests/unit/experiments/test_exp_seeds.py -v
python -m pytest tests/unit/experiments/test_exp_metrics.py -v  
python -m pytest tests/unit/experiments/test_artifacts_path.py -v
```

### Test Categories

- **test_exp_seeds.py**: Tests seed reproducibility and identical results for same config+seed
- **test_exp_metrics.py**: Tests metrics calculation accuracy and CSV column presence
- **test_artifacts_path.py**: Tests file creation, naming, and CSV format validation

## Demo

Run the demo script to see the experiment runner in action:

```bash
python experiments/demo.py
```

This will load the sample configuration, run experiments, and display summary results.
