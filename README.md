# Oligopoly Simulation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/oligopoly/blob/main/oligopoly_demo.ipynb)
[![CI](https://github.com/bangyen/oligopoly/actions/workflows/ci.yml/badge.svg)](https://github.com/bangyen/oligopoly/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/bangyen/oligopoly)](LICENSE)

**Oligopoly market simulation: Cournot and Bertrand competition with learning strategies, collusion detection, and policy shocks**  

<p align="center">
  <img src="docs/cournot_heatmap.png" alt="Oligopoly Dashboard" width="600">
</p>

## Quickstart

Clone the repo and initialize the environment:

```bash
git clone https://github.com/bangyen/oligopoly.git
cd oligopoly
just init      # or: pip install -e ".[dev]"
just test      # or: pytest
python -m scripts.strategy_demo
```

Or open in Colab: [Colab Notebook](https://colab.research.google.com/github/bangyen/oligopoly/blob/main/oligopoly_demo.ipynb).

Once installed, the package provides three console scripts:

```bash
oligopoly     # serve the REST API (http://localhost:8000)
cournot       # run a one-off Cournot simulation
bertrand      # run a one-off Bertrand simulation
```

## Results

| Capability | Description |
|------------|-------------|
| Collusion Detection | Flags cartel behavior and defections with a configurable tolerance |
| Cournot Equilibrium | Matches the textbook closed form to 1e-6 ([benchmarks](tests/benchmarks/)) |
| Strategy Adaptation | Firms learn and evolve using Q-learning and Fictitious Play |

The Bertrand "equilibrium" helper (`bertrand_nash_equilibrium`) is a
capacity-constrained markup heuristic, not the textbook price-equals-cost
Bertrand–Nash outcome. See also the known Cournot engine issue recorded in
`tests/benchmarks/test_cournot_benchmark.py`.

## Features

- **Collusion Detection** — Tolerance-based detection of cartel behavior and defections.  
- **Policy Analysis** — Applies taxes, subsidies and price caps mid-simulation.  
- **Learning Strategies** — Supports Q-learning, Fictitious Play, and Tit-for-Tat algorithms.  
- **Interactive Dashboard** — Visualization using FastAPI and Jinja2 templates.  
- **REST API** — Comprehensive FastAPI endpoints for simulation management and analysis.  
- **Batch Experiments** — Reproducible seeded runs with CSV export (plots need `pip install -e ".[viz]"`).  

## Repo Structure

```plaintext
oligopoly/
├── dashboard/           # FastAPI visualization dashboard
├── experiments/         # Batch experiment configurations
├── scripts/             # Demo and utility scripts
├── src/                 # Core implementation
│   ├── sim/             # Simulation engine
│   │   ├── games/       # Cournot & Bertrand models
│   │   ├── strategies/  # Learning algorithms
│   │   ├── policy/      # Tax/subsidy interventions
│   │   └── api/         # FastAPI app: schemas + simulate/runs/heatmap routers
│   └── ...
├── tests/               # Unit, integration and analytic benchmark tests
└── oligopoly_demo.ipynb # Colab notebook demo
```

## Test Locations

Tests mirror the source tree under `tests/unit/`. Non-obvious mappings:

| Source | Tests |
|--------|-------|
| `src/sim/games/` | `tests/unit/games/` |
| `src/sim/strategies/` | `tests/unit/strategies/` |
| `src/sim/policy/` | `tests/unit/policy/` |
| `src/sim/collusion.py` | `tests/unit/runners/` |
| `src/sim/runners/` | `tests/unit/runners/` |
| `src/sim/api/` | `tests/unit/api/` + `tests/integration/` |
| Analytic equilibria | `tests/benchmarks/` |
| `dashboard/main.py` | `tests/unit/heatmap/` + `tests/unit/infrastructure/` |

## Validation

- ✅ Test coverage ≥85%, enforced in CI (`just cov`)
- ✅ Cournot results checked against closed-form equilibria (`tests/benchmarks/`)
- ✅ CI on Python 3.10, 3.11 and 3.12
- ✅ Reproducible seeds for experiments
- ✅ `justfile` for common development tasks

## API Endpoints

- `GET /` - Root API information
- `POST /simulate` - Run Cournot/Bertrand simulation
- `GET /runs` - List simulation runs
- `GET /runs/{run_id}` - Get simulation time-series results
- `GET /runs/{run_id}/detail` - Get detailed run metadata
- `GET /runs/{run_id}/events` - Retrieve all simulation events
- `GET /runs/{run_id}/replay` - Get frame-by-frame replay data
- `POST /compare` - Run scenarios for comparison
- `GET /compare/{left_run_id}/{right_run_id}` - Get aligned comparison results
- `POST /heatmap` - Generate profit surface heatmaps
- `GET /healthz` - Health check endpoint

## References

- Cournot, A. (1838). *[Recherches sur les principes mathématiques de la théorie des richesses](https://gallica.bnf.fr/ark:/12148/bpt6k6117257c)*
- Bertrand, J. (1883). *[Théorie mathématique de la richesse sociale](https://en.wikipedia.org/wiki/Bertrand_competition)*
- Nash, J. (1950). *[Equilibrium points in n-person games](https://www.pnas.org/doi/10.1073/pnas.36.1.48)*

## License

This project is licensed under the [MIT License](LICENSE).
