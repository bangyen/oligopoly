# Oligopoly Simulation

A comprehensive platform for simulating oligopoly market competition with advanced economic models, learning strategies, and policy analysis capabilities.

## Features

- **Cournot & Bertrand Competition**: Quantity and price-based competition models
- **Advanced Learning**: Fictitious Play, Deep Q-Learning, Nash strategies
- **Collusion Analysis**: Detection, punishment mechanisms, regulatory intervention
- **Policy Shocks**: Taxes, subsidies, price caps, market interventions
- **Enhanced Models**: Capacity constraints, fixed costs, product differentiation
- **Batch Experiments**: Statistical analysis with multiple configurations and seeds
- **Interactive Dashboard**: Streamlit-based visualization and analysis
- **FastAPI + SQLAlchemy**: Modern web framework with database support

## Quick Start

### Docker (Recommended)
```bash
# Start all services
docker compose up --build

# Access the application
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Dashboard: streamlit run scripts/dashboard.py
```

### Local Development
```bash
# Install dependencies
pip install -e ".[dev]"

# Start database
docker compose up db

# Run migrations
alembic upgrade head

# Start application
uvicorn src.main:app --reload
```

## Usage Examples

### Basic Simulation
```python
from sim.games.cournot import cournot_simulation
from sim.models.models import Demand, Firm

# Define market and firms
demand = Demand(a=100.0, b=1.0)
firms = [Firm(cost=10.0), Firm(cost=15.0)]

# Run simulation
result = cournot_simulation(demand, firms)
print(f"Price: {result.price:.2f}, Profits: {result.profits}")
```

### Learning Strategies
```python
from sim.strategies.advanced_strategies import FictitiousPlayStrategy
from sim.runners.strategy_runner import run_strategy_game

strategies = [FictitiousPlayStrategy(learning_rate=0.1) for _ in range(2)]
result = run_strategy_game(
    model="cournot", rounds=50, strategies=strategies,
    costs=[10.0, 15.0], params={"a": 100.0, "b": 1.0},
    bounds=(0.0, 50.0), db=db
)
```

### Batch Experiments
```bash
# Run experiments with multiple seeds
python experiments/cli.py experiments/sample_config.json --seeds 5
```

## Project Structure

```
src/sim/
├── games/           # Cournot, Bertrand, enhanced simulation
├── strategies/      # Nash, learning, collusion strategies  
├── models/          # Economic models, demand functions
├── runners/         # Simulation orchestrators
├── policy/          # Policy shocks and interventions
├── experiments/     # Batch experiment runner
└── cli/             # Command-line interfaces
```

## Development

```bash
# Code quality checks
make all

# Run tests
python -m pytest

# Format code
black .

# Demo scripts
python scripts/strategy_demo.py
python scripts/collusion_demo.py
```

## API Endpoints

- `GET /healthz` - Health check
- `POST /simulate` - Run single simulation
- `GET /heatmap/cournot` - Generate competition heatmaps
- `POST /analyze/collusion` - Analyze collusion patterns

## Research Applications

- **Academic Research**: Test oligopoly theory predictions
- **Policy Analysis**: Evaluate regulatory interventions
- **Market Analysis**: Understand competitive dynamics
- **Strategy Development**: Test competitive strategies

## License

MIT License
