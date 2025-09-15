# Oligopoly Simulation

A minimal FastAPI + SQLAlchemy + Docker Compose repository for simulating oligopoly market competition.

## Features

- **FastAPI** web framework with automatic API documentation
- **SQLAlchemy** ORM with PostgreSQL database
- **Alembic** for database migrations
- **Docker Compose** for easy development setup
- **Economic models** for oligopoly market simulation

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local development)

### Running the Application

1. **Start the services:**
   ```bash
   docker compose up --build
   ```

2. **Access the application:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/healthz

### Running Tests

```bash
pytest -q
```

## Project Structure

```
oligopoly/
├── src/
│   ├── sim/
│   │   ├── __init__.py
│   │   └── models.py          # Economic models (Demand, Market, Firm, RunConfig)
│   └── main.py               # FastAPI application
├── tests/
│   └── unit/
│       ├── test_health.py    # Health endpoint tests
│       ├── test_models.py    # Model tests
│       └── test_env.py       # Environment configuration tests
├── alembic/                  # Database migrations
├── docker-compose.yml        # Docker services configuration
├── Dockerfile               # Application container
└── pyproject.toml           # Project configuration
```

## Economic Models

### Demand Curve
Linear inverse demand function: `P(Q) = a - b*Q`

- `a`: Maximum price when quantity is zero
- `b`: Slope of demand curve (price sensitivity to quantity)

### Market Structure
- **Market**: Contains demand parameters and firm configuration
- **Firm**: Individual firms with cost structures
- **RunConfig**: Simulation parameters and convergence criteria

## API Endpoints

- `GET /healthz` - Health check endpoint
- `GET /` - Basic API information
- `GET /docs` - Interactive API documentation

## Development

### Local Development Setup

1. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Start PostgreSQL:**
   ```bash
   docker compose up db
   ```

3. **Run migrations:**
   ```bash
   alembic upgrade head
   ```

4. **Start the application:**
   ```bash
   uvicorn src.main:app --reload
   ```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

## Testing

The project includes comprehensive tests for:

- **Health endpoint**: Verifies `/healthz` returns `{"ok": true}`
- **Economic models**: Tests `Demand` class with integer/float values and stable `repr()`
- **Environment**: Tests `DATABASE_URL` environment variable handling and Alembic migrations

Run tests with:
```bash
pytest -q
```

## License

MIT License
