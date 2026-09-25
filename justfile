# Task runner for the project

# Auto-detect uv - falls back to plain python if not available
PYTHON := `command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python"`

# install tooling
init:
    #!/usr/bin/env bash
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv sync --extra dev
        uv run pre-commit install
    else
        echo "Using pip..."
        python -m pip install -U pip
        pip install -e ".[dev]"
        pre-commit install
    fi

# format code
fmt:
    {{PYTHON}} -m ruff format .
    {{PYTHON}} -m ruff check --fix .

# check formatting without modifying files
fmt-check:
    {{PYTHON}} -m ruff format --check .

# lint code
lint:
    {{PYTHON}} -m ruff check .

# type-check
type:
    {{PYTHON}} -m mypy .

# run tests
test:
    {{PYTHON}} -m pytest

# run tests with a coverage report (fails under the configured threshold)
cov:
    {{PYTHON}} -m pytest --cov --cov-report=term-missing:skip-covered

# run all checks without modifying files (used by CI)
check: fmt-check lint type cov

# format, then run all checks
all: fmt lint type test
    echo "All checks completed!"

# serve the REST API locally (http://localhost:8000/docs)
api:
    {{PYTHON}} -m uvicorn sim.api:app --reload

# serve the Scenario Lab dashboard (http://localhost:5050)
dashboard:
    {{PYTHON}} dashboard/main.py

# build the Docker image
docker:
    docker build -f deployment/Dockerfile -t oligopoly:latest .

# run the Scenario Lab from the Docker image
docker-dashboard: docker
    docker run --rm -p 5050:5050 oligopoly:latest python dashboard/main.py

# learning benchmarks and long-horizon collusion experiments (minutes)
benchmarks:
    {{PYTHON}} -m scripts.learning_benchmark
    {{PYTHON}} -m scripts.long_horizon_learning
