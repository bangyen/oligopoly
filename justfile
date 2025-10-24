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
    {{PYTHON}} -m black .

# lint code
lint:
    {{PYTHON}} -m ruff check .

# type-check
type:
    {{PYTHON}} -m mypy .

# run tests
test:
    {{PYTHON}} -m pytest

# run all checks (fmt, lint, type, test)
all: fmt lint type test
    echo "All checks completed!"

# build and start services
docker:
    docker build -f deployment/Dockerfile -t oligopoly:latest .
    cd deployment && docker compose up -d

# stop API server only
api-stop:
    echo "Stopping API server..."
    cd deployment && docker compose down

# stop all services
docker-stop:
    cd deployment && docker compose down

# remove all containers, images, and volumes
docker-clean:
    cd deployment && docker compose down -v --remove-orphans
    docker rmi oligopoly:latest deployment-app 2>/dev/null || true

# run experiments demo
experiments:
    {{PYTHON}} experiments/demo.py

# start API server only
api:
    #!/usr/bin/env bash
    echo "Starting API server..."
    cd deployment && docker compose up -d
    echo "Waiting for API to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
            echo "API is ready!"
            break
        fi
        echo "Waiting for API... ($timeout seconds remaining)"
        sleep 2
        timeout=$((timeout-2))
    done
    if [ $timeout -le 0 ]; then
        echo "ERROR: API failed to start within 60 seconds"
        echo "Check Docker logs with: cd deployment && docker compose logs app"
        exit 1
    fi

# start Flask dashboard
dashboard:
    echo "Starting dashboard..."
    echo "Dashboard will be available at http://localhost:5050"
    {{PYTHON}} dashboard/main.py

