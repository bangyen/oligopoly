.PHONY: init fmt lint type test all docker docker-stop docker-clean experiments api api-stop dashboard

init: ## install tooling
	python -m pip install -U pip
	pip install -e ".[dev]"

fmt:  ## format code
	black .

lint: ## lint code
	ruff check .

type: ## type-check
	mypy .

test: ## run tests
	python -m pytest

all: fmt lint type test

# Docker targets
docker: ## build and start services
	docker build -f deployment/Dockerfile -t oligopoly:latest .
	cd deployment && docker compose up -d

api-stop: ## stop API server only
	@echo "Stopping API server..."
	cd deployment && docker compose down

docker-stop: ## stop all services
	cd deployment && docker compose down

docker-clean: ## remove all containers, images, and volumes
	cd deployment && docker compose down -v --remove-orphans
	docker rmi oligopoly:latest deployment-app 2>/dev/null || true

# Tool targets
experiments: ## run experiments demo
	python experiments/demo.py

api: ## start API server only
	@echo "Starting API server..."
	cd deployment && docker compose up -d
	@echo "Waiting for API to be ready..."
	@timeout=60; \
	while [ $$timeout -gt 0 ]; do \
		if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then \
			echo "API is ready!"; \
			break; \
		fi; \
		echo "Waiting for API... ($$timeout seconds remaining)"; \
		sleep 2; \
		timeout=$$((timeout-2)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "ERROR: API failed to start within 60 seconds"; \
		echo "Check Docker logs with: cd deployment && docker compose logs app"; \
		exit 1; \
	fi

dashboard: ## start streamlit dashboard (starts API if needed)
	@echo "Checking if API is running..."
	@if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then \
		echo "API is already running"; \
	else \
		echo "API not running, starting Docker services..."; \
		$(MAKE) api; \
	fi
	@echo "Starting dashboard..."
	streamlit run scripts/dashboard.py