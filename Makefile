.PHONY: init fmt lint type test all docker docker-stop docker-clean experiments dashboard

init: ## install tooling
	python -m pip install -U pip
	pip install black ruff mypy pytest
	pip install -e .

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

docker-stop: ## stop services
	cd deployment && docker compose down

docker-clean: ## remove all containers, images, and volumes
	cd deployment && docker compose down -v --remove-orphans
	docker rmi oligopoly:latest deployment-app 2>/dev/null || true

# Tool targets
experiments: ## run experiments demo
	python experiments/demo.py

dashboard: ## start streamlit dashboard (starts API if needed)
	@echo "Checking if API is running..."
	@curl -s http://localhost:8000/ > /dev/null 2>&1 || (echo "API not running, starting Docker services..." && $(MAKE) docker)
	@echo "Starting dashboard..."
	streamlit run scripts/dashboard.py