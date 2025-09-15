.PHONY: help init fmt lint type test all docker-build docker-up docker-down docker-logs docker-shell docker-clean
help: ## show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

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
docker-build: ## build Docker image
	docker build -f deployment/Dockerfile -t oligopoly:latest .

docker-up: ## start services with docker-compose
	cd deployment && docker compose up -d

docker-down: ## stop services with docker-compose
	cd deployment && docker compose down

docker-logs: ## show logs from running containers
	cd deployment && docker compose logs -f

docker-shell: ## open shell in running app container
	cd deployment && docker compose exec app /bin/bash

docker-clean: ## remove all containers, images, and volumes
	cd deployment && docker compose down -v --remove-orphans
	docker rmi oligopoly:latest deployment-app 2>/dev/null || true