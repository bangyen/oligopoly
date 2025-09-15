.PHONY: init fmt lint type test all docker docker-stop docker-clean collusion-demo epsilon-demo policy-demo segmented-demo strategy-demo experiments dashboard

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

# Individual script targets
collusion-demo: ## run collusion and regulator dynamics demo
	python scripts/collusion_demo.py

epsilon-demo: ## run epsilon-greedy learning agents demo
	python scripts/epsilon_greedy_demo.py

policy-demo: ## run policy shocks demonstration
	python scripts/policy_demo.py

segmented-demo: ## run segmented demand markets demo
	python scripts/segmented_demand_demo.py

strategy-demo: ## run strategy comparison demo
	python scripts/strategy_demo.py

experiments: ## run experiments demo
	python experiments/demo.py

dashboard: ## start streamlit dashboard (requires API running)
	streamlit run scripts/dashboard.py