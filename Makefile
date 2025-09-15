.PHONY: init fmt lint type test all
init: ## install tooling
	python -m pip install -U pip
	pip install black ruff mypy pytest

fmt:  ## format code
	black .

lint: ## lint code
	ruff check .

type: ## type-check
	mypy .

test: ## run tests
	pytest

all: fmt lint type test