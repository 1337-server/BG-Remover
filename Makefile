PYTHON ?= python3
VENV ?= .venv
RUNTIME ?= cli
IMAGE ?= br-remover-$(RUNTIME)

.PHONY: venv format lint test run-cli run-flask build-docker clean

venv:
	$(PYTHON) -m venv $(VENV)

format:
	ruff format .

lint:
	ruff check .

test:
	pytest -q

run-cli:
	python -m runtimes.cli.bgr_cli --help

run-flask:
python -m runtimes.flask.app

build-docker:
	docker build --build-arg RUNTIME=$(RUNTIME) -t $(IMAGE) .

clean:
	rm -rf build dist .pytest_cache .ruff_cache **/__pycache__
