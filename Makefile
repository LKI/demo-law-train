VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv

help:
	@echo "Available commands:"
	@echo "  make install  - Create venv and install dependencies"
	@echo "  make dev      - Start the local development server"
	@echo "  make test     - Run tests/benchmarks (Quick)"
	@echo "  make benchmark- Run full benchmark on SFT dataset"
	@echo "  make fmt      - Format code using ruff"

benchmark:
	@echo "[Makefile] Running benchmark..."
	$(UV) run python app/benchmark.py


install:
	@echo "[Makefile] Creating virtual environment and installing dependencies..."
	$(UV) sync
	@echo "[Makefile] Downloading base model and datasets..."
	$(PYTHON) app/download.py

fmt:
	@echo "[Makefile] Formatting code..."
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

dev:
	@echo "[Makefile] Starting development server..."
	@echo "[Makefile] Serving at http://localhost:8234"
	$(UV) run uvicorn app.server:app --reload --host 0.0.0.0 --port 8234

test:
	@echo "[Makefile] Running model inference tests..."
	$(PYTHON) app/inference.py
