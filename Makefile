VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv

help:
	@echo "Available commands:"
	@echo "  make install  - Create venv and install dependencies"
	@echo "  make dev      - Start the local development server"
	@echo "  make test     - Run tests/benchmarks"

install:
	@echo "[Makefile] Creating virtual environment and installing dependencies..."
	$(UV) sync
	@echo "[Makefile] Downloading base model and datasets..."
	$(PYTHON) app/download.py

dev:
	@echo "[Makefile] Starting development server..."
	@echo "[Makefile] Backend API (Mock) would be at localhost:8000"
	@echo "[Makefile] Frontend served at http://localhost:3000"
	# Simple static serve for now. In real dev, this might run 'vite' and 'uvicorn' in parallel.
	$(PYTHON) -m http.server 3000 --directory web

test:
	@echo "[Makefile] Running model inference tests..."
	$(PYTHON) app/inference.py
