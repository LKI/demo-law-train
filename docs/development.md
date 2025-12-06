# Development Guide

## Overview
The `demo-law-train` project provides a local development environment to test the trained model using a chatbot interface. It typically uses a FastAPI backend that serves both the API and the static frontend files.

## Getting Started

### Prerequisites
Ensure you have installed the project dependencies:
```bash
make install
```

### Running the Dev Server
To start the local server:
```bash
make dev
```
This command runs `uvicorn` with hot-reloading enabled.
- **URL**: [http://localhost:8000](http://localhost:8000)
- **API Endpoint**: `POST /api/chat`

## Architecture

### Backend (`app/`)
- **`server.py`**: The entry point for the FastAPI application. It mounts the `web/` directory for static files and defines the `/api/chat` endpoint.
- **`inference.py`**: Contains the model loading logic (`get_model_and_tokenizer`) and the streaming generation logic (`stream_response`). It uses `TextIteratorStreamer` to yield tokens as they are generated.

### Frontend (`web/`)
- **`index.html`**: The main structure of the chat interface.
- **`style.css`**: Styling for the application.
- **`app.js`**: Handles user input, sends requests to the backend, and processes the streaming response using the `Fetch API` and `TextDecoder`.

## Troubleshooting
- **Model Not Found**: If you see an error about the model not being found, ensure you have run `make install` to download the base model and datasets.
- **Port In Use**: If port 8000 is occupied, you can modify the port in the `Makefile` or `app/server.py`.

## Benchmarking

To evaluate the model's performance on the DISC-Law-SFT dataset:
```bash
make benchmark
```
This runs the `app/benchmark.py` script, which:
1. Loads the model and dataset.
2. Generates responses for a subset of the data (default 50 samples).
3. Computes the **Rouge-L** score.
4. Saves detailed results to `benchmark_results.jsonl`.

You can configure the benchmark by modifying `app/benchmark.py` or passing arguments (if supported).
