# System Architecture

## Overview
The system consists of a Python-based SFT (Supervised Fine-Tuning) pipeline and a web-based local inference playground.

## Data Flow

```mermaid
graph TD
    Data[Dataset (HuggingFace)] --> |Download| LocalData[app/data]
    BaseModel[Qwen2.5-1.5B] --> |Download| LocalModel[app/models]

    subgraph "Training Pipeline (Python)"
        LocalData --> SFT[SFT Trainer]
        LocalModel --> SFT
        SFT --> LoRA[LoRA Adapters]
    end

    subgraph "Inference Engine"
        LoRA --> Merger[Model Merger]
        Merger --> Serving[Inference API]
    end

    subgraph "User Interface (Web)"
        User[User] --> |Chat| WebApp[Web Frontend]
        WebApp --> |REST/WS| Serving
    end
```

## Components

### 1. Training Application (`app/`)
- **Manager**: `uv`
- **Responsibility**: Data processing, Model Fine-tuning, Evaluation.
- **Key Modules**:
    - `app/train.py`: Main entry point for SFT.
    - `app/download.py`: Handles HuggingFace downloads.
    - `app/inference.py`: CLI or API for model testing.

### 2. Web Frontend (`web/`)
- **Tech**: Standard Web (HTML/TS) or Micro-framework.
- **Responsibility**: Provide a chat interface for the user to verify model performance locally.
- **Interaction**: Calls the `app` inference API.

## Directory Structure
- `/app`: Backend source code.
- `/web`: Frontend source code.
- `/docs`: Documentation.
- `Makefile`: Automation scripts.
