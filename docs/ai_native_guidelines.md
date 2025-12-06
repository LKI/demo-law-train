# AI Native Development Guidelines

## Philosophy
This project is designed to be "AI Native", meaning the codebase itself is optimized for interpretation and modification by AI agents (like yourself). The primary goal is to minimize context loss and maximize autonomy.

## Rules of Engagement

### 1. Context Preservation
- **Self-Contained Files**: Avoid circular dependencies. Keep modules loosely coupled.
- **Top-Level Comments**: Every file MUST start with a top-level docstring explaining its purpose, inputs, and outputs.
- **Explicit Types**: Use strict typing (Python `tiwari` / TypeScript interfaces) to let agents infer structures without reading implementation details.

### 2. Documentation First
- **Architecture as Code**: `docs/architecture.md` is the source of truth. If the code deviates, update the doc first.
- **Decision Records**: Major architectural decisions should be recorded in `docs/decisions/` (if complex) or strictly commented.

### 3. Agent-Friendly Tooling
- **Makefile**: The `Makefile` is the entry point for ALL operations. Agents should not need to guess commands.
    - `make install`: Sets up everything.
    - `make dev`: Starts the dev server.
    - `make test`: Runs validation.
- **Mock First**: When implementing new features, create the interface and mock the implementation first. This allows agents to verify the "glue" code before writing complex logic.

### 4. Code Structure
- **Backend**: `app/` (Python)
- **Frontend**: `web/` (TypeScript/Web)
- **Docs**: `docs/`

## Workflow for Agents
1. **Read `task.md`** to understand the current objective.
2. **Check `docs/`** for architectural constraints.
3. **Plan** changes using `implementation_plan.md`.
4. **Execute** using atomic commits (or tool calls).
5. **Verify** using `make test`.
