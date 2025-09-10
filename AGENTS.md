# AGENTS.md — How to Work With This Repo Using an AI Agent

This document defines how AI coding agents (and contributors) should operate on this Python repository to stay safe, fast, and correct.

## Project Snapshot
- Language: Python 3.13
- Packaging: `pyproject.toml` (PEP 621)
- Tests: `pytest`
- Notebooks: `main.ipynb`
- Code: `src/`
- RL stack: Stable-Baselines3 PPO under `src/models/RL/`
- Backtesting: `src/backtester/`
- Exchange utils: `src/binance/`

## Repository Layout (key paths)
- RL env: `src/models/RL/env.py` (class `TradingEnv`)
- RL training: `src/models/RL/train.py` (e.g., `train_ppo`)
- RL utils: `src/models/RL/utils.py`
- Backtester: `src/backtester/crypto_broker_env.py`
- Tests: `tests/`

## Critical Safety Rules
- Do not read, modify, or commit configuration/secrets in the following files unless the user explicitly asks:
  - All `.env` files (e.g., `.env`)
  - All `.ini` files (`*.ini`)
- Treat `.env` contents and any credentials as sensitive. Never print them, log them, or include them in code or commits.
- Respect `.gitignore` (already excludes `.env`, `*.ini`, `*.pkl`, build artifacts, and venvs).
- Make the smallest viable change for the task; avoid unrelated refactors.
- Follow the existing code’s style and structure.

## Local Setup
- Python: 3.13
- Install dependencies (editable):
  ```bash
  pip install -e .
  ```
- Optional tools used by the project: `pytest`, `tensorboard`, Jupyter.

## Running Tests
- Run unit tests:
  ```bash
  pytest -q
  ```
- Add tests beside the code they exercise under `tests/` when introducing non-trivial behavior.

## Common Tasks for Agents
- RL work:
  - Environment logic: edit `src/models/RL/env.py` and keep the step/reset/render contract stable.
  - Training pipeline: edit `src/models/RL/train.py` (PPO setup, callbacks, TensorBoard logging).
  - Utilities: extend `src/models/RL/utils.py` for data loading, showcases, and rendering.
- Backtesting or exchange utilities: modify code under `src/backtester/` or `src/binance/` with minimal, focused diffs.
- Documentation: update `README.md` and docstrings when APIs or workflows change.
- Tests: for bug fixes or new features, add or adjust tests in `tests/`.

## Data, Models, and Artifacts
- Do not commit large artifacts or datasets. The repo already ignores common binaries (e.g., `*.pkl`) and run outputs.
- Store local models under `src/models/RL/models/` and tensorboard logs under `src/models/RL/runs/` (both untracked by default).

## Coding Standards
- Prefer clear, typed function signatures where practical.
- Keep functions small and cohesive; avoid unnecessary abstractions.
- Use docstrings for public functions/classes; mention shapes, units, and assumptions for RL/stateful code.
- Handle errors explicitly; raise actionable exceptions with context.

## Security & Secrets
- Load credentials via environment variables (e.g., `python-dotenv`) but never commit `.env` files.
- Never echo keys to logs, stack traces, or test output.
- If a task requires credentials, request sanitized placeholders and show users how to inject them locally.

## Review Checklist (for agents before finalizing changes)
- [ ] Change is minimal and scoped to the task.
- [ ] No `.env` or `*.ini` files were read, modified, or committed.
- [ ] Sensitive values are not logged or embedded.
- [ ] Tests pass locally (or updated appropriately).
- [ ] README/docstrings updated if behavior or usage changed.

## Notes for This Repo
- Python version pin is modern (3.13). If adding new deps, ensure they support 3.13.
- RL rendering and TensorBoard are optional during CI-style runs; keep them guarded by flags.
- When in doubt, prefer simple, fast features and clear reward definitions for RL experiments.

---
If you need changes to `.gitignore` or project policies (e.g., stricter formatting or CI setup), propose them first rather than applying unrequested repo-wide edits.
