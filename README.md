# Quant Project

End-to-end quantitative trading playground: market data ingestion, research, backtesting, and a fast Reinforcement Learning (RL) stack powered by Stable-Baselines3 PPO on a custom trading environment with live rendering.

Broker/data: Binance (connector integrated). Codebase targets reproducible research and quick iteration on models and strategies.

## What’s New

- RL module under `src/models/RL` with:
  - Custom Gymnasium environment (`TradingEnv`) optimized for speed (vectorized features, minimal overhead).
  - PPO training pipeline with optional live rendering and TensorBoard logging.
  - Handy utilities to load OHLCV data, run an inference showcase with rendering, and load saved models.
- Visual live render showing price, trade markers, position timeline, and equity curve while an agent acts.

## RL Overview (PPO)

- Algorithm: PPO from Stable-Baselines3 (`stable-baselines3[extra]`).
- Action space: discrete positions {-1, 0, +1} (encoded as {0,1,2}).
- Observation: sliding window (default 64) of simple, fast features derived from OHLCV:
  - `log_ret` (log return), `hl_range` (high-low range), `oc_change` (open-close change), `vol_chg` (volume change).
- Reward: `position * log_return - fee * |position_change|`, scaled by `reward_scale`.
- Rendering (live):
  - Top: price with trade markers (green=long, red=short, blue=flat/close).
  - Bottom: step plot of position (-1/0/+1) and right-side stats (cumulative reward, fees, realized PnL, trades).

Code reference:
- Env: `src/models/RL/env.py` (`TradingEnv`)
- Training: `src/models/RL/train.py` (`train_ppo`, `TensorBoardServer`)
- Utils: `src/models/RL/utils.py` (`load_pkls`, `make_env_from_df`, `showcase_model`, `load_model`)

## Quickstart

### Installation

- Python ≥ 3.13 (see `pyproject.toml`).
- Install in editable mode (or use your preferred tool):

```
pip install -e .
```

This pulls dependencies including `stable-baselines3[extra]`, `gymnasium`, `pandas`, `numpy`, `matplotlib`, etc.

### Data format

- Use one or many `.pkl` files containing OHLCV columns (case-insensitive): `Open, High, Low, Close, Volume`.
- Loading helper merges/sorts and keeps numeric columns: `load_pkls(path_or_folder)`.

### Train PPO

Python API example:

```python
from src.models.RL.train import train_ppo

model = train_ppo(
    data_dir="path/to/pkl_or_folder",  # single file or directory of .pkl
    total_timesteps=200_000,
    window=64,
    fee=0.0005,
    use_tensorboard=True,
    tensorboard_log_dir="src/models/RL/runs/ppo_trader",
    model_path="src/models/RL/models/ppo_trader/ppo_trading.zip",
)
```

Notes:
- When `use_tensorboard=True`, a TensorBoard process starts automatically and stops at the end of training.
- A lightweight render callback shows the environment live during training at a low frequency (configurable).

### Showcase (Live Render)

Run a saved model over a dataset and open the live plot window:

```python
from src.models.RL.utils import load_model, showcase_model

model = load_model("src/models/RL/models/ppo_trader/ppo_trading.zip")
summary, steps_info, last_info = showcase_model(
    model,
    data="path/to/pkl_or_folder",
    window=64,
    fee=0.0005,
    deterministic=True,
    render_sleep=0.0,  # increase to slow down
)
print(summary, steps_info, last_info)
```

The render shows:
- Price + trade markers: green (enter long), red (enter short), blue (close/flat).
- Position timeline: discrete -1/0/+1.
- Stats panel: cumulative reward, fees, (realized) log returns, number of trades.

## Backtesting & Data

- A separate backtesting environment lives in `src/backtester/crypto_broker_env.py` (Gymnasium-compatible) for research.
- Binance connector and data utilities are under `src/binance/`.

## Prior Research Notes

- Random-Forest chain classifier for forward-return quantile classification (≈30% accuracy per quantile in rolling CV) — above random but insufficient edge.
- Random Forest + Logistic Regression stacking with ATR context for thresholded log-return hits — increased trade frequency but poor risk-adjusted metrics.
- Permutation tests: validate in-sample signal isn’t spurious by benchmarking against many time-shuffled permutations with identical moments/correlations; only promote strategies that rank in the top quantiles on real sequences vs permutations, then confirm on out-of-sample.

## Repository Layout

- `src/models/RL/`: RL env, training loop (PPO), and utilities for rendering and inference.
- `src/backtester/`: standalone backtesting environment and tests.
- `src/binance/`: exchange connectivity and utilities.
- `tests/`: unit tests for core pieces (e.g., envs).

## Tips

- If your render window does not appear, ensure a local GUI backend is available (e.g., install `matplotlib`).
- Use smaller `window` or fewer timesteps for quick iteration; increase once stable.
- For TensorBoard: point your browser to the printed URL (defaults to `http://localhost:6006/`).
