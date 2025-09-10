from __future__ import annotations

# Core/typing/path utilities
from pathlib import Path
from typing import Optional

# Third-party libs
import numpy as np
import pandas as pd
import gymnasium as gym
import time

# Local env import (works for package or script execution)
try:
    from .env import TradingEnv
except Exception:  # pragma: no cover
    from env import TradingEnv
import os
import stable_baselines3 as sb3


def load_pkls(data_path: str | os.PathLike) -> pd.DataFrame:
    """Load one or many .pkl files into a single chronological DataFrame.

    - If `data_path` is a file, loads it directly.
    - If `data_path` is a folder, loads all `*.pkl` inside and concatenates by index.
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {data_path}")

    def _read_one(fp: Path) -> pd.DataFrame:
        df = pd.read_pickle(fp)
        if df.index.is_monotonic_increasing is False:
            df = df.sort_index()
        return df

    if p.is_file():
        df = _read_one(p)
    else:
        files = sorted([f for f in p.glob("*.pkl") if f.is_file()])
        if not files:
            raise FileNotFoundError(f"No .pkl files found in folder: {data_path}")
        parts = [_read_one(f) for f in files]
        df = pd.concat(parts, axis=0)
        df = df[~df.index.duplicated(keep="last")]

    # Keep numeric columns and ensure OHLCV present (case-insensitive)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns and col not in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_df = numeric_df.dropna()
    return numeric_df


def make_env_from_df(
    df: pd.DataFrame,
    window: int = 64,
    fee: float = 0.0005,
    render_mode: str | None = None,
) -> gym.Env:
    return TradingEnv(df=df, window=window, fee=fee, render_mode=render_mode)


def showcase_model(
    model,
    data: pd.DataFrame | str | os.PathLike,
    window: int = 64,
    fee: float = 0.0005,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    render_sleep: float = 0.0,
):
    """Run a single pass using `model.predict` and live-render the environment.

    Parameters
    - model: Trained SB3 model with `.predict(obs, deterministic=...)`
    - data: DataFrame with OHLCV or a path to .pkl file(s)
    - window, fee: Environment parameters
    - deterministic: Use deterministic actions
    - max_steps: Optional cap on number of steps to run
    - render_sleep: Optional `time.sleep` per step to slow down visualization

    Returns
    - dict summary with total_reward, steps, and last info
    """
    df = data if isinstance(data, pd.DataFrame) else load_pkls(data)
    env = make_env_from_df(df, window=window, fee=fee)
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    last_info = {}

    try:
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            # Extract scalar action
            if np.isscalar(action):
                act = int(action)
            else:
                act = int(np.asarray(action).ravel()[0])

            obs, reward, terminated, truncated, info = env.step(act)
            total_reward += float(reward)
            steps += 1
            last_info = info

            # Render live
            env.render()
            if render_sleep > 0:
                time.sleep(render_sleep)

            if terminated or truncated:
                break
            if max_steps is not None and steps >= max_steps:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass

    return {"total_reward": total_reward}, {"steps": steps}, {"last_info": last_info}


def load_model(model_path: str):
    return sb3.PPO.load(model_path)


__all__ = ["load_pkls", "make_env_from_df", "showcase_model", "load_model"]
