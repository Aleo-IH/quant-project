"""Thin wrapper to keep backward compatibility and provide a simple CLI.

Environment and training have been moved to separate modules:
- env.py  -> TradingEnv
- train.py -> train_ppo (with optional TensorBoard auto-launch/stop)
"""

from __future__ import annotations

import os

try:  # Re-export with fallback so this file runs standalone
    from .env import TradingEnv  # type: ignore
    from .train import train_ppo  # type: ignore
except Exception:  # pragma: no cover - direct run fallback
    from env import TradingEnv  # type: ignore
    from train import train_ppo  # type: ignore


if __name__ == "__main__":
    # Simple environment-variable based CLI to trigger training.
    data_path = os.environ.get("RL_DATA_DIR", "src/data")
    timesteps = int(os.environ.get("RL_TIMESTEPS", "100000"))
    use_tb = os.environ.get("RL_USE_TENSORBOARD", "1") not in {"0", "false", "False"}
    tb_log_dir = os.environ.get("RL_TB_LOG", "runs/ppo_trader")
    tb_port = int(os.environ.get("RL_TB_PORT", "6006"))
    save_dir = os.environ.get("RL_SAVE_DIR", "models/ppo_trader")

    print(f"Training PPO on data from: {data_path}")
    print(f"TensorBoard: {'enabled' if use_tb else 'disabled'} | logdir: {tb_log_dir}")
    print(f"Saving best models to: {save_dir}")

    _ = train_ppo(
        data_dir=data_path,
        total_timesteps=timesteps,
        use_tensorboard=use_tb,
        tensorboard_log_dir=tb_log_dir,
        tensorboard_port=tb_port,
        save_best_path=save_dir,
    )
