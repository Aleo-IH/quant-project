from __future__ import annotations

import os
import subprocess
from typing import Optional
import json
import socket


from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

try:  # Support both package and script execution
    from .utils import load_pkls, make_env_from_df
except Exception:  # pragma: no cover - fallback for direct script runs
    from utils import load_pkls, make_env_from_df


class TensorBoardServer:
    """Manage a TensorBoard subprocess that starts/stops with context manager.

    Uses `python -m tensorboard` to avoid OS-specific executables.
    """

    def __init__(self, logdir: str, host: str = "localhost", port: int = 6006):
        self.logdir = str(logdir)
        self.host = host
        self.port = int(port)
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        if self.proc is not None:
            # If already started, report whether it's running
            return self.proc.poll() is None

        # Helper: test if TCP port is accepting connections
        def _port_ready(host: str, port: int, timeout: float = 0.2) -> bool:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                return False

        # Prepare common subprocess flags
        creationflags = 0
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP (0x00000200) to allow safe termination
            creationflags = 0x00000200

        self.port = int(self.port)
        cmd = ["tensorboard", "--logdir", self.logdir]

        # Start process with output suppressed to avoid pipe backpressure
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        if _port_ready(self.host, self.port):
            return True

        # If here, either the process exited or never bound the port in time
        # If process is still alive, consider it started (will likely bind soon)
        if self.proc.poll() is None:
            return True
        return False

    def stop(self):
        if self.proc is None:
            return True
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=2)
                except Exception:
                    pass
            # Success if process is no longer running
            success = self.proc.poll() is not None
        finally:
            self.proc = None
        return success

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def train_ppo(
    data_dir: str | os.PathLike,
    total_timesteps: int = 100_000,
    window: int = 64,
    fee: float = 0.0005,
    use_tensorboard: bool = True,
    tensorboard_log_dir: str | os.PathLike = "src/models/RL/runs/ppo_trader",
    tensorboard_host: str = "localhost",
    tensorboard_port: int = 6006,
    eval_ratio: float = 0.2,
    seed: int = 42,
    model_path: Optional[
        str | os.PathLike
    ] = "src/models/RL/models/ppo_trader/ppo_trading.zip",
) -> PPO:
    """Train PPO on OHLCV PKL data located in `data_dir`.

    - Loads .pkl files from the given folder (or a single .pkl file path).
    - Splits the data into train/eval.
    - If `use_tensorboard` is True, starts a TensorBoard server and logs training.
      TensorBoard is automatically stopped at the end.
    """

    # Lightweight callback to render the env during training without blocking
    class RenderCallback(BaseCallback):
        def __init__(self, render_freq: int = 10):
            super().__init__()
            self.render_freq = max(1, int(render_freq))

        def _on_step(self) -> bool:
            # Render every N calls to keep overhead small
            if (self.n_calls % self.render_freq) != 0:
                return True
            try:
                # Prefer vecenv's render if available
                vecenv = self.training_env
                try:
                    vecenv.render("human")  # type: ignore[arg-type]
                except Exception:
                    # Fallback: iterate underlying envs and call their render
                    envs = getattr(vecenv, "envs", [])
                    for e in envs or []:
                        try:
                            # unwrap common wrappers
                            base = getattr(e, "env", e)
                            base = getattr(base, "unwrapped", base)
                            if hasattr(base, "render"):
                                base.render()
                        except Exception:
                            continue
            except Exception:
                # Never break training on render issues
                pass
            return True

    df = load_pkls(data_dir)
    n = len(df)
    if n < window * 2 + 10:
        raise ValueError("Not enough data for training and evaluation.")

    split = int(n * (1 - eval_ratio))
    train_df = df.iloc[:split]
    eval_df = df.iloc[max(0, split - window) :]

    def _train_env_fn():
        return Monitor(
            make_env_from_df(train_df, window=window, fee=fee, render_mode="human")
        )

    def _eval_env_fn():
        return Monitor(
            make_env_from_df(eval_df, window=window, fee=fee, render_mode="human")
        )

    train_env = DummyVecEnv([_train_env_fn])

    policy_kwargs = dict(net_arch=[256, 256])
    tensorboard_log = str(tensorboard_log_dir) if use_tensorboard else None

    with open(os.path.join(os.path.dirname(__file__), "model_params.json")) as f:
        params_dict = json.load(f)

    params_dict.update(
        {
            "env": train_env,
            "tensorboard_log": tensorboard_log,
            "seed": seed,
            "policy_kwargs": policy_kwargs,
        }
    )

    model = PPO(**params_dict)

    # Always attach render callback so you can see live updates while training
    render_cb = RenderCallback(render_freq=60)

    if use_tensorboard:
        os.makedirs(str(tensorboard_log_dir), exist_ok=True)
        tb_server = TensorBoardServer(
            logdir=str(tensorboard_log_dir),
            host=tensorboard_host,
            port=tensorboard_port,
        )
        started_ok = tb_server.start()
        url = f"http://{tb_server.host}:{tb_server.port}/"
        print(f"TensorBoard launch: {'SUCCESS' if started_ok else 'FAILED'} -> {url}")
        try:
            model.learn(
                total_timesteps=int(total_timesteps),
                progress_bar=True,
                tb_log_name="PPO-DRL-Trader",
                callback=render_cb,
            )
        finally:
            stopped_ok = tb_server.stop()
            print(f"TensorBoard stop: {'SUCCESS' if stopped_ok else 'FAILED'} -> {url}")

    else:
        model.learn(
            total_timesteps=int(total_timesteps),
            progress_bar=True,
            callback=render_cb,
        )

    if model_path is not None:
        model.save(model_path)

    return model


__all__ = [
    "train_ppo",
    "TensorBoardServer",
]
