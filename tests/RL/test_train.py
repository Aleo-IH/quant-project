from pathlib import Path
import os
import types
import numpy as np
import pandas as pd
import pytest

def test_train_ppo_happy_path(monkeypatch, tmp_pkls_dir: Path):
    # Import module under test
    import models.RL.train as train

    calls = types.SimpleNamespace(
        init_kwargs=None,
        learned=False,
        learned_timesteps=None,
        saved_to=None,
    )

    class DummyPPO:
        def __init__(self, **kwargs):
            calls.init_kwargs = kwargs
            self.kwargs = kwargs

        def learn(self, total_timesteps: int, progress_bar: bool = False, tb_log_name: str | None = None, callback=None):
            calls.learned = True
            calls.learned_timesteps = int(total_timesteps)
            return self

        def save(self, path):
            calls.saved_to = str(path)
            # Touch a file to emulate save
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"dummy-model")

    # Swap PPO with dummy to avoid real training
    monkeypatch.setattr(train, "PPO", DummyPPO)

    model_path = tmp_pkls_dir.parent / "out" / "ppo.zip"
    model = train.train_ppo(
        data_dir=str(tmp_pkls_dir),
        total_timesteps=100,
        window=10,
        fee=0.001,
        use_tensorboard=False,
        model_path=str(model_path),
        eval_ratio=0.2,
    )

    # Validate dummy interactions
    assert isinstance(model, DummyPPO)
    assert calls.learned is True
    assert calls.learned_timesteps == 100
    assert calls.saved_to == str(model_path)
    assert Path(model_path).exists()

    # Verify env was passed via kwargs and seed/policy present from params file
    assert "env" in calls.init_kwargs
    assert "policy" in calls.init_kwargs
