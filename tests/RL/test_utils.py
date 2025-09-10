import os
from pathlib import Path
import numpy as np
import pandas as pd

from models.RL import utils
from models.RL.env import TradingEnv


def test_load_pkls_single_file(tmp_single_pkl: Path):
    df = utils.load_pkls(tmp_single_pkl)
    # Must contain numeric OHLCV and drop non-numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns or col.lower() in df.columns
    assert len(df) > 0


def test_load_pkls_folder(tmp_pkls_dir: Path):
    df = utils.load_pkls(tmp_pkls_dir)
    # Deduplicated and concatenated
    assert len(df) > 0
    assert df.index.is_monotonic_increasing


def test_make_env_from_df(simple_ohlcv_df: pd.DataFrame):
    df = simple_ohlcv_df.copy()
    env = utils.make_env_from_df(df, window=10, fee=0.001)
    assert isinstance(env, TradingEnv)
    obs, _ = env.reset()
    assert obs.shape == (10, 4)


class DummyModel:
    def __init__(self, action=2):
        self._a = action

    def predict(self, obs, deterministic=True):
        # Return a valid discrete action and dummy state
        return self._a, None


def test_showcase_model_with_dummy(simple_ohlcv_df: pd.DataFrame):
    model = DummyModel(action=2)
    res, steps_info, last = utils.showcase_model(model, simple_ohlcv_df, window=8, max_steps=25)
    assert isinstance(res, dict) and "total_reward" in res
    assert isinstance(steps_info, dict) and "steps" in steps_info
    assert steps_info["steps"] > 0
    assert isinstance(last, dict) and "last_info" in last
