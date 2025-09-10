import numpy as np
import pandas as pd
import pytest

from models.RL.env import TradingEnv


def build_df_from_base(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase columns to test case-insensitive handling
    out = pd.DataFrame(
        {
            "open": df["Open"].astype(float).values,
            "high": df["High"].astype(float).values,
            "low": df["Low"].astype(float).values,
            "close": df["Close"].astype(float).values,
            "volume": df["Volume"].astype(float).values,
        },
        index=df.index,
    )
    return out


def test_env_reset_and_obs_shape(simple_ohlcv_df: pd.DataFrame):
    df = build_df_from_base(simple_ohlcv_df)
    env = TradingEnv(df, window=16, fee=0.001)
    obs, info = env.reset()
    assert obs.shape == (16, 4)
    assert isinstance(info, dict)
    # Initial time index set to window
    assert env._t == 16


def test_env_step_reward_and_fee(simple_ohlcv_df: pd.DataFrame):
    window = 8
    fee = 0.01
    df = build_df_from_base(simple_ohlcv_df)
    env = TradingEnv(df, window=window, fee=fee, reward_scale=1.0)
    obs, _ = env.reset()

    # At first step, t == window, reward uses features[t-1, 0] which is log return at index window
    # Take long (action=2 => pos=+1)
    obs, reward1, done, trunc, info1 = env.step(2)
    assert not done and not trunc
    # expected: pos change 0->1 => transaction cost = fee * |1 - 0| = fee
    # ret = log(close_t / close_{t-1}), where t maps to features[env._t-1, 0]
    # env._t incremented after step, so for reward1, used features[window-1]
    t0 = window - 1
    # Reconstruct log return as in env.get_features
    close = df["close"].to_numpy()
    expected_ret = float(np.log(close[t0 + 1] / close[t0]))
    expected_reward1 = (1 * expected_ret - fee)
    assert np.isclose(reward1, expected_reward1, atol=1e-8)
    assert info1["position"] == 1

    # Stay long again, no transaction cost now
    obs, reward2, done, trunc, info2 = env.step(2)
    t1 = window  # now features[t1, 0]
    expected_ret2 = float(np.log(close[t1 + 1] / close[t1]))
    expected_reward2 = (1 * expected_ret2 - 0.0)
    assert np.isclose(reward2, expected_reward2, atol=1e-8)
    assert info2["position"] == 1

    # Go flat (action=1 => pos=0), incur fee once
    obs, reward3, done, trunc, info3 = env.step(1)
    expected_ret3 = float(np.log(close[t1 + 2] / close[t1 + 1]))
    expected_reward3 = (0 * expected_ret3 - fee)
    assert np.isclose(reward3, expected_reward3, atol=1e-8)
    assert info3["position"] == 0
    assert info3["trades"] >= 2  # at least two changes happened


def test_env_terminates_at_end(simple_ohlcv_df: pd.DataFrame):
    df = build_df_from_base(simple_ohlcv_df)
    env = TradingEnv(df, window=4, fee=0.0)
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done and steps < 10_000:
        # alternate actions
        a = [0, 1, 2][steps % 3]
        obs, r, done, trunc, info = env.step(a)
        steps += 1
    assert done
    assert env._t == env._done_t


def test_equity_series_and_markers(simple_ohlcv_df: pd.DataFrame):
    df = build_df_from_base(simple_ohlcv_df)
    env = TradingEnv(df, window=6, fee=0.001)
    env.reset()
    # open long -> hold -> close
    env.step(2)
    env.step(2)
    env.step(1)
    # markers should include at least one long open and one close
    assert len(env._trade_long_idx) >= 1
    assert len(env._trade_close_idx) >= 1
    # equity series should have finite values at filled indices
    filled = np.isfinite(env._equity_series).sum()
    assert filled >= 1
