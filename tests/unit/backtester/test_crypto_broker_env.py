import numpy as np
import pytest
from src.backtester.crypto_broker_env import CryptoTradingEnv

@pytest.fixture
def env(trending_close_df):
    cols = ["ATRr_14", "Number of Trades", "volume"]
    return CryptoTradingEnv(
        df=trending_close_df,
        column_names=cols,
        n_positions=5,
        initial_balance=1000.0,
        max_steps=20,
        sharpe_length=5,
        transaction_fee=0.0,
        trade_threshold=0.0,
    )

def test_reset_shape(env):
    obs, info = env.reset()
    assert obs.shape[0] == len(env.column_names) + 2
    assert env.current_step == 0
    assert env.balance == env.initial_balance

def test_step_progress_and_info(env):
    obs0, _ = env.reset()
    obs1, r, term, trunc, info = env.step(action=4)  # full BTC
    assert isinstance(r, float)
    assert info["current_step"] == 1
    assert "rolling_sharpe_ratio" in info
    assert obs1.shape == obs0.shape

def test_termination(env):
    env.reset()
    for _ in range(env.max_steps):
        _, _, term, _, _ = env.step(0)
    assert term is True

def test_trade_threshold(env):
    env.trade_threshold = 1e9  # tellement grand qu'aucun trade ne passe
    b0 = env.balance
    env.step(4)
    assert env.balance == b0  # aucune exécution -> balance inchangée

def test_fee_applied(trending_close_df):
    cols = ["ATRr_14", "Number of Trades", "volume"]
    env = CryptoTradingEnv(trending_close_df, cols, transaction_fee=0.01, trade_threshold=0.0)
    env.reset()
    # Achat plein -> des frais doivent diminuer la balance vs sans frais
    env_no_fee = CryptoTradingEnv(trending_close_df, cols, transaction_fee=0.0, trade_threshold=0.0)
    env_no_fee.reset()
    env.step(4)
    env_no_fee.step(4)
    assert env.balance < env_no_fee.balance

def test_reward_sign_when_price_up(trending_close_df):
    cols = ["ATRr_14", "Number of Trades", "volume"]
    env = CryptoTradingEnv(trending_close_df, cols, transaction_fee=0.0)
    env.reset()
    # Long BTC sur série haussière -> reward agrégé devrait être >= 0
    total_r = 0.0
    for _ in range(10):
        _, r, *_ = env.step(4)  # viser 100% BTC
        total_r += r
    assert total_r >= -1e-6  # tolérance numérique