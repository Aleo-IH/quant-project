from src.backtester.crypto_broker_env import CryptoTradingEnv


def test_env_walk_through(trending_close_df):
    cols = ["ATRr_14", "Number of Trades", "volume"]
    env = CryptoTradingEnv(
        trending_close_df, cols, n_positions=3, transaction_fee=0.0005
    )
    obs, _ = env.reset()

    total_r = 0.0
    # stratégie bête: alterne 0% BTC / 50% / 100%
    for i in range(15):
        a = i % env.n_positions
        _, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            break

    assert "balance" in info and info["balance"] > 0
