import pandas as pd
from src.binance.binance_trader import BinanceTrader


def test_get_candlestick_data_single():
    t = BinanceTrader(testnet=True)
    df = t.get_candlestick_data("BTCUSDT", "1m", limit=50)
    assert isinstance(df, pd.DataFrame)
    assert {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "Quote Asset Volume",
        "Number of Trades",
        "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume",
    } <= set(df.columns)


def test_get_candlestick_data_multi():
    t = BinanceTrader(testnet=True)
    df = t.get_candlestick_data(["BTCUSDT", "ETHUSDT"], "1m", limit=10)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.MultiIndex)


def test_get_candlestick_data_range():
    t = BinanceTrader(testnet=True)
    df = t.get_candlestick_data_range("BTCUSDT", "1m", "2024-01-01", "2024-01-01 00:20")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "open" in df.columns
