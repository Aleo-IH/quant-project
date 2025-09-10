import pandas as pd
from src.binance.binance_trader import BinanceTrader


def test_place_order_and_status_cancel():
    t = BinanceTrader(testnet=True)
    resp = t.place_order("BTCUSDT", "BUY", "MARKET", quantity=0.01)
    assert resp["status"] == "FILLED"
    status = t.get_order_status("BTCUSDT", order_id=resp["orderId"])
    assert isinstance(status, pd.DataFrame)
    cancel = t.cancel_order("BTCUSDT", order_id=resp["orderId"])
    assert isinstance(cancel, pd.DataFrame)


def test_open_orders_trade_fee_and_tickers():
    t = BinanceTrader(testnet=True)
    oo = t.get_open_orders()
    assert isinstance(oo, pd.DataFrame)
    fees = t.get_trade_fee("BTCUSDT")
    assert not fees.empty
    tickers = t.get_all_tickers()
    assert "symbol" in tickers.columns and "price" in tickers.columns
