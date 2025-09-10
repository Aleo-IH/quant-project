import pandas as pd
from src.binance.binance_trader import BinanceTrader


def test_account_and_misc_endpoints():
    t = BinanceTrader(testnet=True)
    info = t.get_account_info()
    assert "balances" in info
    ob = t.get_order_book("BTCUSDT", limit=5)
    assert isinstance(ob, pd.DataFrame)
    tr = t.get_recent_trades("BTCUSDT", limit=5)
    assert isinstance(tr, pd.DataFrame)
    htr = t.get_historical_trades("BTCUSDT", limit=5)
    assert isinstance(htr, pd.DataFrame)
    mytr = t.get_my_trades("BTCUSDT", limit=5)
    assert isinstance(mytr, pd.DataFrame)
    dep = t.get_deposit_history()
    assert isinstance(dep, pd.DataFrame)
    wdr = t.get_withdraw_history()
    assert isinstance(wdr, pd.DataFrame)
    asset = t.get_asset_detail()
    assert isinstance(asset, pd.DataFrame)
