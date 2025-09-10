import time
import pytest


# Client factice qui imite les points utilisés par BinanceTrader
class FakeSpotClient:
    def __init__(self, api_key, api_secret, base_url=None):
        self.base_url = base_url
        self._start = int(time.time() * 1000) - 1000 * 60 * 60  # 1h d'historique
        self._step_ms = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}
        self._bars = 200

    def account_status(self):
        return {"data": "Normal"}

    def _gen_klines(self, symbol, interval, limit, startTime=None):
        step = self._step_ms.get(interval, 60_000)
        base = self._start if startTime is None else startTime
        out = []
        t = base
        for i in range(limit):
            open_ = 100 + 0.1 * i
            high = open_ + 0.5
            low = open_ - 0.5
            close = open_ + 0.2
            vol = 10 + i
            out.append(
                [
                    t,
                    open_,
                    high,
                    low,
                    close,
                    vol,
                    t + step - 1,
                    vol * close,
                    5 + i,
                    vol / 2,
                    vol * close / 2,
                    0,
                ]
            )
            t += step
        # stop si on dépasse une borne arbitraire
        if startTime and startTime > self._start + self._bars * step:
            return []
        return out

    def klines(self, symbol, interval, limit=500, startTime=None):
        return self._gen_klines(symbol, interval, limit, startTime)

    def depth(self, symbol, limit=100):
        return {
            "lastUpdateId": 1,
            "bids": [["100.0", "1.0"]],
            "asks": [["101.0", "2.0"]],
        }

    def trades(self, symbol, limit=500):
        return [{"id": i, "price": "100.0", "qty": "0.01"} for i in range(limit)]

    def historical_trades(self, symbol, limit=500):
        return [{"id": i, "price": "99.0", "qty": "0.02"} for i in range(limit)]

    def account(self):
        return {
            "balances": [
                {"asset": "BTC", "free": "0.5", "locked": "0"},
                {"asset": "USDT", "free": "1000", "locked": "0"},
            ]
        }

    def new_order(self, symbol, side, type, quantity, **kwargs):
        return {
            "symbol": symbol,
            "side": side,
            "type": type,
            "status": "FILLED",
            "orderId": 123,
        }

    def get_order(self, symbol, orderId):
        return {"symbol": symbol, "orderId": orderId, "status": "FILLED"}

    def cancel_order(self, symbol, orderId):
        return {"symbol": symbol, "orderId": orderId, "status": "CANCELED"}

    def get_open_orders(self, symbol=None):
        return [] if symbol else []

    def ticker_price(self):
        return [{"symbol": "BTCUSDT", "price": "50000.0"}]

    def my_trades(self, symbol, limit=50):
        return [
            {"symbol": symbol, "id": i, "price": "100.5", "qty": "0.01"}
            for i in range(limit)
        ]

    def deposit_history(self):
        return {"depositList": []}

    def withdraw_history(self):
        return {"withdrawList": []}

    def asset_detail(self):
        return {"assetDetail": {"BTC": {"minWithdrawAmount": "0.001"}}}

    def trade_fee(self, symbol=None):
        if symbol:
            return {
                "tradeFee": [{"symbol": symbol, "maker": "0.001", "taker": "0.001"}]
            }
        return {"tradeFee": [{"symbol": "BTCUSDT", "maker": "0.001", "taker": "0.001"}]}


@pytest.fixture(autouse=True)
def patch_binance_client(monkeypatch):
    # Patch des clés + du client
    import src.binance.binance_trader as bt

    monkeypatch.setattr(bt, "get_api_keys", lambda: ("k", "s"))
    monkeypatch.setattr(bt, "Client", FakeSpotClient)
