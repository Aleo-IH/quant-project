import logging
import pandas as pd
from typing import Union, List, Optional
from datetime import datetime
from dateutil import parser
import os
from configparser import ConfigParser

# Binance Connector
from binance.spot import Spot as Client
from binance.error import ClientError, ServerError

# Configure a simple logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
)
logger.addHandler(stream_handler)


def get_api_keys():
    """
    Récupère les clés API depuis un fichier config.ini situé dans le même dossier que ce script.

    Returns:
        tuple: (api_key, api_secret) si les clés sont trouvées, sinon None.
    """
    # Obtenez le chemin du fichier config.ini dans le même dossier que ce script
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")

    # Vérifiez si le fichier existe
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Le fichier config.ini est introuvable à l'emplacement : {config_path}"
        )

    # Lisez le fichier de configuration
    config = ConfigParser()
    config.read(config_path)

    # Récupérez les clés
    try:
        api_key = config["keys"]["api_key"]
        api_secret = config["keys"]["api_secret"]
        return api_key, api_secret
    except KeyError as e:
        raise KeyError(f"Clé manquante dans le fichier config.ini : {e}")


class BinanceTrader:
    """
    A class to interact with the Binance API (testnet or live),
    including market data retrieval, order placement, account management,
    and historical data retrieval between two dates.
    """

    def __init__(self, testnet: bool = True):
        self.testnet = testnet

        api_key, api_secret = get_api_keys()

        if self.testnet:
            # Note: The testnet base_url for Spot is usually https://testnet.binance.vision
            self.client = Client(
                api_key, api_secret, base_url="https://testnet.binance.vision"
            )
        else:
            # Production environment
            self.client = Client(api_key, api_secret)
        logger.info(self.client.account_status())

    def get_candlestick_data(
        self, symbols: Union[str, List[str]], interval: str, limit: int = 500
    ) -> pd.DataFrame:
        """
        Retrieve candlestick (OHLCV) data for one or multiple symbols.
        Limited per request (maximum 1000).
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    klines = self.client.klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                    )
                    df = pd.DataFrame(klines)
                    df.columns = [
                        "Open Time",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Close Time",
                        "Quote Asset Volume",
                        "Number of Trades",
                        "Taker Buy Base Asset Volume",
                        "Taker Buy Quote Asset Volume",
                        "Ignore",
                    ]
                    data_frames.append((symbol, df))

                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                klines = self.client.klines(
                    symbol=symbols, interval=interval, limit=limit
                )
                df = pd.DataFrame(klines)
                df.columns = [
                    "Open Time",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Close Time",
                    "Quote Asset Volume",
                    "Number of Trades",
                    "Taker Buy Base Asset Volume",
                    "Taker Buy Quote Asset Volume",
                    "Ignore",
                ]
                return df
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving candlestick data: {e}")
            return pd.DataFrame()

    def get_candlestick_data_range(
        self,
        symbol: Union[str, List[str]],
        interval: str,
        start_dt: Union[str, datetime],
        end_dt: Union[str, datetime],
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Retrieve candlestick (OHLCV) data for a symbol between two specified dates,
        looping to exceed the per-request limit if needed.
        """
        if isinstance(start_dt, str):
            start_dt = parser.parse(start_dt)
        if isinstance(end_dt, str):
            end_dt = parser.parse(end_dt)

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        frames = []
        current_start = start_ts

        try:
            # Handling multiple symbols
            if isinstance(symbol, list):
                all_symbol_frames = []
                for sym in symbol:
                    sym_frames = []
                    current_start_sym = current_start

                    while True:
                        klines = self.client.klines(
                            symbol=sym,
                            interval=interval,
                            limit=limit,
                            startTime=current_start_sym,
                        )

                        if not klines:
                            break

                        df = pd.DataFrame(
                            klines,
                            columns=[
                                "Open Time",
                                "Open",
                                "High",
                                "Low",
                                "Close",
                                "Volume",
                                "Close Time",
                                "Quote Asset Volume",
                                "Number of Trades",
                                "Taker Buy Base Asset Volume",
                                "Taker Buy Quote Asset Volume",
                                "Ignore",
                            ],
                        )
                        sym_frames.append(df)

                        last_close_time = klines[-1][6]
                        next_start = last_close_time + 1

                        if next_start > end_ts:
                            break

                        current_start_sym = next_start

                    if sym_frames:
                        sym_df = pd.concat(sym_frames, ignore_index=True)
                        sym_df = sym_df[sym_df["Open Time"] <= end_ts]
                        sym_df.drop(columns=["Close Time", "Ignore"], inplace=True)
                        sym_df["Open Time"] = pd.to_datetime(
                            sym_df["Open Time"], unit="ms"
                        )
                        all_symbol_frames.append((sym, sym_df))

                if all_symbol_frames:
                    df = pd.concat(
                        [df.set_index("Open Time") for _, df in all_symbol_frames],
                        keys=[sym for sym, _ in all_symbol_frames],
                        names=["Symbol", "Open Time"],
                    )
                    df = df.astype(float)
                    df = df.swaplevel(0, 1).sort_index()
                    df.name = (
                        f"{' | '.join(symbol)} ; {interval} tf ; "
                        f"{df.index.get_level_values('Open Time').min()} to "
                        f"{df.index.get_level_values('Open Time').max()}"
                    )
                    return df
                else:
                    return pd.DataFrame()

            # Single symbol
            else:
                while True:
                    klines = self.client.klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        startTime=current_start,
                    )

                    if not klines:
                        break

                    df = pd.DataFrame(
                        klines,
                        columns=[
                            "Open Time",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                            "Close Time",
                            "Quote Asset Volume",
                            "Number of Trades",
                            "Taker Buy Base Asset Volume",
                            "Taker Buy Quote Asset Volume",
                            "Ignore",
                        ],
                    )
                    frames.append(df)

                    last_close_time = klines[-1][6]
                    next_start = last_close_time + 1

                    if next_start > end_ts:
                        break

                    current_start = next_start

                if frames:
                    big_df = pd.concat(frames, ignore_index=True)
                    big_df = big_df[big_df["Open Time"] <= end_ts]
                    big_df.drop(columns=["Close Time", "Ignore"], inplace=True)
                    big_df["Open Time"] = pd.to_datetime(big_df["Open Time"], unit="ms")
                    return big_df.set_index("Open Time").astype(float)
                else:
                    return pd.DataFrame()

        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving range candlestick data: {e}")
            return pd.DataFrame()

    def get_order_book(
        self, symbols: Union[str, List[str]], limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve the order book (depth) for one or multiple symbols.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    ob = self.client.depth(symbol=symbol, limit=limit)
                    df = pd.DataFrame(ob)
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                ob = self.client.depth(symbol=symbols, limit=limit)
                return pd.DataFrame(ob)
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving order book: {e}")
            return pd.DataFrame()

    def get_recent_trades(
        self, symbols: Union[str, List[str]], limit: int = 500
    ) -> pd.DataFrame:
        """
        Retrieve recent trades for one or multiple symbols.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    trades = self.client.trades(symbol=symbol, limit=limit)
                    df = pd.DataFrame(trades)
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                trades = self.client.trades(symbol=symbols, limit=limit)
                return pd.DataFrame(trades)
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving recent trades: {e}")
            return pd.DataFrame()

    def get_historical_trades(
        self, symbols: Union[str, List[str]], limit: int = 500
    ) -> pd.DataFrame:
        """
        Retrieve historical trades for one or multiple symbols.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    trades = self.client.historical_trades(symbol=symbol, limit=limit)
                    df = pd.DataFrame(trades)
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                trades = self.client.historical_trades(symbol=symbols, limit=limit)
                return pd.DataFrame(trades)
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving historical trades: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Account information methods
    # -------------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """
        Retrieve Binance account information.
        """
        try:
            return self.client.account()
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving account info: {e}")
            return {}

    def get_position(
        self, symbols: Union[str, List[str]]
    ) -> Union[pd.DataFrame, dict, None]:
        """
        Retrieve position(s) for one or multiple symbols (spot balances).
        """
        try:
            account_info = self.client.account()
            positions = account_info.get("balances", [])

            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    # For spot, we usually consider the base asset
                    asset = symbol.replace("USDT", "")
                    pos = next((p for p in positions if p["asset"] == asset), None)
                    if pos:
                        data_frames.append((symbol, pd.DataFrame([pos])))
                if data_frames:
                    return pd.concat(
                        [df for _, df in data_frames],
                        keys=[sym for sym, _ in data_frames],
                    )
                else:
                    return None
            else:
                asset = symbols.replace("USDT", "")
                return next((p for p in positions if p["asset"] == asset), None)

        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving position: {e}")
            return None

    # -------------------------------------------------------------------------
    # Trading methods
    # -------------------------------------------------------------------------

    def place_order(
        self,
        symbols: Union[str, List[str]],
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> Union[dict, None]:
        """
        Place an order (or multiple) with the specified parameters.
        """
        try:
            if isinstance(symbols, list):
                results = {}
                for symbol in symbols:
                    results[symbol] = self._place_order_helper(
                        symbol,
                        side,
                        order_type,
                        quantity,
                        price,
                        stop_price,
                        time_in_force,
                    )
                return results
            else:
                return self._place_order_helper(
                    symbols,
                    side,
                    order_type,
                    quantity,
                    price,
                    stop_price,
                    time_in_force,
                )
        except (ClientError, ServerError) as e:
            logger.error(f"Error placing order: {e}")
            return None

    def _place_order_helper(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        stop_price: Optional[float],
        time_in_force: str,
    ) -> dict:
        """
        Internal method to place an order on a single symbol.
        """
        # The binance-connector uses client.new_order(...)
        # with arguments as strings in some cases
        if order_type == "MARKET":
            return self.client.new_order(
                symbol=symbol, side=side, type=order_type, quantity=quantity
            )
        elif order_type == "LIMIT":
            return self.client.new_order(
                symbol=symbol,
                side=side,
                type=order_type,
                timeInForce=time_in_force,
                quantity=quantity,
                price=str(price),
            )
        elif order_type == "STOP_LOSS_LIMIT":
            return self.client.new_order(
                symbol=symbol,
                side=side,
                type=order_type,
                timeInForce=time_in_force,
                quantity=quantity,
                price=str(price),
                stopPrice=str(stop_price),
            )
        else:
            raise ValueError("Unsupported order type.")

    def get_order_status(
        self, symbols: Union[str, List[str]], order_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Check the status of an order.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    order_data = self.client.get_order(symbol=symbol, orderId=order_id)
                    df = pd.DataFrame([order_data])
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                order_data = self.client.get_order(symbol=symbols, orderId=order_id)
                return pd.DataFrame([order_data])
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving order status: {e}")
            return None

    def cancel_order(
        self, symbols: Union[str, List[str]], order_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Cancel an order.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    cancel_resp = self.client.cancel_order(
                        symbol=symbol, orderId=order_id
                    )
                    df = pd.DataFrame([cancel_resp])
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                cancel_resp = self.client.cancel_order(symbol=symbols, orderId=order_id)
                return pd.DataFrame([cancel_resp])
        except (ClientError, ServerError) as e:
            logger.error(f"Error cancelling order: {e}")
            return None

    def get_open_orders(
        self, symbols: Optional[Union[str, List[str]]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve all open orders for one or multiple symbols.
        If no symbol is specified, retrieve all open orders.
        """
        try:
            if symbols:
                if isinstance(symbols, list):
                    data_frames = []
                    for symbol in symbols:
                        open_orders = self.client.get_open_orders(symbol=symbol)
                        df = pd.DataFrame(open_orders)
                        data_frames.append((symbol, df))
                    return pd.concat(
                        [df for _, df in data_frames],
                        keys=[sym for sym, _ in data_frames],
                    )
                else:
                    open_orders = self.client.get_open_orders(symbol=symbols)
                    return pd.DataFrame(open_orders)
            else:
                # Retrieve all open orders
                open_orders = self.client.get_open_orders()
                return pd.DataFrame(open_orders)
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving open orders: {e}")
            return None

    # -------------------------------------------------------------------------
    # Additional features
    # -------------------------------------------------------------------------

    def get_all_tickers(self) -> pd.DataFrame:
        """
        Retrieve current prices for all tickers.
        """
        try:
            # binance-connector returns a list of dicts with "symbol" and "price"
            tickers = self.client.ticker_price()
            return pd.DataFrame(tickers)
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving tickers: {e}")
            return pd.DataFrame()

    def get_my_trades(
        self, symbols: Union[str, List[str]], limit: int = 50
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve account trades for one or multiple symbols.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    trades = self.client.my_trades(symbol=symbol, limit=limit)
                    df = pd.DataFrame(trades)
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                trades = self.client.my_trades(symbol=symbols, limit=limit)
                return pd.DataFrame(trades)
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving account trades: {e}")
            return None

    def get_deposit_history(self) -> Optional[pd.DataFrame]:
        """
        Retrieve deposit history.
        """
        try:
            # binance-connector: deposit_history() returns a dict with "depositList"
            resp = self.client.deposit_history()
            if "depositList" in resp:
                return pd.DataFrame(resp["depositList"])
            else:
                return pd.DataFrame()
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving deposit history: {e}")
            return None

    def get_withdraw_history(self) -> Optional[pd.DataFrame]:
        """
        Retrieve withdrawal history.
        """
        try:
            # binance-connector: withdraw_history() returns a dict with "withdrawList"
            resp = self.client.withdraw_history()
            if "withdrawList" in resp:
                return pd.DataFrame(resp["withdrawList"])
            else:
                return pd.DataFrame()
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving withdrawal history: {e}")
            return None

    def get_asset_detail(self) -> Optional[pd.DataFrame]:
        """
        Retrieve asset details (deposit/withdraw info, etc.).
        """
        try:
            # binance-connector: asset_detail() returns a dict with "assetDetail"
            resp = self.client.asset_detail()
            if "assetDetail" in resp:
                # resp["assetDetail"] is a dict keyed by asset name
                # e.g. {"BTC": { ... }, "ETH": {...}}
                details = []
                for asset, detail in resp["assetDetail"].items():
                    detail["asset"] = asset
                    details.append(detail)
                return pd.DataFrame(details)
            else:
                return pd.DataFrame()
        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving asset detail: {e}")
            return None

    def get_trade_fee(
        self, symbols: Optional[Union[str, List[str]]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve trading fees for one or multiple symbols.
        If no symbol is specified, retrieve fees for all symbols.
        """
        try:
            if symbols:
                if isinstance(symbols, list):
                    data_frames = []
                    for symbol in symbols:
                        fee_data = self.client.trade_fee(symbol=symbol)
                        # Usually returns something like {"tradeFee": [{...}]}
                        if "tradeFee" in fee_data:
                            df = pd.DataFrame(fee_data["tradeFee"])
                            data_frames.append((symbol, df))
                    if data_frames:
                        return pd.concat(
                            [df for _, df in data_frames],
                            keys=[sym for sym, _ in data_frames],
                        )
                    else:
                        return pd.DataFrame()
                else:
                    fee_data = self.client.trade_fee(symbol=symbols)
                    if "tradeFee" in fee_data:
                        return pd.DataFrame(fee_data["tradeFee"])
                    else:
                        return pd.DataFrame()
            else:
                # Retrieve fees for all symbols
                fee_data = self.client.trade_fee()
                if "tradeFee" in fee_data:
                    return pd.DataFrame(fee_data["tradeFee"])
                else:
                    return pd.DataFrame()

        except (ClientError, ServerError) as e:
            logger.error(f"Error retrieving trade fee: {e}")
            return None
