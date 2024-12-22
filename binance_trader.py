import logging
import pandas as pd
from typing import Union, List, Optional
from datetime import datetime
from dateutil import parser
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException

# Configuration d'un logger simple
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
)
logger.addHandler(stream_handler)


class BinanceTrader:
    """
    Une classe pour interagir avec l'API Binance (testnet ou live).
    Elle inclut la récupération de données de marché, passage d'ordres,
    gestion du compte, et maintenant la récupération de gros volumes de données
    entre deux dates.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.testnet = testnet
        if self.testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = "https://testnet.binance.vision/api"
        else:
            self.client = Client(api_key, api_secret)

    def get_candlestick_data(
        self, symbols: Union[str, List[str]], interval: str, limit: int = 500
    ) -> pd.DataFrame:
        """
        Récupère les données de chandeliers (OHLCV) pour un ou plusieurs symboles.
        Limité par la requête (maximum 1000).

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        interval : str
            Intervalle de temps pour chaque bougie (ex: '1m', '1h', '1d')
        limit : int, optional
            Nombre de bougies à récupérer (max 1000, par défaut 500)

        Returns
        -------
        pd.DataFrame
            DataFrame contenant les données de chandeliers.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    df = pd.DataFrame(
                        self.client.get_klines(
                            symbol=symbol,
                            interval=interval,
                            limit=limit,
                        )
                    )
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
                df = pd.DataFrame(
                    self.client.get_klines(
                        symbol=symbols, interval=interval, limit=limit
                    )
                )
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
        except BinanceAPIException as e:
            logger.error(
                f"Erreur lors de la récupération des données de chandeliers: {e}"
            )
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
        Récupère les données de chandeliers (OHLCV) pour un symbole entre deux dates spécifiées,
        en bouclant pour dépasser la limite autorisée par requête.

        Parameters
        ----------
        symbol : str
            Le symbole (ex: 'BTCUSDT').
        interval : str
            Intervalle de temps pour chaque bougie (ex: '1m', '1h', '1d').
        start_dt : str ou datetime
            Date/heure de début (ex: '2020-01-01' ou datetime(2020,1,1)).
        end_dt : str ou datetime
            Date/heure de fin (ex: '2020-02-01' ou datetime(2020,2,1)).
        limit : int, optional
            Nombre de bougies par requête (max 1000, par défaut 1000).

        Returns
        -------
        pd.DataFrame
            DataFrame contenant toutes les bougies entre start_dt et end_dt.
        """

        # Conversion des paramètres start_dt et end_dt en datetime si besoin
        if isinstance(start_dt, str):
            start_dt = parser.parse(start_dt)
        if isinstance(end_dt, str):
            end_dt = parser.parse(end_dt)

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        frames = []
        current_start = start_ts

        try:
            if isinstance(symbol, list):
                all_symbol_frames = []
                for sym in symbol:
                    sym_frames = []
                    current_start_sym = current_start

                    while True:
                        klines = self.client.get_klines(
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
                    df.name = f"{' | '.join(symbol)} ; {interval} tf ; {df.index.get_level_values('Open Time').min()} to {df.index.get_level_values('Open Time').max()}"
                    return df
                else:
                    return pd.DataFrame()
            else:
                while True:
                    klines = self.client.get_klines(
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

        except BinanceAPIException as e:
            logger.error(
                f"Erreur lors de la récupération des données de chandeliers sur plage : {e}"
            )
            return pd.DataFrame()

    def get_order_book(
        self, symbols: Union[str, List[str]], limit: int = 100
    ) -> pd.DataFrame:
        """
        Récupère le carnet d'ordres (order book) pour un ou plusieurs symboles.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        limit : int, optional
            Nombre d'ordres à récupérer (5, 10, 100), 100 par défaut

        Returns
        -------
        pd.DataFrame
            DataFrame avec les données du carnet d'ordres.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    ob = self.client.get_order_book(symbol=symbol, limit=limit)
                    df = pd.DataFrame(ob)
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )

            else:
                return pd.DataFrame(
                    self.client.get_order_book(symbol=symbols, limit=limit)
                )
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres: {e}")
            return pd.DataFrame()

    def get_recent_trades(
        self, symbols: Union[str, List[str]], limit: int = 500
    ) -> pd.DataFrame:
        """
        Récupère les trades récents pour un ou plusieurs symboles.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        limit : int, optional
            Nombre de trades à récupérer (max 1000, 500 par défaut)

        Returns
        -------
        pd.DataFrame
            DataFrame des trades récents.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    df = pd.DataFrame(
                        self.client.get_recent_trades(symbol=symbol, limit=limit)
                    )
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                return pd.DataFrame(
                    self.client.get_recent_trades(symbol=symbols, limit=limit)
                )
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération des trades récents : {e}")
            return pd.DataFrame()

    def get_historical_trades(
        self, symbols: Union[str, List[str]], limit: int = 500
    ) -> pd.DataFrame:
        """
        Récupère les trades historiques pour un ou plusieurs symboles.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        limit : int, optional
            Nombre de trades à récupérer (max 1000, 500 par défaut)

        Returns
        -------
        pd.DataFrame
            DataFrame des trades historiques.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    df = pd.DataFrame(
                        self.client.get_historical_trades(symbol=symbol, limit=limit)
                    )
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                return pd.DataFrame(
                    self.client.get_historical_trades(symbol=symbols, limit=limit)
                )
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération des trades historiques: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Méthodes pour l'information de compte
    # -------------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """
        Récupère les informations du compte Binance.

        Returns
        -------
        dict
            Dictionnaire contenant les informations du compte.
        """
        try:
            return self.client.get_account()
        except BinanceAPIException as e:
            logger.error(
                f"Erreur lors de la récupération des informations du compte : {e}"
            )
            return {}

    def get_position(
        self, symbols: Union[str, List[str]]
    ) -> Union[pd.DataFrame, dict, None]:
        """
        Récupère la (les) position(s) pour un ou plusieurs symboles.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')

        Returns
        -------
        pd.DataFrame ou dict ou None
            Positions pour les symboles spécifiés.
        """
        try:
            positions = self.client.get_account()["balances"]
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
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
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération de la position: {e}")
            return None

    # -------------------------------------------------------------------------
    # Méthodes pour des actions de trading
    # -------------------------------------------------------------------------

    def place_order(
        self,
        symbols: Union[str, List[str]],
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = TIME_IN_FORCE_GTC,
    ) -> Union[dict, None]:
        """
        Passe un ordre (ou plusieurs) avec les paramètres spécifiés.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        side : str
            'BUY' ou 'SELL'
        order_type : str
            Type d'ordre ('MARKET', 'LIMIT', 'STOP_LOSS_LIMIT', etc.)
        quantity : float
            Quantité à acheter ou vendre
        price : float, optional
            Prix limite (pour LIMIT ou STOP_LOSS_LIMIT)
        stop_price : float, optional
            Prix stop (pour STOP_LOSS_LIMIT)
        time_in_force : str, optional
            Validité de l'ordre ('GTC', 'IOC', 'FOK'), GTC par défaut

        Returns
        -------
        dict ou None
            Informations de l'ordre ou None en cas d'erreur.
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
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Erreur lors du passage de l'ordre : {e}")
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
        Méthode interne pour passer un ordre sur un seul symbole.
        """
        if order_type == ORDER_TYPE_MARKET:
            return self.client.create_order(
                symbol=symbol, side=side, type=order_type, quantity=quantity
            )
        elif order_type == ORDER_TYPE_LIMIT:
            return self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                timeInForce=time_in_force,
                quantity=quantity,
                price=str(price),
            )
        elif order_type == ORDER_TYPE_STOP_LOSS_LIMIT:
            return self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                timeInForce=time_in_force,
                quantity=quantity,
                price=str(price),
                stopPrice=str(stop_price),
            )
        else:
            raise ValueError("Type d'ordre non supporté.")

    def get_order_status(
        self, symbols: Union[str, List[str]], order_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Vérifie l'état d'un ordre.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        order_id : int
            Identifiant de l'ordre

        Returns
        -------
        pd.DataFrame ou None
            DataFrame avec l'état de l'ordre ou None en cas d'erreur.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    df = pd.DataFrame(
                        [self.client.get_order(symbol=symbol, orderId=order_id)]
                    )
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                return pd.DataFrame(
                    [self.client.get_order(symbol=symbols, orderId=order_id)]
                )
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Erreur lors de la récupération de l'état de l'ordre : {e}")
            return None

    def cancel_order(
        self, symbols: Union[str, List[str]], order_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Annule un ordre.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        order_id : int
            Identifiant de l'ordre à annuler

        Returns
        -------
        pd.DataFrame ou None
            DataFrame avec les informations de l'annulation ou None en cas d'erreur.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    df = pd.DataFrame(
                        [self.client.cancel_order(symbol=symbol, orderId=order_id)]
                    )
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                return pd.DataFrame(
                    [self.client.cancel_order(symbol=symbols, orderId=order_id)]
                )
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre : {e}")
            return None

    def get_open_orders(
        self, symbols: Optional[Union[str, List[str]]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Récupère tous les ordres ouverts pour un ou plusieurs symboles.
        Si aucun symbole n'est spécifié, récupère tous les ordres ouverts.

        Parameters
        ----------
        symbols : str ou list of str, optional
            Le ou les symboles (ex: 'BTCUSDT'). Si None, récupère tous les ordres ouverts.

        Returns
        -------
        pd.DataFrame ou None
            DataFrame avec les ordres ouverts ou None en cas d'erreur.
        """
        try:
            if symbols:
                if isinstance(symbols, list):
                    data_frames = []
                    for symbol in symbols:
                        df = pd.DataFrame(self.client.get_open_orders(symbol=symbol))
                        data_frames.append((symbol, df))
                    return pd.concat(
                        [df for _, df in data_frames],
                        keys=[sym for sym, _ in data_frames],
                    )
                else:
                    return pd.DataFrame(self.client.get_open_orders(symbol=symbols))
            else:
                return pd.DataFrame(self.client.get_open_orders())
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts : {e}")
            return None

    # -------------------------------------------------------------------------
    # Méthodes pour des fonctionnalités supplémentaires
    # -------------------------------------------------------------------------

    def get_all_tickers(self) -> pd.DataFrame:
        """
        Récupère les prix actuels de tous les tickers.

        Returns
        -------
        pd.DataFrame
            DataFrame avec les prix actuels de tous les tickers.
        """
        try:
            return pd.DataFrame(self.client.get_all_tickers())
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération des tickers : {e}")
            return pd.DataFrame()

    def get_my_trades(
        self, symbols: Union[str, List[str]], limit: int = 50
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les trades du compte utilisateur pour un ou plusieurs symboles.

        Parameters
        ----------
        symbols : str ou list of str
            Le ou les symboles (ex: 'BTCUSDT')
        limit : int, optional
            Nombre de trades à récupérer (max 1000), 50 par défaut

        Returns
        -------
        pd.DataFrame ou None
            DataFrame des trades du compte ou None en cas d'erreur.
        """
        try:
            if isinstance(symbols, list):
                data_frames = []
                for symbol in symbols:
                    df = pd.DataFrame(
                        self.client.get_my_trades(symbol=symbol, limit=limit)
                    )
                    data_frames.append((symbol, df))
                return pd.concat(
                    [df for _, df in data_frames], keys=[sym for sym, _ in data_frames]
                )
            else:
                return pd.DataFrame(
                    self.client.get_my_trades(symbol=symbols, limit=limit)
                )
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Erreur lors de la récupération des trades du compte : {e}")
            return None

    def get_deposit_history(self) -> Optional[pd.DataFrame]:
        """
        Récupère l'historique des dépôts.

        Returns
        -------
        pd.DataFrame ou None
            DataFrame de l'historique des dépôts ou None en cas d'erreur.
        """
        try:
            return pd.DataFrame(self.client.get_deposit_history())
        except BinanceAPIException as e:
            logger.error(
                f"Erreur lors de la récupération de l'historique des dépôts : {e}"
            )
            return None

    def get_withdraw_history(self) -> Optional[pd.DataFrame]:
        """
        Récupère l'historique des retraits.

        Returns
        -------
        pd.DataFrame ou None
            DataFrame de l'historique des retraits ou None en cas d'erreur.
        """
        try:
            return pd.DataFrame(self.client.get_withdraw_history())
        except BinanceAPIException as e:
            logger.error(
                f"Erreur lors de la récupération de l'historique des retraits : {e}"
            )
            return None

    def get_asset_detail(self) -> Optional[pd.DataFrame]:
        """
        Récupère les détails des actifs (info sur dépôts, retraits, etc.).

        Returns
        -------
        pd.DataFrame ou None
            DataFrame des détails des actifs ou None en cas d'erreur.
        """
        try:
            return pd.DataFrame(self.client.get_asset_details())
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération des détails des actifs : {e}")
            return None

    def get_trade_fee(
        self, symbols: Optional[Union[str, List[str]]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les frais de trading pour un ou plusieurs symboles.
        Si aucun symbole n'est spécifié, récupère les frais pour tous.

        Parameters
        ----------
        symbols : str ou list of str, optional
            Le ou les symboles (ex: 'BTCUSDT'). Si None, récupère les frais pour tous les symboles.

        Returns
        -------
        pd.DataFrame ou None
            DataFrame des frais de trading ou None en cas d'erreur.
        """
        try:
            if symbols:
                if isinstance(symbols, list):
                    data_frames = []
                    for symbol in symbols:
                        fee = self.client.get_trade_fee(symbol=symbol)
                        data_frames.append((symbol, pd.DataFrame([fee])))
                    return pd.concat(
                        [df for _, df in data_frames],
                        keys=[sym for sym, _ in data_frames],
                    )
                else:
                    fee = self.client.get_trade_fee(symbol=symbols)
                    return pd.DataFrame([fee])
            else:
                return pd.DataFrame(self.client.get_trade_fee())
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération des frais de trading : {e}")
            return None
