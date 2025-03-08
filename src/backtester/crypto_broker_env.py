import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque


class CryptoTradingEnv(gym.Env):
    """
    Environnement de trading de cryptomonnaies pour entraînement RL.
    Cet environnement est compatible avec Stable Baselines3.

    L'agent reçoit à chaque pas de temps :
      - Un vecteur d'observation qui contient les colonnes numériques du DataFrame.
      - Une récompense calculée en fonction de l'évolution du PnL (Profit and Loss)
        ou d'un autre critère défini par l'utilisateur.
      - Une possibilité de prendre une action discrète parmi n niveaux de positions.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données de marché (historique ou en direct).
        Les colonnes doivent être numériques.
    column_names : list
        Liste des noms de colonnes à utiliser dans l'observation. Toutes doivent
        être de type numérique (float, int).
    n_positions : int
        Nombre de positions discrètes possibles (p. ex. 20 : 0.0, 0.05, 0.1, ..., 1.0
        si on veut coder une répartition de 0% à 100% en crypto).
    initial_balance : float
        Balance initiale du compte de trading pour calculer le PnL.
    max_steps : int
        Nombre maximum de pas de temps avant que l'épisode se termine
        (peut être len(df) ou un nombre inférieur).
    sharpe_length : int
        Période de calcul du ratio de Sharpe (ex : 30 jours).
    transaction_fee : float
        Frais de transaction en fraction du montant tradé (ex : 0.0001 = 0.01%).
    trade_threshold : float
        Seuil minimal en dollars du montant à échanger pour exécuter réellement
        la transaction (sinon on ignore le trade).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        column_names: list,
        n_positions: int = 20,
        initial_balance: float = 1000.0,
        max_steps: int = None,
        sharpe_length: int = 30,
        transaction_fee: float = 0.0001,
        trade_threshold: float = 1.0,
    ):
        super().__init__()

        # Conserver le DataFrame original
        self.df = df.reset_index(drop=True)
        self.column_names = column_names

        # Paramètres
        self.n_positions = n_positions
        self.initial_balance = initial_balance
        self.max_steps = max_steps if max_steps is not None else len(self.df) - 1
        self.sharpe_length = sharpe_length
        self.transaction_fee = transaction_fee
        self.trade_threshold = trade_threshold

        # Espace d'actions : un entier [0..n_positions-1]
        self.action_space = spaces.Discrete(self.n_positions)

        # Espace d'observation :
        #  -> len(self.column_names) colonnes de marché
        #  -> + 1 pour la balance ou le total
        #  -> + 1 pour la répartition en BTC
        # => total = len(column_names) + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.column_names) + 2,),
            dtype=np.float32,
        )

        # Variables internes
        self.current_step = 0

        # positions_base : [quantité de BTC, quantité de USDT]
        self.positions_base = None
        # position_quote : [valeur en $ de BTC, valeur en $ d'USDT]
        self.position_quote = None
        # repartition : [part BTC, part USDT]
        self.repartition = None

        # intended_repartition : la part BTC voulue (entre 0 et 1)
        self.intended_repartition = 0.0
        self.balance = self.initial_balance
        self.terminated = False
        self.truncated = False

        # Historique
        self.trades_history = []
        self.past_action = None

        # Tracking Sharpe ratio
        self.daily_returns = deque(maxlen=self.sharpe_length)
        self.rolling_sharpe_ratio = 0.0

        # Appel du reset pour initialiser toutes les variables
        self.reset()

    def _update_position_quote(self):
        """Met à jour la valeur de chaque composante (BTC, USDT) en dollars."""
        current_price = self.df["close"].iloc[self.current_step]
        # positions_base = [ BTC_amount, USDT_amount ]
        # On convertit la partie BTC en dollars selon le prix de marché
        self.position_quote = self.positions_base * np.array([current_price, 1.0])

    def _update_repartition(self):
        """Calcule la répartition (en %) entre BTC et USDT, sur base de la valeur en $."""
        total_value = np.sum(self.position_quote)
        if total_value > 0:
            self.repartition = self.position_quote / total_value
        else:
            # En cas de balance 0 ou négative, on met 0,0 par défaut
            self.repartition = np.array([0.0, 0.0])

    def _get_observation(self):
        """
        Récupère l'observation courante depuis le DataFrame +
        la balance et la répartition courante en BTC.
        """
        if self.terminated or self.truncated:
            # Si l'épisode est fini, on peut retourner un vecteur vide
            # ou répéter la dernière observation.
            obs = np.zeros((len(self.column_names) + 2,), dtype=np.float32)
            return obs

        # Données marché (colonnes numériques)
        obs_df = (
            self.df[self.column_names].iloc[self.current_step].values.astype(np.float32)
        )

        # On ajoute la balance (ou sum(position_quote)) et la proportion BTC
        obs_bot = np.array(
            [
                self.balance,  # Montant total
                self.repartition[0],  # Part en BTC (0..1)
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([obs_df, obs_bot])
        return obs

    def _calculate_reward(self, old_balance, new_balance):
        """
        Exemple : ratio de Sharpe sur la fenêtre self.sharpe_length.
        Pour la première transaction, on renvoie le simple ratio (PnL %).
        """
        # Variation journalière ou step-based
        daily_return = (new_balance / old_balance) - 1.0
        self.daily_returns.append(daily_return)

        if len(self.daily_returns) >= 2:
            mean_ret = np.mean(self.daily_returns)
            std_ret = np.std(self.daily_returns)
            if std_ret > 0:
                reward = mean_ret / std_ret
                self.rolling_sharpe_ratio = reward
            else:
                # Si la volatilité est nulle, on évite la division par zéro
                reward = mean_ret
                self.rolling_sharpe_ratio = reward
        else:
            # Peu de données => on se contente du return
            reward = daily_return
            self.rolling_sharpe_ratio = reward

        return reward

    def _execute_trade(self, action):
        """
        Exécute la logique de mise à jour de la balance en fonction
        de l'action choisie et du prix, pour tendre vers la répartition voulue.
        """
        old_balance = self.balance
        current_price = self.df["close"].iloc[self.current_step]

        # Convertit l'action discrète en fraction désirée de BTC
        self.intended_repartition = action / (self.n_positions - 1)

        # Récupère la valeur totale courante (en $) et la part actuelle en BTC
        total_value = np.sum(self.position_quote)
        current_btc_value = self.position_quote[0]

        # Calcule la valeur désirée en BTC
        desired_btc_value = total_value * self.intended_repartition
        trade_value = desired_btc_value - current_btc_value  # en dollars

        # Si le montant à trader est inférieur au trade_threshold, on ignore
        if abs(trade_value) >= self.trade_threshold:
            # Frais de transaction
            cost = abs(trade_value) * self.transaction_fee

            if trade_value > 0:
                # On ACHÈTE du BTC : trade_value dollars de BTC
                btc_to_buy = trade_value / current_price
                # Mise à jour quantités
                self.positions_base[0] += btc_to_buy
                self.positions_base[1] -= trade_value + cost  # on retire les USDT
            else:
                # On VEND du BTC
                btc_to_sell = abs(trade_value) / current_price
                self.positions_base[0] -= btc_to_sell
                # On récupère du USDT, moins le coût
                self.positions_base[1] += abs(trade_value) - cost

            # Après le trade, on met à jour quote, repartition, balance
            self._update_position_quote()
            self._update_repartition()

        # Mettre à jour la balance (somme des deux valeurs en $)
        self.balance = np.sum(self.position_quote)
        new_balance = self.balance

        # Calcul de la récompense
        reward = self._calculate_reward(old_balance, new_balance)

        # Historique
        self.trades_history.append(
            {
                "step": self.current_step,
                "action": action,
                "intended_btc_part": self.intended_repartition,
                "old_balance": old_balance,
                "new_balance": new_balance,
                "reward": reward,
                "trade_value": trade_value,
                "positions_base": self.positions_base.copy(),
                "positions_quote": self.position_quote.copy(),
            }
        )
        return reward

    def step(self, action):
        """
        Un pas de simulation :
          1) Met à jour la position en fonction de l'action.
          2) Avance d'un step dans le DataFrame.
          3) Calcule la récompense.
          4) Détermine si l'épisode est terminé.
          5) Retourne (observation, reward, terminated, truncated, info).
        """
        # 1) Exécuter le trade
        reward = self._execute_trade(action)

        # 2) Avancer d'un pas
        self.current_step += 1

        # 3) Vérifier si on est arrivé à la fin
        if self.current_step >= self.max_steps:
            self.terminated = True

        # 4) Récupère l'observation
        obs = self._get_observation()

        # 5) Info pour debug
        info = {
            "balance": self.balance,
            "current_step": self.current_step,
            "current_repartition": self.repartition.copy(),
            "rolling_sharpe_ratio": self.rolling_sharpe_ratio,
        }

        return obs, reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'épisode de trading.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.terminated = False
        self.truncated = False

        self.trades_history = []
        self.daily_returns.clear()
        self.rolling_sharpe_ratio = 0.0

        # Réinitialise la balance et les positions
        self.balance = self.initial_balance

        # Au début, on met tout en USDT : [0 BTC, initial_balance USDT]
        self.positions_base = np.array([0.0, self.initial_balance], dtype=np.float32)
        self._update_position_quote()
        self._update_repartition()

        # Récupère l'observation initiale
        obs = self._get_observation()
        return obs, {}

    def render(self, mode="human"):
        """
        Rendu texte simple.
        """
        print(
            f"Step: {self.current_step}, "
            f"Balance: {self.balance:.2f}, "
            f"BTC part: {self.repartition[0]*100:.2f}%, "
            f"USDT part: {self.repartition[1]*100:.2f}%, "
            f"Sharpe: {self.rolling_sharpe_ratio:.4f}"
        )

    def close(self):
        """
        Nettoyage si nécessaire.
        """
        pass
