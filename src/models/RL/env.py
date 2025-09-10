from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple


class TradingEnv(gym.Env):
    """Fast single-asset trading environment with discrete positions.

    - Actions: {-1: short, 0: flat, +1: long} (encoded as {0, 1, 2}).
    - Observation: rolling window of simple market features from OHLCV.
    - Reward: position * log_return - fee * |Δposition|.
    """

    # Support both Gym (render.modes) and Gymnasium (render_modes)
    metadata = {"render.modes": ["human"], "render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 64,
        fee: float = 0.0005,
        reward_scale: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if len(df) < window + 2:
            raise ValueError("DataFrame is too short for the specified window.")

        # Normalize/standardize columns (case-insensitive) expected in the provided dataset
        cols = {c.lower(): c for c in df.columns}
        required = ["open", "high", "low", "close", "volume"]
        lowered = [c.lower() for c in df.columns]
        if not all(req in lowered for req in required):
            raise ValueError(
                f"DataFrame must include columns {required} (case-insensitive). Got: {list(df.columns)}"
            )

        # Build a normalized frame with required lowercase names
        self._df = pd.DataFrame(
            {
                "open": df[cols["open"]].astype(float).to_numpy(copy=False),
                "high": df[cols["high"]].astype(float).to_numpy(copy=False),
                "low": df[cols["low"]].astype(float).to_numpy(copy=False),
                "close": df[cols["close"]].astype(float).to_numpy(copy=False),
                "volume": df[cols["volume"]].astype(float).to_numpy(copy=False),
            }
        )

        # Precompute features (vectorized, float32) for speed.
        close = self._df["close"].to_numpy()
        open_ = self._df["open"].to_numpy()
        high = self._df["high"].to_numpy()
        low = self._df["low"].to_numpy()
        vol = self._df["volume"].to_numpy()

        # Align features with time index starting at 1 due to differencing
        self._features = self.get_features(close, open_, high, low, vol)
        self._close = close[1:]
        self._window = int(window)
        self._fee = float(fee)
        self._reward_scale = float(reward_scale)
        # Gymnasium compatibility: specify the chosen render mode
        self.render_mode: str | None = render_mode

        # Spaces
        self.action_space = spaces.Discrete(
            3
        )  # 0: short (-1), 1: flat (0), 2: long (+1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._window, self._features.shape[1]),
            dtype=np.float32,
        )

        # State
        self._t: int = 0
        self._pos: int = 0  # -1 / 0 / +1
        self._done_t: int = self._features.shape[0]  # number of feature rows
        # Stats for rendering/analysis
        self._cum_reward: float = 0.0
        self._cum_fee: float = 0.0
        self._trades: int = 0
        self._last_reward: float = 0.0
        self._last_action: int = 1  # encoded action (0,1,2). Default to flat
        self._cum_log_ret: float = (
            0.0  # cumulative position-weighted log return (unscaled)
        )
        self._realized_log_ret: float = 0.0  # realized portion when closing positions
        self._entry_price: Optional[float] = None
        self._render_header_printed: bool = False
        # Rendering state (matplotlib window)
        self._render_initialized: bool = False
        self._fig = None
        self._ax_price = None
        self._ax_pos = None
        self._ax_gain = None
        self._ln_price = None
        self._ln_pos = None
        self._ln_gain = None
        self._sc_long = None
        self._sc_short = None
        self._sc_close = None
        self._stat_text = None
        self._pos_series: Optional[np.ndarray] = None  # per-step position history
        self._equity_value: float = 100.0
        self._equity_series: Optional[np.ndarray] = (
            None  # per-step equity (gains) history
        )
        self._trade_long_idx: list[int] = []
        self._trade_short_idx: list[int] = []
        self._trade_close_idx: list[int] = []
        self._last_flat_marker_idx: Optional[int] = None
        self._render_tail: int = 600

    def get_features(self, close, open_, high, low, vol):
        log_ret = np.log((close[1:]) / (close[:-1]))
        hl_range = (high[1:] - low[1:]) / (close[:-1])
        oc_change = (close[1:] - open_[1:]) / (open_[1:])
        vol_chg = np.log((vol[1:] + 1.0) / (vol[:-1] + 1.0))

        features = np.stack([log_ret, hl_range, oc_change, vol_chg], axis=1).astype(
            np.float32
        )
        return features

    def _obs(self) -> np.ndarray:
        # Provide last `window` rows. If at beginning, pad from start.
        start = max(0, self._t - self._window)
        obs = self._features[start : self._t]
        if obs.shape[0] < self._window:
            pad = np.zeros(
                (self._window - obs.shape[0], self._features.shape[1]), dtype=np.float32
            )
            obs = np.vstack([pad, obs])
        return obs.astype(np.float32, copy=False)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Start after initial window to avoid excessive padding
        self._t = self._window
        self._pos = 0
        # Reset stats
        self._cum_reward = 0.0
        self._cum_fee = 0.0
        self._trades = 0
        self._last_reward = 0.0
        self._last_action = 1
        self._cum_log_ret = 0.0
        self._realized_log_ret = 0.0
        self._entry_price = None
        self._render_header_printed = False
        self._render_initialized = False
        self._pos_series = np.zeros_like(self._close, dtype=np.int8)
        self._equity_value = 100.0
        self._equity_series = np.full(self._close.shape, np.nan, dtype=float)
        self._trade_long_idx.clear()
        self._trade_short_idx.clear()
        self._trade_close_idx.clear()
        self._last_flat_marker_idx = None
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        new_pos = action - 1  # {-1,0,+1}
        prev_pos = self._pos

        # Instantaneous reward using log return from t-1 -> t
        ret = float(self._features[self._t - 1, 0])  # log_ret at t
        trans_cost = self._fee * abs(new_pos - prev_pos)
        reward = (new_pos * ret - trans_cost) * self._reward_scale

        # Update tracking stats before time advances
        self._cum_log_ret += new_pos * ret
        self._cum_fee += trans_cost
        self._cum_reward += reward
        self._last_reward = reward
        self._last_action = action

        # Handle entry/exit bookkeeping at the current market price
        cur_price = (
            float(self._close[self._t - 1])
            if self._t - 1 < len(self._close)
            else float("nan")
        )
        if new_pos != prev_pos:
            self._trades += 1
            # If we are closing or flipping, realize PnL on the previous leg
            if prev_pos != 0 and self._entry_price not in (None, 0.0):
                # Use log return for realization robustness
                self._realized_log_ret += prev_pos * float(
                    np.log((cur_price + 1e-12) / (self._entry_price + 1e-12))
                )
            # Set new entry price if we are opening a position
            self._entry_price = cur_price if new_pos != 0 else None
            # Record trade markers (at index t-1)
            idx = self._t - 1
            if prev_pos != 0 and new_pos == 0:
                # Entering flat: only mark once at the start of the flat run
                if self._last_flat_marker_idx != idx:
                    if not self._trade_close_idx or self._trade_close_idx[-1] != idx:
                        self._trade_close_idx.append(idx)
                    self._last_flat_marker_idx = idx
            elif prev_pos == 0 and new_pos != 0:
                # Leaving flat: reset flat marker tracker
                self._last_flat_marker_idx = None
                (self._trade_long_idx if new_pos > 0 else self._trade_short_idx).append(
                    idx
                )
            elif (
                prev_pos != 0 and new_pos != 0 and np.sign(prev_pos) != np.sign(new_pos)
            ):
                # flip: close then open
                # Do not mark a close-circle on flips to avoid repeated circles; only mark new open
                (self._trade_long_idx if new_pos > 0 else self._trade_short_idx).append(
                    idx
                )

        # Log position at this time index for plotting history
        self._pos = new_pos
        if self._t - 1 < (
            self._pos_series.shape[0] if self._pos_series is not None else 0
        ):
            self._pos_series[self._t - 1] = new_pos
        # Update gains/equity curve: multiply by exp(net_log_return)
        net_log = new_pos * ret - trans_cost
        try:
            self._equity_value *= float(np.exp(net_log))
        except Exception:
            pass
        if (
            self._equity_series is not None
            and self._t - 1 < self._equity_series.shape[0]
        ):
            self._equity_series[self._t - 1] = self._equity_value
        self._t += 1

        terminated = self._t >= self._done_t
        truncated = False
        info = {
            "position": self._pos,
            "return": ret,
            "t": self._t,
            "reward": self._last_reward,
            "cum_reward": self._cum_reward,
            "cum_fee": self._cum_fee,
            "cum_log_ret": self._cum_log_ret,
            "realized_log_ret": self._realized_log_ret,
            "trades": self._trades,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        """Open/update a live window with price, position, trades, and stats.

        - Top subplot: price with trade markers (green=long, red=short, blue=close).
        - Bottom subplot: position over time (-1/0/+1 step plot).
        - Right-side text: real-time stats (reward, fees, PnL, etc.).
        """
        # Validity check
        if self._t <= 0 or self._t - 1 >= len(self._close):
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            # Fallback to text if matplotlib not available
            print(
                {
                    "t": self._t,
                    "price": float(self._close[self._t - 1]),
                    "pos": int(self._pos),
                    "reward": float(self._last_reward),
                    "cum_reward": float(self._cum_reward),
                    "cum_fee": float(self._cum_fee),
                    "trades": int(self._trades),
                }
            )
            return

        # Lazy init the figure and artists
        if (
            (not self._render_initialized)
            or (self._fig is None)
            or (self._ax_gain is None)
        ):
            import matplotlib

            backend = str(matplotlib.get_backend()).lower()
            inline_backend = "inline" in backend or "nbagg" in backend
            if not inline_backend:
                plt.ion()
            self._fig, (self._ax_price, self._ax_pos, self._ax_gain) = plt.subplots(
                3, 1, sharex=True, figsize=(11, 9), dpi=100
            )
            self._fig.canvas.manager.set_window_title(
                "TradingEnv - Live Render"
            ) if hasattr(self._fig.canvas.manager, "set_window_title") else None
            self._ax_price.set_ylabel("Price")
            self._ax_pos.set_ylabel("Position")
            self._ax_pos.set_xlabel("t")
            self._ax_pos.axhline(0, color="#888", lw=0.8)
            self._ax_pos.set_yticks([-1, 0, 1])
            self._ax_gain.set_ylabel("Equity (Gains)")
            # Lines
            (self._ln_price,) = self._ax_price.plot(
                [], [], color="#1f77b4", lw=1.25, label="Close"
            )
            (self._ln_pos,) = self._ax_pos.step(
                [], [], color="#2ca02c", lw=1.25, where="post", label="Pos"
            )
            (self._ln_gain,) = self._ax_gain.plot(
                [], [], color="#ff7f0e", lw=1.25, label="Equity"
            )
            # Trades scatters
            self._sc_long = self._ax_price.scatter(
                [], [], marker="^", c="#2ca02c", s=35, label="Long open"
            )
            self._sc_short = self._ax_price.scatter(
                [], [], marker="v", c="#d62728", s=35, label="Short open"
            )
            self._sc_close = self._ax_price.scatter(
                [],
                [],
                marker="o",
                facecolors="none",
                edgecolors="#1f77b4",
                s=35,
                label="Close",
            )
            self._ax_price.legend(loc="upper left", fontsize=8, frameon=False)
            # Stats text on price axes
            self._stat_text = self._ax_price.text(
                1.02,
                0.98,
                "",
                transform=self._ax_price.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
            )
            self._fig.tight_layout()
            self._render_initialized = True

        # Determine tail window for speed
        t = self._t
        left = max(0, t - self._render_tail)
        x = np.arange(left, t)
        # Price
        y_price = np.asarray(self._close[left:t], dtype=float)
        self._ln_price.set_data(np.asarray(x, dtype=float), y_price)
        # Position series (if missing, create zeros)
        if self._pos_series is None or len(self._pos_series) != len(self._close):
            self._pos_series = np.zeros_like(self._close, dtype=np.int8)
        y_pos = self._pos_series[left:t]
        # Step requires x to include final point for 'post'; extend by one if possible
        if len(x) > 0:
            x_pos = np.append(x, x[-1] + 1)
            y_pos_step = np.append(y_pos, y_pos[-1])
        else:
            x_pos = x
            y_pos_step = y_pos
        self._ln_pos.set_data(
            np.asarray(x_pos, dtype=float), np.asarray(y_pos_step, dtype=float)
        )

        # Equity/gains series
        if self._equity_series is None or len(self._equity_series) != len(self._close):
            self._equity_series = np.full(self._close.shape, np.nan, dtype=float)
        y_gain = self._equity_series[left:t]
        self._ln_gain.set_data(
            np.asarray(x, dtype=float), np.asarray(y_gain, dtype=float)
        )

        # Update trades
        def _scatter_update(sc, idxs, color=None):
            if len(idxs) == 0:
                sc.set_offsets(np.empty((0, 2)))
                return
            # Deduplicate and sort to avoid rendering multiple markers on same bar
            xs = np.unique(np.array(idxs, dtype=int))
            xs = xs.astype(float)
            # Clip to window
            m = xs >= left
            xs = xs[m]
            if xs.size == 0:
                sc.set_offsets(np.empty((0, 2)))
                return
            ys = self._close[xs.astype(int)]
            sc.set_offsets(np.c_[xs, ys])
            if color is not None:
                sc.set_color(color)

        _scatter_update(self._sc_long, self._trade_long_idx, color="#2ca02c")
        _scatter_update(self._sc_short, self._trade_short_idx, color="#d62728")
        _scatter_update(self._sc_close, self._trade_close_idx, color="#1f77b4")

        # Axes limits
        if len(x) >= 1:
            x_right = x[-1] if x[-1] > x[0] else x[0] + 1
            self._ax_price.set_xlim(x[0], x_right)
            ymin, ymax = float(np.nanmin(y_price)), float(np.nanmax(y_price))
            if np.isfinite(ymin) and np.isfinite(ymax):
                pad = (ymax - ymin) * 0.05 + (1e-8)
                self._ax_price.set_ylim(ymin - pad, ymax + pad)
            # Keep pos axis in sync on x
            self._ax_pos.set_xlim(x[0], x_right)
        self._ax_pos.set_ylim(-1.5, 1.5)
        # Equity axis limits
        if len(x) >= 1:
            self._ax_gain.set_xlim(x[0], x_right)
            yg = y_gain[~np.isnan(y_gain)] if isinstance(y_gain, np.ndarray) else y_gain
            if len(yg) > 0:
                gmin, gmax = float(np.nanmin(yg)), float(np.nanmax(yg))
                if np.isfinite(gmin) and np.isfinite(gmax):
                    gpad = (gmax - gmin) * 0.05 + (1e-8)
                    if gpad == 0:
                        gpad = 1.0
                    self._ax_gain.set_ylim(gmin - gpad, gmax + gpad)

        # Let Matplotlib recompute limits from artists in case of edge cases
        try:
            self._ax_price.relim()
            self._ax_price.autoscale_view()
            self._ax_pos.relim()
            self._ax_pos.autoscale_view(scaley=False)
        except Exception:
            pass

        # Stats panel content
        price = float(self._close[self._t - 1])
        pos = int(self._pos)
        ret = (
            float(self._features[self._t - 1, 0])
            if self._t - 1 < len(self._features)
            else 0.0
        )
        equity = float(self._equity_value)
        cum_pct = (equity / 100.0 - 1.0) * 100.0
        realized_pct = (float(np.exp(self._realized_log_ret)) - 1.0) * 100.0
        if pos != 0 and self._entry_price not in (None, 0.0):
            unreal_log = pos * float(
                np.log((price + 1e-12) / (self._entry_price + 1e-12))
            )
            unreal_pct = (float(np.exp(unreal_log)) - 1.0) * 100.0
        else:
            unreal_pct = 0.0
        act_sym = {0: "Short", 1: "Flat", 2: "Long"}.get(int(self._last_action), "?")
        text = (
            f"t: {t}\n"
            f"price: {price:.4f}\n"
            f"pos: {pos:+d}  act: {act_sym}\n"
            f"ret(log): {ret:+.5f}\n"
            f"reward: {self._last_reward:+.6f}\n"
            f"cum_reward: {self._cum_reward:+.6f}\n"
            f"fees: {self._cum_fee:.6f}\n"
            f"trades: {self._trades}\n"
            f"equity: {equity:.4f}  cum%: {cum_pct:+.2f}%\n"
            f"unreal%: {unreal_pct:+.2f}%  realized%: {realized_pct:+.2f}%"
        )
        self._stat_text.set_text(text)

        # Draw and display depending on backend (GUI vs notebook inline)
        import matplotlib

        backend = str(matplotlib.get_backend()).lower()
        inline_backend = "inline" in backend or "nbagg" in backend

        # Always draw artists to update the figure state
        try:
            self._fig.canvas.draw()
        except Exception:
            self._fig.canvas.draw_idle()

        if inline_backend:
            # Jupyter inline update: clear output and display the updated figure
            try:
                from IPython.display import display, clear_output

                clear_output(wait=True)
                display(self._fig)
            except Exception:
                pass
            try:
                import matplotlib.pyplot as plt  # noqa: F401

                plt.pause(0.001)
            except Exception:
                pass
        else:
            # GUI backends: flush events to update the window without blocking
            try:
                self._fig.canvas.flush_events()
            except Exception:
                pass
