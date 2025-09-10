import sys
import types
import builtins
import numpy as np
import pandas as pd

from models.RL.env import TradingEnv


def _build_df(n=40):
    idx = pd.RangeIndex(n)
    close = 100 * (1.005 ** np.arange(n))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(500, 700, n),
        },
        index=idx,
    )
    return df


def test_render_text_fallback(monkeypatch):
    # Force matplotlib import to fail to exercise text fallback path
    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib for test")
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)

    env = TradingEnv(_build_df(), window=8)
    env.reset()
    env.step(2)
    # Should not raise; will print dict and return
    env.render()


def test_render_with_minimal_matplotlib_stub(monkeypatch):
    # Build a very small stub to satisfy calls in render
    class Canvas:
        def draw(self):
            pass

        def draw_idle(self):
            pass

        def flush_events(self):
            pass

        class manager:
            @staticmethod
            def set_window_title(title):
                return None

    class Ax:
        def __init__(self):
            self._data = []

        def set_ylabel(self, *_):
            pass

        def set_xlabel(self, *_):
            pass

        def axhline(self, *_, **__):
            pass

        def set_yticks(self, *_):
            pass

        def legend(self, *_, **__):
            pass

        def text(self, *_, **__):
            return types.SimpleNamespace(set_text=lambda *_: None)

        def plot(self, *_, **__):
            return (types.SimpleNamespace(set_data=lambda *_: None),)

        def step(self, *_, **__):
            return (types.SimpleNamespace(set_data=lambda *_: None),)

        def scatter(self, *_, **__):
            return types.SimpleNamespace(
                set_offsets=lambda *_: None, set_color=lambda *_: None
            )

        def set_xlim(self, *_):
            pass

        def set_ylim(self, *_):
            pass

        def relim(self):
            pass

        def autoscale_view(self, *_, **__):
            pass

    class Fig:
        def __init__(self):
            self.canvas = Canvas()

        def tight_layout(self):
            pass

    def subplots(nrows, ncols, *_, **__):
        assert (nrows, ncols) == (3, 1)
        return Fig(), (Ax(), Ax(), Ax())

    # matplotlib module stub
    matplotlib = types.SimpleNamespace(get_backend=lambda: "qt5agg")
    plt = types.SimpleNamespace(
        subplots=subplots,
        ion=lambda: None,
        show=lambda **_: None,
        pause=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)

    env = TradingEnv(_build_df(), window=8)
    env.reset()
    for _ in range(5):
        env.step(2)
    env.render()  # should go through matplotlib path without errors
