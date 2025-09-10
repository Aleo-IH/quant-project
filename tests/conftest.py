import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend non-interactif pour les tests
import pytest

# S'assure que 'src' est importable si lancé localement
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(os.path.dirname(ROOT), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def small_ohlcv_df():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "sample_ohlcv.csv")
    )
    df.index = pd.RangeIndex(len(df))
    return df


@pytest.fixture
def trending_close_df():
    # close croissant, + colonnes numériques pour l'env
    n = 40
    close = np.linspace(100, 120, n)
    df = pd.DataFrame(
        {
            "close": close,
            "ATRr_14": np.random.rand(n),
            "Number of Trades": np.random.randint(10, 100, n),
            "volume": np.random.rand(n) * 1000,
        }
    )
    return df


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)
