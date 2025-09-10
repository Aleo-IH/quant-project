import numpy as np
import pandas as pd
from src.processes.Permutations import bar_permutation


def _toy_ohlc(n=30):
    # série simple cohérente OHLC
    open_ = np.linspace(100, 101, n)
    high = open_ + 0.5
    low = open_ - 0.5
    close = open_ + 0.2
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def test_bar_permutation_basic_properties():
    df = _toy_ohlc(40)
    perm = bar_permutation(df, start_index=5, seed=123)
    assert list(perm.columns) == ["Open", "High", "Low", "Close"]
    assert len(perm) == len(df)
    # la partie avant start_index inchangée
    pd.testing.assert_frame_equal(
        df.iloc[:5].reset_index(drop=True).round(10),
        perm.iloc[:5].reset_index(drop=True).round(10),
    )
    # pas de NaN
    assert not perm.isna().any().any()


def test_bar_permutation_multi_market():
    df1 = _toy_ohlc(20)
    df2 = _toy_ohlc(20) + 5
    out = bar_permutation([df1, df2], start_index=2, seed=7)
    assert isinstance(out, list) and len(out) == 2
