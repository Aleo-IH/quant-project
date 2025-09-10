import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

"""Ensure the project 'src' directory is on sys.path for src-layout."""
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def simple_ohlcv_df() -> pd.DataFrame:
    # Deterministic synthetic OHLCV with monotonic increasing close
    n = 80
    idx = pd.RangeIndex(n)
    close = 100 * (1.01 ** np.arange(n))
    open_ = close * 1.0
    high = close * 1.01
    low = close * 0.99
    volume = np.linspace(1000, 2000, n)
    df = pd.DataFrame(
        {
            "Open": open_.astype(float),
            "High": high.astype(float),
            "Low": low.astype(float),
            "Close": close.astype(float),
            "Volume": volume.astype(float),
            # Non-numeric and extra columns should be ignored by loaders
            "Symbol": ["XYZ"] * n,
        },
        index=idx,
    )
    return df


@pytest.fixture()
def tmp_pkls_dir(tmp_path: Path, simple_ohlcv_df: pd.DataFrame) -> Path:
    # Create a directory with multiple pickle files to exercise concatenation
    p = tmp_path / "pkls"
    p.mkdir(parents=True, exist_ok=True)
    # Split df in two parts with overlapping index to test deduplication
    split = len(simple_ohlcv_df) // 2
    df1 = simple_ohlcv_df.iloc[: split + 5]
    df2 = simple_ohlcv_df.iloc[split:]
    df1.to_pickle(p / "part1.pkl")
    df2.to_pickle(p / "part2.pkl")
    return p


@pytest.fixture()
def tmp_single_pkl(tmp_path: Path, simple_ohlcv_df: pd.DataFrame) -> Path:
    fp = tmp_path / "one.pkl"
    simple_ohlcv_df.to_pickle(fp)
    return fp
