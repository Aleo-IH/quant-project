import numpy as np
import pandas as pd
from src.processes.Processes_101 import getXy, add_returns_targets


def test_getXy_splits_targets():
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0, 1],
            "Close": [1.5, 2.5],
            "feat1": [10, 11],
            "Target": [0, 1],
            "Target_5": [0.1, -0.2],
        }
    )
    X, y = getXy(df)
    assert "feat1" in X.columns and "Close" not in X.columns
    assert list(y.columns) == ["Target", "Target_5"]


def test_add_returns_targets():
    df = pd.DataFrame({"Close": [100, 101, 99, 102, 103]})
    out = add_returns_targets(df.copy(), [1, 2])
    assert "Return_1" in out.columns and "Return_2" in out.columns
    assert "Target_1" in out.columns and "Target_2" in out.columns
    # vérifie le calcul simple
    np.testing.assert_allclose(out["Return_1"].iloc[1], (101 - 100) / 100)
