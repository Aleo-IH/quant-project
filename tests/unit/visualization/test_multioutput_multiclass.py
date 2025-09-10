import numpy as np
import pandas as pd
from src.visualization.multioutput_multiclass import (
    plot_distributions,
    plot_confusion_matrices,
    classif_report,
    plot_cross_validation_metrics,
    plot_bin_precision,
)


def _fake_y(n=50, k=3, bins=5):
    cols = [f"Target_{i}" for i in range(k)]
    y_true = pd.DataFrame({c: np.random.randint(0, bins, size=n) for c in cols})
    y_pred = pd.DataFrame({c: np.random.randint(0, bins, size=n) for c in cols})
    return y_true, y_pred


def test_plots_run_without_error():
    y_true, y_pred = _fake_y()
    plot_distributions(y_true, y_pred)
    plot_confusion_matrices(y_true, y_pred)
    classif_report(y_true, y_pred)  # imprime simplement
    cv_results = [(y_true, y_pred), (y_true, y_pred)]
    plot_cross_validation_metrics(cv_results)
    plot_bin_precision(cv_results, n_bins=5)
