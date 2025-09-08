import pandas_ta as ta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from tqdm.rich import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from src.processes.Processes_101 import add_returns_targets, getXy


def _technical_strategy() -> ta.Strategy:
    """Return a lightweight TA strategy with RSI/SMA/EMA/ATR."""

    class TechnicalStrategy(ta.Strategy):
        def __init__(self):
            super().__init__(name="technical_indicators")
            self.ta = [
                {"kind": "rsi", "length": 14},
                {"kind": "sma", "length": 20},
                {"kind": "ema", "length": 20},
                {"kind": "atr", "length": 14},
            ]

    return TechnicalStrategy()


def load_and_prepare_data(params: dict, target_threshold: float = 0.0005) -> pd.DataFrame:
    """Load price data, drop unused cols, add TA, and base target."""
    df = pd.read_pickle("./src/data/" + "_".join(params.values()) + ".pkl")

    # Drop heavy unused columns if present
    drop_cols = [
        "Quote Asset Volume",
        "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume",
    ]
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)

    # Limit period early to reduce compute
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.loc[:"2024-06-01"]

    # Technical indicators
    df.ta.strategy(_technical_strategy())

    # Base next-bar classification target
    df["Target"] = (df["Close"].pct_change().shift(-1) > target_threshold).astype(int)

    # Extra targets for getXy helper
    df = add_returns_targets(df, [1, 2, 3, 5, 8, 13])
    return df.dropna()


def train_meta_on_fold(
    rf_model: RandomForestClassifier,
    df_na: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    holding_period: int,
):
    """Train RF on train fold, fit meta model, and return test df with probs and accuracy."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Prepare test frame and primary probabilities
    df_test = df_na.iloc[test_idx].copy()
    test_probs = rf_model.predict_proba(X_test)[:, 1]
    df_test["Primary_Prob"] = test_probs

    # Meta training frame
    df_train = df_na.iloc[train_idx].copy()
    df_train["Primary_Prob"] = rf_model.predict_proba(X_train)[:, 1]
    df_train["Meta_Target"] = (
        df_train["Close"].shift(-holding_period) / df_train["Close"] > 1.001
    ).astype(int)

    meta_features_train = pd.DataFrame(
        {
            "Primary_Prob": df_train["Primary_Prob"],
            "ATRr_14": df_train["ATRr_14"],
        }
    ).dropna()

    # Not enough meta training samples
    if len(meta_features_train) < 10:
        return df_test, acc

    meta_target_train = df_train["Meta_Target"].loc[meta_features_train.index]

    meta_model = LogisticRegression(max_iter=200)
    meta_model.fit(meta_features_train, meta_target_train)

    meta_features_test = pd.DataFrame(
        {"Primary_Prob": test_probs, "ATRr_14": df_test["ATRr_14"]}
    )
    df_test["Meta_Prob"] = meta_model.predict_proba(meta_features_test)[:, 1]
    return df_test, acc


def cross_val_meta_probabilities(
    df_na: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    holding_period: int,
):
    """Run time series CV, return concatenated test folds and accuracy list."""
    console = Console()
    console.print("[bold green]Starting TimeSeries Cross-Validation Training...[/bold green]")

    # Retrain weekly: 1 week of minutes ~ 10080
    nb_weeks = max(2, len(df_na) // 10080)
    tscv = TimeSeriesSplit(n_splits=nb_weeks)
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, min_samples_leaf=25, n_jobs=-1
    )

    cv_accuracies = []
    test_dfs = []

    for train_idx, test_idx in tqdm(tscv.split(X), total=tscv.get_n_splits(), desc="CV Folds"):
        df_test, acc = train_meta_on_fold(
            rf_model, df_na, X, y, train_idx, test_idx, holding_period
        )
        cv_accuracies.append(acc)
        console.log(f"Fold accuracy: {acc:.2%}")
        test_dfs.append(df_test)

    avg_acc = float(np.mean(cv_accuracies)) if cv_accuracies else float("nan")
    console.print(f"[bold blue]Average CV Accuracy: {avg_acc*100:.2f}%[/bold blue]")
    return pd.concat(test_dfs).sort_index(), cv_accuracies


def compute_signals(df_out: pd.DataFrame, signal_threshold: float, holding_period: int) -> pd.DataFrame:
    """Create primary and meta signals and associated returns in-place."""
    df_out["Primary_Signal"] = (df_out["Primary_Prob"] > signal_threshold).astype(int)
    df_out["Primary_Strategy_Log_Return"] = df_out["Primary_Signal"] * np.log(
        df_out["Close"].shift(-holding_period) / df_out["Close"]
    )

    if "Meta_Prob" in df_out:
        df_out["Meta_Signal"] = (df_out["Meta_Prob"] > signal_threshold).astype(int)
        df_out["Strategy_Log_Return"] = df_out["Meta_Signal"] * np.log(
            df_out["Close"].shift(-holding_period) / df_out["Close"]
        )
    else:
        # Fallback if meta wasn't trained on some folds
        df_out["Meta_Signal"] = 0
        df_out["Strategy_Log_Return"] = 0.0

    df_out["Log_Return"] = np.log(df_out["Close"].shift(-holding_period) / df_out["Close"])
    df_out["Meta_Target"] = (
        df_out["Close"].shift(-holding_period) / df_out["Close"] > 1.001
    ).astype(int)
    return df_out


def summarize_metrics(df_out: pd.DataFrame, signal_threshold: float):
    """Compute summary metrics needed for print and plots."""
    # Returns and Sharpe
    ret_meta = float(df_out["Strategy_Log_Return"].sum())
    ret_primary = float(df_out["Primary_Strategy_Log_Return"].sum())

    meta_mean = float(df_out["Strategy_Log_Return"].mean())
    meta_std = float(df_out["Strategy_Log_Return"].std())
    sharpe_meta = meta_mean / meta_std if meta_std != 0 else float("nan")

    primary_mean = float(df_out["Primary_Strategy_Log_Return"].mean())
    primary_std = float(df_out["Primary_Strategy_Log_Return"].std())
    sharpe_primary = primary_mean / primary_std if primary_std != 0 else float("nan")

    # Classification metrics
    idx = df_out["Meta_Target"].dropna().index
    meta_preds = (
        (df_out["Meta_Prob"].loc[idx] > signal_threshold).astype(int)
        if "Meta_Prob" in df_out
        else pd.Series(0, index=idx)
    )
    primary_preds = (df_out["Primary_Prob"].loc[idx] > signal_threshold).astype(int)

    meta_report = classification_report(
        df_out["Meta_Target"].loc[idx], meta_preds, output_dict=True
    )
    primary_report = classification_report(
        df_out["Meta_Target"].loc[idx], primary_preds, output_dict=True
    )

    meta_roc = roc_auc_score(
        df_out["Meta_Target"].loc[idx],
        df_out.get("Meta_Prob", pd.Series(0.5, index=idx)).loc[idx],
    )
    primary_roc = roc_auc_score(
        df_out["Meta_Target"].loc[idx], df_out["Primary_Prob"].loc[idx]
    )

    return {
        "cumulative_return_meta": ret_meta,
        "cumulative_return_primary": ret_primary,
        "sharpe_meta": sharpe_meta,
        "sharpe_primary": sharpe_primary,
        "meta_report": meta_report,
        "primary_report": primary_report,
        "meta_roc": float(meta_roc),
        "primary_roc": float(primary_roc),
    }


def plot_all(df_out: pd.DataFrame, metrics: dict, signal_threshold: float):
    """Plot ROC curves, confusion matrices, metrics bars, Sharpe and cumulative returns."""
    idx = df_out["Meta_Target"].dropna().index

    # ROC curves
    fpr_meta, tpr_meta, _ = roc_curve(
        df_out["Meta_Target"].loc[idx], df_out["Meta_Prob"].loc[idx]
    )
    fpr_primary, tpr_primary, _ = roc_curve(
        df_out["Meta_Target"].loc[idx], df_out["Primary_Prob"].loc[idx]
    )
    roc_auc_meta = auc(fpr_meta, tpr_meta)
    roc_auc_primary = auc(fpr_primary, tpr_primary)

    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr_meta,
        tpr_meta,
        color="darkorange",
        lw=2,
        label=f"Meta ROC (AUC = {roc_auc_meta:.2f})",
    )
    plt.plot(
        fpr_primary,
        tpr_primary,
        color="blue",
        lw=2,
        label=f"Primary ROC (AUC = {roc_auc_primary:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # Confusion matrices
    meta_preds = (df_out["Meta_Prob"].loc[idx] > signal_threshold).astype(int)
    primary_preds = (df_out["Primary_Prob"].loc[idx] > signal_threshold).astype(int)
    meta_cm = confusion_matrix(df_out["Meta_Target"].loc[idx], meta_preds)
    primary_cm = confusion_matrix(df_out["Meta_Target"].loc[idx], primary_preds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(meta_cm, cmap="Blues")
    axes[0].set_title("Meta Confusion Matrix")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    for (i, j), label in np.ndenumerate(meta_cm):
        axes[0].text(
            j,
            i,
            label,
            ha="center",
            va="center",
            color="white" if label > meta_cm.max() / 2 else "black",
        )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    axes[1].imshow(primary_cm, cmap="Greens")
    axes[1].set_title("Primary Confusion Matrix")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    for (i, j), label in np.ndenumerate(primary_cm):
        axes[1].text(
            j,
            i,
            label,
            ha="center",
            va="center",
            color="white" if label > primary_cm.max() / 2 else "black",
        )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Classification metrics bar chart (positive class)
    metrics_names = ["precision", "recall", "f1-score"]
    meta_metrics = [metrics["meta_report"]["1"][m] for m in metrics_names]
    primary_metrics = [metrics["primary_report"]["1"][m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, meta_metrics, width, label="Meta", color="magenta")
    plt.bar(x + width / 2, primary_metrics, width, label="Primary", color="orange")
    plt.xticks(x, metrics_names)
    plt.ylabel("Score")
    plt.title("Classification Metrics (Positive Class)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Sharpe comparison
    plt.figure(figsize=(6, 6))
    plt.bar(
        ["Meta Strategy", "Primary Strategy"],
        [metrics["sharpe_meta"], metrics["sharpe_primary"]],
        color=["magenta", "orange"],
    )
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe Ratio Comparison")
    ymax = max(1e-9, metrics["sharpe_meta"], metrics["sharpe_primary"])
    plt.ylim(0, ymax * 1.2)
    plt.grid(alpha=0.3)
    plt.show()

    # Cumulative returns
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    df_out["Strategy_Log_Return"].cumsum().plot(
        color="magenta", label="Meta Strategy Returns"
    )
    df_out["Primary_Strategy_Log_Return"].cumsum().plot(
        color="orange", label="Primary Strategy Returns"
    )
    df_out["Log_Return"].cumsum().plot(color="cyan", label="Market Returns")
    plt.title("Cumulative Log Returns (Out-of-Sample)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


def main():
    # Config
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "start_dt": "2024-01-01",
        "end_dt": "2024-12-31",
    }
    threshold = 0.0005
    holding_period = 5
    signal_threshold = 0.60

    # Data
    df_na = load_and_prepare_data(params, target_threshold=threshold)
    X, y_df = getXy(df_na)
    y = y_df["Target"]

    # CV + Meta probabilities
    df_out, _ = cross_val_meta_probabilities(df_na, X, y, holding_period)

    # Signals and metrics
    df_out = compute_signals(df_out, signal_threshold, holding_period)
    m = summarize_metrics(df_out, signal_threshold)

    # Plots
    plot_all(df_out, m, signal_threshold)

    # Summary
    print("=== Strategy Performance ===")
    print(f"Cumulative Return (Meta Strategy): {m['cumulative_return_meta']:.4f}")
    print(
        f"Cumulative Return (Primary Strategy): {m['cumulative_return_primary']:.4f}"
    )
    print(f"Sharpe Ratio (Meta Strategy): {m['sharpe_meta']:.4f}")
    print(f"Sharpe Ratio (Primary Strategy): {m['sharpe_primary']:.4f}")
    print("\n=== Classification Metrics ===")
    print("Meta Model Classification Report:")
    print(m["meta_report"])  # dict
    print("\nPrimary Model Classification Report:")
    print(m["primary_report"])  # dict
    print("\nROC AUC:")
    print(f"Meta Model ROC AUC: {m['meta_roc']:.4f}")
    print(f"Primary Model ROC AUC: {m['primary_roc']:.4f}")


if __name__ == "__main__":
    main()

