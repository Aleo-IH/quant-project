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
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                             roc_auc_score)

from src.processes.Processes_101 import add_returns_targets, getXy

# Load data (adjust path if needed)
params = {
    "symbol": "BTCUSDT",
    "interval": "1m",
    "start_dt": "2024-01-01",
    "end_dt": "2024-12-31",
}

df = pd.read_pickle("./src/data/" + "_".join(params.values()) + ".pkl")
df.drop(
    columns=[
        "Quote Asset Volume",
        "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume",
    ],
    inplace=True,
)

df = df.loc[:"2024-06-01"]

class TechnicalStrategy(ta.Strategy):
    """
    A custom technical analysis strategy that calculates RSI, SMA, EMA, and ATR.
    """
    def __init__(self):
        super().__init__(name="technical_indicators")
        self.ta = [
            {"kind": "rsi", "length": 14},
            {"kind": "sma", "length": 20},
            {"kind": "ema", "length": 20},
            {"kind": "atr", "length": 14},
        ]

df.ta.strategy(TechnicalStrategy())

# Define target: 1 if the next return (using Close) exceeds 0.05%
threshold = 0.0005
df["Target"] = (df["Close"].pct_change().shift(-1) > threshold).astype(int)

# Add additional return targets and drop rows with NaNs
df = add_returns_targets(df, [1, 2, 3, 5, 8, 13])
df_na = df.dropna()

# Extract features and target for the primary model
X, y_df = getXy(df_na)
y = y_df["Target"]

holding_period = 5
signal_threshold = 0.60

# ------------------------------
# 2. Cross-Validation & Out-of-Sample Predictions
# ------------------------------
console = Console()
console.print("[bold green]Starting TimeSeries Cross-Validation Training...[/bold green]")

nb_weeks = len(df_na)//10080

tscv = TimeSeriesSplit(n_splits=nb_weeks) #Retrain the model every week
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=25)
cv_accuracies = []

# We'll collect out-of-sample test folds in this list.
test_dfs = []

for train_idx, test_idx in tqdm(tscv.split(X), total=tscv.get_n_splits(), desc="CV Folds"):
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train primary model on training fold
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)
    console.log(f"Fold accuracy: {acc:.2%}")
    
    # Create a copy of the test fold for out-of-sample predictions
    df_test = df_na.iloc[test_idx].copy()
    
    # Compute primary probabilities on test fold
    test_probs = rf_model.predict_proba(X_test)[:, 1]
    df_test["Primary_Prob"] = test_probs
    
    # ------------------------------
    # Meta-Model: Train on training fold and apply to test fold
    # ------------------------------
    df_train = df_na.iloc[train_idx].copy()
    df_train["Primary_Prob"] = rf_model.predict_proba(X_train)[:, 1]
    # Meta target: 1 if Close moves >0.1% in the next holding period
    df_train["Meta_Target"] = (df_train["Close"].shift(-holding_period) / df_train["Close"] > 1.001).astype(int)
    
    meta_features_train = pd.DataFrame({
        "Primary_Prob": df_train["Primary_Prob"],
        "ATRr_14": df_train["ATRr_14"]
    }).dropna()
    meta_target_train = df_train["Meta_Target"].loc[meta_features_train.index]
    
    if len(meta_features_train) < 10:
        continue  # Skip fold if insufficient data for meta model training
    
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, meta_target_train)
    
    meta_features_test = pd.DataFrame({
        "Primary_Prob": test_probs,
        "ATRr_14": df_test["ATRr_14"]
    })
    meta_probs = meta_model.predict_proba(meta_features_test)[:, 1]
    df_test["Meta_Prob"] = meta_probs
    
    test_dfs.append(df_test)

avg_acc = np.mean(cv_accuracies)
console.print(f"[bold blue]Average CV Accuracy: {avg_acc*100:.2f}%[/bold blue]")

# ------------------------------
# 3. Strategy Evaluation on Out-of-Sample Data
# ------------------------------
# Concatenate all out-of-sample test folds
df_out = pd.concat(test_dfs).sort_index()

# For comparison, create two sets of signals:
# Primary signal: threshold applied directly to primary probabilities
df_out["Primary_Signal"] = (df_out["Primary_Prob"] > signal_threshold).astype(int)
df_out["Primary_Strategy_Log_Return"] = df_out["Primary_Signal"] * np.log(df_out["Close"].shift(-holding_period) / df_out["Close"])

# Meta signal: threshold applied to meta probabilities
df_out["Meta_Signal"] = (df_out["Meta_Prob"] > signal_threshold).astype(int)
df_out["Strategy_Log_Return"] = df_out["Meta_Signal"] * np.log(df_out["Close"].shift(-holding_period) / df_out["Close"])

# Market returns (for reference)
df_out["Log_Return"] = np.log(df_out["Close"].shift(-holding_period) / df_out["Close"])

# Define the ground truth for classification: meta target computed on out-of-sample data
df_out["Meta_Target"] = (df_out["Close"].shift(-holding_period) / df_out["Close"] > 1.001).astype(int)

# ------------------------------
# 4. Compute Performance Metrics
# ------------------------------
# Cumulative Returns
cumulative_return_meta = df_out["Strategy_Log_Return"].sum()
cumulative_return_primary = df_out["Primary_Strategy_Log_Return"].sum()

# Sharpe Ratios (mean/std of strategy returns)
meta_mean = df_out["Strategy_Log_Return"].mean()
meta_std = df_out["Strategy_Log_Return"].std()
sharpe_meta = meta_mean / meta_std if meta_std != 0 else np.nan

primary_mean = df_out["Primary_Strategy_Log_Return"].mean()
primary_std = df_out["Primary_Strategy_Log_Return"].std()
sharpe_primary = primary_mean / primary_std if primary_std != 0 else np.nan

# Classification Evaluation
meta_preds = (df_out["Meta_Prob"] > signal_threshold).astype(int)
primary_preds = (df_out["Primary_Prob"] > signal_threshold).astype(int)

meta_report = classification_report(df_out["Meta_Target"].dropna(),
                                    meta_preds.loc[df_out["Meta_Target"].dropna().index],
                                    output_dict=True)
primary_report = classification_report(df_out["Meta_Target"].dropna(),
                                       primary_preds.loc[df_out["Meta_Target"].dropna().index],
                                       output_dict=True)

meta_roc = roc_auc_score(df_out["Meta_Target"].dropna(),
                         df_out["Meta_Prob"].loc[df_out["Meta_Target"].dropna().index])
primary_roc = roc_auc_score(df_out["Meta_Target"].dropna(),
                            df_out["Primary_Prob"].loc[df_out["Meta_Target"].dropna().index])

# ------------------------------
# 5. Plot Performance Metrics
# ------------------------------

# 5.1 ROC Curves
fpr_meta, tpr_meta, _ = roc_curve(df_out["Meta_Target"].dropna(), 
                                  df_out["Meta_Prob"].loc[df_out["Meta_Target"].dropna().index])
roc_auc_meta = auc(fpr_meta, tpr_meta)

fpr_primary, tpr_primary, _ = roc_curve(df_out["Meta_Target"].dropna(), 
                                        df_out["Primary_Prob"].loc[df_out["Meta_Target"].dropna().index])
roc_auc_primary = auc(fpr_primary, tpr_primary)

plt.figure(figsize=(10,6))
plt.plot(fpr_meta, tpr_meta, color='darkorange', lw=2,
         label=f'Meta ROC curve (AUC = {roc_auc_meta:.2f})')
plt.plot(fpr_primary, tpr_primary, color='blue', lw=2,
         label=f'Primary ROC curve (AUC = {roc_auc_primary:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 5.2 Confusion Matrices
meta_cm = confusion_matrix(df_out["Meta_Target"].dropna(), 
                           meta_preds.loc[df_out["Meta_Target"].dropna().index])
primary_cm = confusion_matrix(df_out["Meta_Target"].dropna(), 
                              primary_preds.loc[df_out["Meta_Target"].dropna().index])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(meta_cm, cmap='Blues')
axes[0].set_title('Meta Confusion Matrix')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
for (i, j), label in np.ndenumerate(meta_cm):
    axes[0].text(j, i, label, ha='center', va='center', 
                 color='white' if label > meta_cm.max()/2 else 'black')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

axes[1].imshow(primary_cm, cmap='Greens')
axes[1].set_title('Primary Confusion Matrix')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
for (i, j), label in np.ndenumerate(primary_cm):
    axes[1].text(j, i, label, ha='center', va='center', 
                 color='white' if label > primary_cm.max()/2 else 'black')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# 5.3 Classification Metrics Bar Chart (Positive Class)
metrics = ['precision', 'recall', 'f1-score']
meta_metrics = [meta_report['1'][m] for m in metrics]
primary_metrics = [primary_report['1'][m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,6))
plt.bar(x - width/2, meta_metrics, width, label='Meta', color='magenta')
plt.bar(x + width/2, primary_metrics, width, label='Primary', color='orange')
plt.xticks(x, metrics)
plt.ylabel('Score')
plt.title('Classification Metrics (Positive Class)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 5.4 Sharpe Ratio Comparison
plt.figure(figsize=(6,6))
plt.bar(['Meta Strategy', 'Primary Strategy'], [sharpe_meta, sharpe_primary],
        color=['magenta', 'orange'])
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio Comparison')
plt.ylim(0, max(sharpe_meta, sharpe_primary) * 1.2)
plt.grid(alpha=0.3)
plt.show()

# 5.5 Cumulative Returns Plot (for reference)
plt.style.use("dark_background")
plt.figure(figsize=(10, 6))
df_out["Strategy_Log_Return"].cumsum().plot(color="magenta", label="Meta Strategy Returns")
df_out["Primary_Strategy_Log_Return"].cumsum().plot(color="orange", label="Primary Strategy Returns")
df_out["Log_Return"].cumsum().plot(color="cyan", label="Market Returns")
plt.title("Cumulative Log Returns (Out-of-Sample)")
plt.xlabel("Time")
plt.ylabel("Cumulative Log Return")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# ------------------------------
# 6. Print Summary Metrics
# ------------------------------
print("=== Strategy Performance ===")
print(f"Cumulative Return (Meta Strategy): {cumulative_return_meta:.4f}")
print(f"Cumulative Return (Primary Strategy): {cumulative_return_primary:.4f}")
print(f"Sharpe Ratio (Meta Strategy): {sharpe_meta:.4f}")
print(f"Sharpe Ratio (Primary Strategy): {sharpe_primary:.4f}")
print("\n=== Classification Metrics ===")
print("Meta Model Classification Report:")
print(meta_report)
print("\nPrimary Model Classification Report:")
print(primary_report)
print("\nROC AUC:")
print(f"Meta Model ROC AUC: {roc_auc_meta:.4f}")
print(f"Primary Model ROC AUC: {roc_auc_primary:.4f}")