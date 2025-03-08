import re
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier

from src.processes.Processes_101 import getXy
from src.visualization.multioutput_multiclass import (
    plot_cross_validation_metrics,
    plot_bin_precision,
)


df = "......."
# Initialize TimeSeriesSplit
nums_split = 31
num_bins = 5
tscv = TimeSeriesSplit(n_splits=nums_split)

# Lists to store metrics for each fold
accuracies = []
hamming_losses = []
jaccard_scores = []

# Initialize KBinsDiscretizer for y
y_discretizer = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="quantile")


pipeline = Pipeline(
    [
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "scaler",
                        StandardScaler(),
                        ["ATRr_14", "Number of Trades", "volume"],
                    ),
                    (
                        "discretizer",
                        KBinsDiscretizer(
                            n_bins=num_bins, encode="ordinal", strategy="quantile"
                        ),
                        [col for col in df if re.search(r"Return", col)],
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        (
            "model",
            ClassifierChain(
                RandomForestClassifier(
                    n_estimators=100, n_jobs=-1, criterion="log_loss"
                )
            ),
        ),
    ]
)

cv_result = []
# Perform cross-validation with progress bar
for train_idx, test_idx in tqdm(
    tscv.split(df), desc="Cross-validation", total=nums_split
):
    # Split data
    Dtrain = df.iloc[train_idx]
    Dtest = df.iloc[test_idx]

    X_train, y_train = getXy(Dtrain)
    X_test, y_test = getXy(Dtest)

    # Transform and fit
    y_train_tf = y_discretizer.fit_transform(y_train)
    pipeline.fit(X_train, y_train_tf)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    y_test_tf = y_discretizer.transform(y_test)
    # Convert y_pred to DataFrame with same structure as y_test_tf
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)
    y_test_df = pd.DataFrame(y_test_tf, index=y_test.index, columns=y_test.columns)
    cv_result.append((y_test_df, y_pred_df))


plot_cross_validation_metrics(cv_result)
plot_bin_precision(cv_result, n_bins=num_bins)
