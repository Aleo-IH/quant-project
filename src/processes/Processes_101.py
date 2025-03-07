from pandas import DataFrame
from typing import List

def getXy(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into features (X) and target variables (y).
    
    Args:
        df (DataFrame): Input DataFrame containing both features and target columns
        
    Returns:
        tuple[DataFrame, DataFrame]: A tuple containing:
            - X (DataFrame): Feature matrix 
            - y (DataFrame): Target variables (columns containing 'Target')
    """
    y = df.filter(like="Target")
    X = df.drop(columns=list(y.columns) + ['Open', 'High', 'Low', 'Close'])
    return X, y


def add_returns_targets(df : DataFrame, lags : List[int]):
    for lag in lags:
        df.loc[:, f"Return_{lag}"] = df.loc[:, "Close"].pct_change(lag)
        df.loc[:, f"Target_{lag}"] = df.loc[:, f"Return_{lag}"].shift(-lag)
    
    return df