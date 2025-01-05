from pandas import DataFrame

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
    X = df.drop(columns=y.columns)
    return X, y
