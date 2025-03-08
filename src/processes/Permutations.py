import numpy as np
import pandas as pd
from typing import List, Union
import matplotlib.pyplot as plt


def bar_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Creates a permuted version of OHLC price data while preserving key statistical properties.

    This function takes OHLC (Open, High, Low, Close) price data and generates a permuted version
    by shuffling the relative price movements while maintaining the overall statistical properties
    of the original data. It can handle both single and multiple market data.

    The permutation process:
    1. Converts prices to log space
    2. Calculates relative movements (open to previous close, high/low/close to open)
    3. Separately shuffles intrabar movements and gaps between bars
    4. Reconstructs a new price series from the shuffled movements

    Args:
        ohlc (Union[pd.DataFrame, List[pd.DataFrame]]): Either a single DataFrame or list of DataFrames
            containing OHLC price data. Each DataFrame must have columns: ['Open','High','Low','Close']
        start_index (int, optional): Index from which to start permuting the data. Data before this
            index remains unchanged. Defaults to 0.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Union[pd.DataFrame, List[pd.DataFrame]]: Permuted version(s) of the input data in the same
            format as the input. If input was a single DataFrame, returns a DataFrame. If input was
            a list of DataFrames, returns a list of permuted DataFrames.

    Raises:
        AssertionError: If start_index is negative or if multiple markets have mismatched indexes
    """
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[["Open", "High", "Low", "Close"]])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars["Open"] - log_bars["Close"].shift()).to_numpy()

        # Get prices relative to this bars open
        r_h = (log_bars["High"] - log_bars["Open"]).to_numpy()
        r_l = (log_bars["Low"] - log_bars["Open"]).to_numpy()
        r_c = (log_bars["Close"] - log_bars["Open"]).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last close to open (gaps) seprately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Create permutation from relative prices
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index
        log_bars = np.log(reg_bars[["Open", "High", "Low", "Close"]]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]

        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(
            perm_bars, index=time_index, columns=["Open", "High", "Low", "Close"]
        )

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]


if __name__ == "__main__":
    df = "....."
    df_perm = bar_permutation(df)

    df_r = np.log(df["Close"]).diff()
    df_perm_r = np.log(df_perm["Close"]).diff()

    print(f"Mean. REAL: {df_r.mean():14.6f} PERM: {df_perm_r.mean():14.6f}")
    print(f"Stdd. REAL: {df_r.std():14.6f} PERM: {df_perm_r.std():14.6f}")
    print(f"Skew. REAL: {df_r.skew():14.6f} PERM: {df_perm_r.skew():14.6f}")
    print(f"Kurt. REAL: {df_r.kurt():14.6f} PERM: {df_perm_r.kurt():14.6f}")

    plt.style.use("dark_background")
    np.log(df["Close"]).diff().cumsum().plot(color="blue", label="Real BTCUSD")

    plt.ylabel("Cumulative Log Return")
    plt.title("Real BTCUSD")
    plt.legend()

    np.log(df_perm["Close"]).diff().cumsum().plot(
        color="orange", label="Permutted BTCUSD"
    )
    plt.title("Permuted BTCUSD and Real BTCUSD")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.show()
