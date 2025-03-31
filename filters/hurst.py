import numpy as np
import pandas as pd
from tqdm import tqdm


def hurst_exponent(time_series, max_lag=100):
    """
    Calculate the Hurst exponent of a time series

    Parameters:
    -----------
    time_series : array-like
        Time series for analysis
    max_lag : int
        Maximum lag to consider

    Returns:
    --------
    float
        Hurst exponent
    """
    time_series = np.array(time_series)
    n = len(time_series)

    # Ensure max_lag doesn't exceed series length / 4
    max_lag = min(max_lag, n // 4)

    # Calculate range of lags (logarithmically spaced)
    lags = np.logspace(0.5, np.log10(max_lag), 20).astype(int)
    lags = np.unique(lags)  # Remove duplicates

    # Calculate various statistics at different lags
    tau = []
    rs = []

    for lag in lags:
        # Split time series into chunks of length lag
        n_chunks = n // lag

        if n_chunks < 1:
            continue

        # Trim series to fit even chunks
        series = time_series[:n_chunks * lag]
        series = series.reshape((n_chunks, lag))

        # Calculate statistics for each chunk
        chunk_rs = []

        for chunk in series:
            # Mean-adjusted series
            mean_adj = chunk - np.mean(chunk)

            # Cumulative deviations
            cumsum = np.cumsum(mean_adj)

            # Range (max - min of cumulative deviations)
            r = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            s = np.std(chunk)

            # Prevent division by zero
            if s == 0:
                continue

            # R/S ratio
            rs_ratio = r / s
            chunk_rs.append(rs_ratio)

        if chunk_rs:
            # Average R/S ratio across chunks
            tau.append(lag)
            rs.append(np.mean(chunk_rs))

    # Log-log regression to estimate Hurst exponent
    if len(tau) < 2:
        return 0.5  # Default to random walk if not enough data

    log_tau = np.log10(tau)
    log_rs = np.log10(rs)

    # Fit linear regression
    hurst = np.polyfit(log_tau, log_rs, 1)[0]

    return hurst


def hurst_filter(pairs, data, min_hurst=0.0, max_hurst=0.4):
    """
    Filter pairs based on the Hurst exponent of their spread

    Parameters:
    -----------
    pairs : list
        List of (asset1, asset2, p_value) tuples
    data : pd.DataFrame
        DataFrame containing price data for all assets
    min_hurst, max_hurst : float
        Acceptable range for Hurst exponent

    Returns:
    --------
    list
        Filtered list of pairs
    """
    filtered_pairs = []

    for asset1, asset2, p_value in tqdm(pairs, desc="Applying Hurst Filter"):
        # Calculate spread
        asset1_prices = data[asset1]
        asset2_prices = data[asset2]

        # Simple OLS regression for hedge ratio
        X = np.vstack([np.ones(len(asset1_prices)), asset1_prices]).T
        beta = np.linalg.lstsq(X, asset2_prices, rcond=None)[0][1]

        # Calculate spread
        spread = asset2_prices - beta * asset1_prices

        # Calculate Hurst exponent
        h = hurst_exponent(spread)

        # H < 0.5 indicates mean-reversion
        # Lower H implies stronger mean-reversion
        if min_hurst <= h <= max_hurst:
            filtered_pairs.append((asset1, asset2, p_value))

    return filtered_pairs
