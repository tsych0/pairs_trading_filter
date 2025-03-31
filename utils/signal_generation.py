import pandas as pd
import numpy as np


def generate_signals(spread, window=20, threshold=1.0):
    """
    Generate trading signals based on z-score of spread

    Parameters:
    -----------
    spread : pd.Series
        Spread between two assets
    window : int
        Rolling window size for z-score calculation
    threshold : float
        Z-score threshold for signal generation

    Returns:
    --------
    tuple
        (signals, z-scores)
    """
    # Calculate rolling statistics
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()

    # Calculate z-score
    zscore = (spread - rolling_mean) / rolling_std

    # Initialize signals
    signals = pd.Series(0, index=spread.index)

    # Generate signals based on z-score
    signals[zscore > threshold] = -1  # Short signal (sell the spread)
    signals[zscore < -threshold] = 1   # Long signal (buy the spread)

    return signals, zscore


def compute_pair_returns(asset1, asset2, beta, signals):
    """
    Compute returns for a pairs trading strategy

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets
    beta : float
        Hedge ratio
    signals : pd.Series
        Trading signals

    Returns:
    --------
    pd.Series
        Strategy returns
    """
    # Calculate returns
    r1 = asset1.pct_change().fillna(0)
    r2 = asset2.pct_change().fillna(0)

    # Shift signals to avoid look-ahead bias
    signals_shifted = signals.shift(1).fillna(0)

    # Calculate strategy returns
    port_returns = signals_shifted * (r2 - beta * r1)

    return port_returns
