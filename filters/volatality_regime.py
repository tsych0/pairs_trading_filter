import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def detect_volatility_regime(returns, n_regimes=2, window=63):
    """
    Detect volatility regimes using Gaussian Mixture Model

    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    n_regimes : int
        Number of volatility regimes to detect
    window : int
        Rolling window size for volatility calculation

    Returns:
    --------
    pd.Series
        Regime labels (0 to n_regimes-1)
    """
    # Calculate rolling volatility
    if isinstance(returns, pd.DataFrame):
        # If multiple columns, calculate mean volatility across assets
        vol = returns.rolling(window).std().mean(axis=1)
    else:
        # Single series
        vol = returns.rolling(window).std()

    # Remove NaNs
    vol = vol.dropna()

    if len(vol) < n_regimes * 10:
        # Not enough data, return default regime
        return pd.Series(0, index=vol.index)

    # Reshape for GMM
    X = vol.values.reshape(-1, 1)

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_regimes, random_state=42)
    gmm.fit(X)

    # Predict regimes
    regimes = gmm.predict(X)

    # Sort regimes by volatility (0 = low volatility, n-1 = high volatility)
    means = gmm.means_.flatten()
    sorting = np.argsort(means)
    regime_map = {old: new for new, old in enumerate(sorting)}
    regimes = np.array([regime_map[r] for r in regimes])

    return pd.Series(regimes, index=vol.index)


def volatility_regime_parameters(regime):
    """
    Return appropriate parameters for each volatility regime

    Parameters:
    -----------
    regime : int
        Volatility regime (0=low, 1=medium, 2=high)

    Returns:
    --------
    dict
        Parameters for the trading strategy
    """
    if regime == 0:  # Low volatility
        return {
            'window': 20,
            'threshold': 1.0,
            'position_size': 1.0
        }
    elif regime == 1:  # Medium volatility
        return {
            'window': 15,
            'threshold': 1.25,
            'position_size': 0.75
        }
    else:  # High volatility
        return {
            'window': 10,
            'threshold': 1.5,
            'position_size': 0.5
        }


def adapt_signals_to_regime(asset1, asset2, beta, volatility_regimes):
    """
    Adapt trading signals based on detected volatility regime

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Asset price series
    beta : float
        Hedge ratio
    volatility_regimes : pd.Series
        Detected volatility regimes

    Returns:
    --------
    pd.DataFrame
        DataFrame with signals and parameter values
    """
    # Calculate spread
    spread = asset2 - beta * asset1

    # Initialize results
    results = pd.DataFrame(index=spread.index)
    results['spread'] = spread
    results['signal'] = 0
    results['zscore'] = np.nan
    results['window'] = np.nan
    results['threshold'] = np.nan
    results['position_size'] = np.nan

    # Fill parameters based on regime
    for date, regime in volatility_regimes.iteritems():
        if date in results.index:
            params = volatility_regime_parameters(regime)
            results.loc[date, 'window'] = params['window']
            results.loc[date, 'threshold'] = params['threshold']
            results.loc[date, 'position_size'] = params['position_size']

    # Forward fill parameters
    results = results.ffill()

    # Generate signals for each day based on that day's parameters
    for i in range(len(results)):
        if i < int(results.iloc[i]['window']):
            continue

        lookback = int(results.iloc[i]['window'])
        threshold = results.iloc[i]['threshold']

        # Calculate z-score using lookback window
        window_data = results['spread'].iloc[i-lookback:i]
        mean = window_data.mean()
        std = window_data.std()

        if std == 0:
            continue

        z_score = (results['spread'].iloc[i] - mean) / std
        results.iloc[i, results.columns.get_loc('zscore')] = z_score

        # Generate signal
        if z_score > threshold:
            results.iloc[i, results.columns.get_loc('signal')] = -1
        elif z_score < -threshold:
            results.iloc[i, results.columns.get_loc('signal')] = 1

    return results


def volatility_regime_filter(pairs, data, max_high_vol_pct=0.3):
    """
    Filter pairs based on their behavior in high volatility regimes

    Parameters:
    -----------
    pairs : list
        List of (asset1, asset2, p_value) tuples
    data : pd.DataFrame
        DataFrame containing price data for all assets
    max_high_vol_pct : float
        Maximum percentage of time a pair can spend in high volatility regime

    Returns:
    --------
    list
        Filtered list of pairs
    """
    # Calculate returns for all assets
    returns = data.pct_change().dropna()

    # Detect volatility regimes
    volatility_regimes = detect_volatility_regime(returns)

    filtered_pairs = []

    for asset1, asset2, p_value in tqdm(pairs, desc="Applying Volatality Regime Filter"):
        # Calculate pair returns
        pair_returns = 0.5 * (returns[asset1] + returns[asset2])

        # Filter pair returns to match volatility regime dates
        pair_returns = pair_returns.loc[volatility_regimes.index]

        # Extract regimes for this pair's dates
        pair_regimes = volatility_regimes.loc[pair_returns.index]

        # Calculate percentage of time in high volatility regime
        high_vol_pct = np.mean(pair_regimes == (
            len(np.unique(volatility_regimes)) - 1))

        # Filter based on volatility regime behavior
        if high_vol_pct <= max_high_vol_pct:
            filtered_pairs.append((asset1, asset2, p_value))

    return filtered_pairs
