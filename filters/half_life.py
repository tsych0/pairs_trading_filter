import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


def compute_half_life(spread):
    """
    Compute the half-life of mean reversion using AR(1) model

    Parameters:
    -----------
    spread : array-like
        Spread between the two assets

    Returns:
    --------
    float
        Half-life of mean reversion
    """
    spread = np.array(spread)

    # Spread change
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    # Add constant to regression
    spread_lag = sm.add_constant(spread_lag)

    # Fit linear regression model (OLS)
    model = sm.OLS(spread_diff, spread_lag).fit()

    # Get beta (coefficient on spread_lag)
    beta = model.params[1]

    # Calculate half-life
    if beta >= 0:
        # No mean reversion, return large value
        return np.inf
    else:
        return -np.log(2) / beta


def half_life_filter(pairs, data, min_half_life=1, max_half_life=20):
    """
    Filter pairs based on the half-life of mean reversion

    Parameters:
    -----------
    pairs : list
        List of (asset1, asset2, p_value) tuples
    data : pd.DataFrame
        DataFrame containing price data for all assets
    min_half_life, max_half_life : int
        Acceptable range for half-life (in days)

    Returns:
    --------
    list
        Filtered list of pairs
    """
    filtered_pairs = []

    for asset1, asset2, p_value in tqdm(pairs, desc="Applying Half Life Filter"):
        # Calculate spread
        asset1_prices = data[asset1]
        asset2_prices = data[asset2]

        # Simple OLS regression for hedge ratio
        X = sm.add_constant(asset1_prices)
        model = sm.OLS(asset2_prices, X).fit()
        hedge_ratio = model.params.iloc[1]

        # Calculate spread
        spread = asset2_prices - hedge_ratio * asset1_prices

        # Calculate half-life
        half_life = compute_half_life(spread)

        # Filter based on half-life
        if min_half_life <= half_life <= max_half_life:
            filtered_pairs.append((asset1, asset2, p_value))

    return filtered_pairs
