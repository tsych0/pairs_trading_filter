import numpy as np
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint
from joblib import Parallel, delayed
from tqdm import tqdm


def find_cointegrated_pairs(data, significance=0.05, n_jobs=-1):
    """
    Find cointegrated asset pairs using the Engle-Granger test

    Parameters:
    -----------
    data : pd.DataFrame
        Price data for all assets
    significance : float
        P-value threshold for cointegration
    n_jobs : int
        Number of parallel jobs

    Returns:
    --------
    tuple
        (p-value matrix, list of cointegrated pairs)
    """
    assets = data.columns
    n = len(assets)

    # All unique combinations of assets
    pairs = list(itertools.combinations(assets, 2))

    def coint_test(pair):
        """Run cointegration test on a single pair"""
        a1, a2 = pair
        s1, s2 = data[a1], data[a2]

        # Ensure series don't have NaNs
        if s1.isnull().any() or s2.isnull().any():
            return a1, a2, 1.0

        try:
            score, p_value, _ = coint(s1, s2)
            return a1, a2, p_value
        except:
            return a1, a2, 1.0

    # Run tests in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(coint_test)(pair) for pair in tqdm(pairs, desc="Testing cointegration")
    )

    # Build p-value matrix and list of cointegrated pairs
    pvalue_matrix = pd.DataFrame(np.ones((n, n)), index=assets, columns=assets)
    cointegrated_pairs = []

    for asset1, asset2, p_value in results:
        pvalue_matrix.loc[asset1, asset2] = p_value
        pvalue_matrix.loc[asset2, asset1] = p_value  # For symmetry

        if p_value < significance:
            cointegrated_pairs.append((asset1, asset2, p_value))

    return pvalue_matrix, cointegrated_pairs


def compute_hedge_ratio(asset1, asset2):
    """
    Compute hedge ratio using OLS regression

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets

    Returns:
    --------
    float
        Hedge ratio (beta)
    """
    import statsmodels.api as sm

    # Add constant to independent variable
    X = sm.add_constant(asset1)

    # Fit OLS model
    model = sm.OLS(asset2, X).fit()

    # Extract slope coefficient (beta)
    beta = model.params.iloc[1]

    return beta


def compute_spread(asset1, asset2, beta):
    """
    Compute the spread between two assets

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets
    beta : float
        Hedge ratio

    Returns:
    --------
    pd.Series
        Spread series
    """
    return asset2 - beta * asset1
