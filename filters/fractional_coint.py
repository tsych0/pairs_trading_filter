import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize
from tqdm import tqdm
from joblib import Parallel, delayed


def fractional_diff(series, d, k=None):
    """Highly optimized fractional differencing"""
    series = np.array(series)
    n = len(series)

    if k is None:
        k = n

    k = min(k, n)

    # Pre-compute all weights at once
    weights = np.zeros(k)
    weights[0] = 1
    for i in range(1, k):
        weights[i] = weights[i-1] * (d - i + 1) / i

    # Create a weight matrix for faster computation
    weight_matrix = np.zeros((n, k))
    for i in range(n):
        j_max = min(i+1, k)
        weight_matrix[i, :j_max] = weights[:j_max][::-1]

    # Use matrix operations for the entire series at once
    fractionalized = np.zeros(n)
    for i in range(n):
        j_max = min(i+1, k)
        if j_max > 0:
            fractionalized[i] = np.sum(
                weight_matrix[i, :j_max] * series[max(0, i-j_max+1):i+1])

    return fractionalized


def process_single_pair(asset1, asset2, p_value, data, d_min, d_max, adf_threshold):
    """Process a single pair for fractional cointegration"""
    try:
        # Compute spread
        spread = np.log(data[asset1].values) - np.log(data[asset2].values)

        # Grid search for optimal d
        d_values = np.arange(0, 1, 0.01)
        best_d = None
        best_adf = float('inf')
        best_p_value = 1.0

        for d in d_values:
            diff_spread = fractional_diff(spread, d)
            diff_spread = diff_spread[~np.isnan(diff_spread)]

            if len(diff_spread) < 20:
                continue

            adf_result = adfuller(diff_spread)

            if adf_result[1] < best_p_value:
                best_p_value = adf_result[1]
                best_adf = adf_result[0]
                best_d = d

        # Check if pair meets criteria
        if (best_d is not None and
            d_min <= best_d <= d_max and
                best_adf < adf_threshold):
            return (asset1, asset2, p_value)
        else:
            return None
    except Exception as e:
        # Skip pairs that cause errors
        return None


def fractional_coint_filter(pairs, data, d_min=0.2, d_max=0.8, adf_threshold=-3.5, n_jobs=-1):
    """Parallel version with progress tracking"""
    # Using generator version for real-time updates
    def _parallel_wrapper():
        parallel = Parallel(n_jobs=n_jobs, return_as='generator')
        for result in parallel(
            delayed(process_single_pair)(
                asset1, asset2, p_value, data, d_min, d_max, adf_threshold
            )
            for asset1, asset2, p_value in pairs
        ):
            yield result

    filtered_pairs = []
    for result in tqdm(_parallel_wrapper(), total=len(pairs), desc="Applying Fractional Coint Filter"):
        if result:
            filtered_pairs.append(result)

    return filtered_pairs
