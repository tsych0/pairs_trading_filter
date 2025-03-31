import numpy as np
import pandas as pd


def kalman_filter_regression(asset1, asset2, delta=1e-5, R=0.001):
    """
    Estimate time-varying regression parameters using Kalman filter:
    asset2 = alpha + beta * asset1 + noise

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets
    delta : float
        Process noise variance multiplier
    R : float
        Observation noise variance

    Returns:
    --------
    tuple
        (alpha series, beta series)
    """
    n = len(asset1)

    # State vector [alpha, beta]
    theta = np.zeros((n, 2))

    # State covariance matrices
    P = np.zeros((n, 2, 2))

    # Use first 20 observations (or fewer) for initial OLS estimate
    window = min(20, n)
    X = np.column_stack((np.ones(window), asset1.iloc[:window].values))
    y = asset2.iloc[:window].values
    theta0 = np.linalg.lstsq(X, y, rcond=None)[0]

    # Initialize
    theta[0] = theta0
    P[0] = np.eye(2) * 1.0

    # Process noise covariance (assumed constant)
    Q = np.eye(2) * delta

    for t in range(1, n):
        # --- Prediction step ---
        theta_pred = theta[t-1]
        P_pred = P[t-1] + Q

        # --- Observation step ---
        # Observation matrix: F = [1, asset1_t]
        F = np.array([1, asset1.iloc[t]])

        # Calculate prediction error
        y_pred = np.dot(F, theta_pred)
        e = asset2.iloc[t] - y_pred

        # Innovation (residual) variance
        S = np.dot(F, np.dot(P_pred, F)) + R

        # Kalman gain
        K = np.dot(P_pred, F) / S

        # --- Update step ---
        theta[t] = theta_pred + K * e
        P[t] = P_pred - np.outer(K, F).dot(P_pred)

    # Return the filtered series
    return pd.Series(theta[:, 0], index=asset1.index), pd.Series(theta[:, 1], index=asset1.index)


def compute_spread_kalman(asset1, asset2, alpha, beta):
    """
    Compute the spread using Kalman filter estimates

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets
    alpha, beta : pd.Series
        Time-varying intercept and slope from Kalman filter

    Returns:
    --------
    pd.Series
        Spread series
    """
    return asset2 - (alpha + beta * asset1)


def compute_kalman_returns(asset1, asset2, beta, signals):
    """
    Compute portfolio returns from Kalman filter-based strategy

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets
    beta : pd.Series
        Time-varying hedge ratio from Kalman filter
    signals : pd.Series
        Trading signals

    Returns:
    --------
    pd.Series
        Daily portfolio returns
    """
    # Calculate asset returns
    r1 = asset1.pct_change().fillna(0)
    r2 = asset2.pct_change().fillna(0)

    # Shift signals to avoid look-ahead bias
    signals_shifted = signals.shift(1).fillna(0)

    # Shift beta to avoid look-ahead bias
    beta_shifted = beta.shift(1).fillna(0)

    # Calculate portfolio returns
    port_returns = signals_shifted * (r2 - beta_shifted * r1)

    return port_returns
