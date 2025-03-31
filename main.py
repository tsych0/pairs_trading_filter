import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import os

# Import project modules
from models.pairs_generation import find_cointegrated_pairs, compute_hedge_ratio, compute_spread
from models.kalman import kalman_filter_regression, compute_spread_kalman, compute_kalman_returns
from utils.data_utils import load_data, split_data
from utils.signal_generation import generate_signals, compute_pair_returns
from utils.performance import portfolio_statistics, plot_performance_comparison, print_performance_table

# Import filters
from filters.ssd import detailed_ssd_efficiency_test
from filters.fractional_coint import fractional_coint_filter
from filters.half_life import half_life_filter
from filters.hurst import hurst_filter
from filters.volatality_regime import volatility_regime_filter, detect_volatility_regime, adapt_signals_to_regime

# Import configuration
import config


def setup():
    """Set up directories if they don't exist"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)


def apply_ssd_filter(pairs, data, benchmark, tol=0.5):
    """Apply the Second-Order Stochastic Dominance filter"""
    filtered_pairs = []

    for asset1, asset2, p_value in tqdm(pairs, desc="Applying SSD Filter"):
        # Calculate hedge ratio
        beta = compute_hedge_ratio(data[asset1], data[asset2])

        # Calculate spread
        spread = compute_spread(data[asset1], data[asset2], beta)

        # Generate signals
        signals, _ = generate_signals(
            spread, window=config.ZSCORE_WINDOW, threshold=config.ZSCORE_THRESHOLD)

        # Calculate portfolio returns
        port_returns = compute_pair_returns(
            data[asset1], data[asset2], beta, signals)

        # Prepare data for SSD test
        candidate_returns = port_returns.dropna().values
        benchmark_returns = benchmark.pct_change().dropna().values

        # Align lengths
        min_len = min(len(candidate_returns), len(benchmark_returns))
        candidate_returns = candidate_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # SSD test
        try:
            status, _, total_adj, _, _, is_ssd_eff = detailed_ssd_efficiency_test(
                candidate_returns, benchmark_returns, num_bins=100, tol=tol
            )

            if is_ssd_eff:
                filtered_pairs.append((asset1, asset2, p_value))
        except:
            # Skip if test fails
            continue

    return filtered_pairs


def build_pair_returns(filtered_pairs, data, method='static', window=20, threshold=1.5):
    """
    Build returns dataframe for a list of pairs

    Parameters:
    -----------
    filtered_pairs : list
        List of (asset1, asset2, p_value) tuples
    data : pd.DataFrame
        Price data
    method : str
        'static' for traditional hedge ratio or 'kalman' for Kalman filter
    window, threshold : float
        Signal generation parameters

    Returns:
    --------
    pd.DataFrame
        DataFrame of strategy returns for each pair
    """
    pair_returns = {}

    for asset1, asset2, _ in tqdm(filtered_pairs, desc=f"Computing {method.title()} Returns"):
        a1 = data[asset1]
        a2 = data[asset2]

        if method == 'static':
            # Calculate static hedge ratio
            beta = compute_hedge_ratio(a1, a2)

            # Calculate spread
            spread = compute_spread(a1, a2, beta)

            # Generate signals
            signals, _ = generate_signals(
                spread, window=window, threshold=threshold)

            # Calculate returns
            rets = compute_pair_returns(a1, a2, beta, signals)

        elif method == 'kalman':
            # Estimate time-varying parameters
            alpha, beta = kalman_filter_regression(
                a1, a2, delta=config.KALMAN_DELTA, R=config.KALMAN_R)

            # Calculate dynamic spread
            spread = compute_spread_kalman(a1, a2, alpha, beta)

            # Generate signals
            signals, _ = generate_signals(
                spread, window=window, threshold=threshold)

            # Calculate returns
            rets = compute_kalman_returns(a1, a2, beta, signals)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Store returns
        pair_returns[f"{asset1}-{asset2}"] = rets

    # Combine into DataFrame
    returns_df = pd.DataFrame(pair_returns)

    # Drop rows with all NaN
    returns_df.dropna(how='all', inplace=True)

    return returns_df


def optimize_weights(returns_df, method='max_sharpe', no_short=True):
    """
    Calculate optimal portfolio weights

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns for each pair
    method : str
        'max_sharpe' or 'min_variance'
    no_short : bool
        Whether to allow short positions

    Returns:
    --------
    pd.Series
        Optimal weights
    """
    from scipy.optimize import minimize

    # Drop rows with any NaN
    returns = returns_df.dropna()

    # Define objective function based on method
    if method == 'max_sharpe':
        def objective(weights):
            # Portfolio return and volatility
            port_return = np.sum(returns.mean() * weights) * 252
            port_vol = np.sqrt(weights.T @ returns.cov() @ weights * 252)

            # Sharpe ratio (negative for minimization)
            return -port_return / port_vol if port_vol > 0 else 0

    elif method == 'min_variance':
        def objective(weights):
            # Portfolio volatility
            return np.sqrt(weights.T @ returns.cov() @ weights * 252)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    # Number of assets
    n_assets = returns.shape[1]

    # Initial weights (equal)
    init_weights = np.ones(n_assets) / n_assets

    # Constraints
    # weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds
    bounds = [(0, 1) if no_short else (-1, 1) for _ in range(n_assets)]

    # Run optimization
    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Check convergence
    if not result.success:
        print(f"Optimization failed: {result.message}")
        # Fallback to equal weights
        weights = init_weights
    else:
        weights = result.x

    # Create Series with pair names
    weight_series = pd.Series(weights, index=returns.columns)

    return weight_series


def run_backtest(train_data, test_data, benchmark_name='^IXIC'):
    """
    Run the full backtest with multiple filtering approaches

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data for parameter estimation
    test_data : pd.DataFrame
        Test data for backtesting
    benchmark_name : str
        Column name for benchmark index

    Returns:
    --------
    dict
        Dictionary of results
    """
    # Extract benchmark
    train_benchmark = train_data[benchmark_name]
    test_benchmark = test_data[benchmark_name]

    # Remove benchmark from asset universe
    train_data_assets = train_data.drop(columns=['^DJI', '^GSPC', '^IXIC'])
    test_data_assets = test_data.drop(columns=['^DJI', '^GSPC', '^IXIC'])

    # Find cointegrated pairs
    print("Finding cointegrated pairs...")
    _, unfiltered_pairs = find_cointegrated_pairs(
        train_data_assets,
        significance=config.COINTEGRATION_PVALUE,
        n_jobs=config.PARALLEL_JOBS if config.USE_PARALLEL else 1
    )
    print(f"Found {len(unfiltered_pairs)} cointegrated pairs")

    # Apply different filters
    print("\nApplying filters...")
    filtered_pairs_ssd = apply_ssd_filter(
        unfiltered_pairs,
        train_data_assets,
        train_benchmark,
        tol=config.SSD_TOLERANCE
    )
    print(f"SSD Filter: {len(filtered_pairs_ssd)} pairs")

    filtered_pairs_frac = fractional_coint_filter(
        unfiltered_pairs,
        train_data_assets,
        d_min=config.FRAC_D_MIN,
        d_max=config.FRAC_D_MAX,
        adf_threshold=config.FRAC_ADF_THRESHOLD
    )
    print(f"Fractional Cointegration Filter: {len(filtered_pairs_frac)} pairs")

    filtered_pairs_half_life = half_life_filter(
        unfiltered_pairs,
        train_data_assets,
        min_half_life=config.HALF_LIFE_MIN,
        max_half_life=config.HALF_LIFE_MAX
    )
    print(f"Half-Life Filter: {len(filtered_pairs_half_life)} pairs")

    filtered_pairs_hurst = hurst_filter(
        unfiltered_pairs,
        train_data_assets,
        min_hurst=config.HURST_MIN,
        max_hurst=config.HURST_MAX
    )
    print(f"Hurst Exponent Filter: {len(filtered_pairs_hurst)} pairs")

    filtered_pairs_vol = volatility_regime_filter(
        unfiltered_pairs,
        train_data_assets,
        max_high_vol_pct=config.MAX_HIGH_VOL_PCT
    )
    print(f"Volatility Regime Filter: {len(filtered_pairs_vol)} pairs")

    # Build returns dataframes for training data
    print("\nBuilding returns dataframes for training data...")
    unfiltered_returns_train = build_pair_returns(
        unfiltered_pairs,
        train_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    ssd_returns_train = build_pair_returns(
        filtered_pairs_ssd,
        train_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    frac_returns_train = build_pair_returns(
        filtered_pairs_frac,
        train_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    half_life_returns_train = build_pair_returns(
        filtered_pairs_half_life,
        train_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    hurst_returns_train = build_pair_returns(
        filtered_pairs_hurst,
        train_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    vol_returns_train = build_pair_returns(
        filtered_pairs_vol,
        train_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    # Calculate optimal weights
    print("\nCalculating optimal weights...")
    unfiltered_weights = optimize_weights(
        unfiltered_returns_train,
        method=config.OPTIMIZATION_METHOD,
        no_short=config.NO_SHORT
    )

    ssd_weights = optimize_weights(
        ssd_returns_train,
        method=config.OPTIMIZATION_METHOD,
        no_short=config.NO_SHORT
    ) if not ssd_returns_train.empty else pd.Series()

    frac_weights = optimize_weights(
        frac_returns_train,
        method=config.OPTIMIZATION_METHOD,
        no_short=config.NO_SHORT
    ) if not frac_returns_train.empty else pd.Series()

    half_life_weights = optimize_weights(
        half_life_returns_train,
        method=config.OPTIMIZATION_METHOD,
        no_short=config.NO_SHORT
    ) if not half_life_returns_train.empty else pd.Series()

    hurst_weights = optimize_weights(
        hurst_returns_train,
        method=config.OPTIMIZATION_METHOD,
        no_short=config.NO_SHORT
    ) if not hurst_returns_train.empty else pd.Series()

    vol_weights = optimize_weights(
        vol_returns_train,
        method=config.OPTIMIZATION_METHOD,
        no_short=config.NO_SHORT
    ) if not vol_returns_train.empty else pd.Series()

    # Test out-of-sample performance
    print("\nTesting out-of-sample performance...")
    unfiltered_returns_test = build_pair_returns(
        unfiltered_pairs,
        test_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    ssd_returns_test = build_pair_returns(
        filtered_pairs_ssd,
        test_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    frac_returns_test = build_pair_returns(
        filtered_pairs_frac,
        test_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    half_life_returns_test = build_pair_returns(
        filtered_pairs_half_life,
        test_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    hurst_returns_test = build_pair_returns(
        filtered_pairs_hurst,
        test_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    vol_returns_test = build_pair_returns(
        filtered_pairs_vol,
        test_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    # Calculate portfolio returns
    print("\nCalculating portfolio returns...")
    unfiltered_portfolio = unfiltered_returns_test.loc[:, unfiltered_weights.index].dot(
        unfiltered_weights)

    ssd_portfolio = pd.Series(0, index=unfiltered_portfolio.index)
    if not ssd_weights.empty and not ssd_returns_test.empty:
        common_pairs = [
            p for p in ssd_weights.index if p in ssd_returns_test.columns]
        if common_pairs:
            weights_subset = ssd_weights.loc[common_pairs]
            weights_subset = weights_subset / weights_subset.sum()
            ssd_portfolio = ssd_returns_test.loc[:, common_pairs].dot(
                weights_subset)

    frac_portfolio = pd.Series(0, index=unfiltered_portfolio.index)
    if not frac_weights.empty and not frac_returns_test.empty:
        common_pairs = [
            p for p in frac_weights.index if p in frac_returns_test.columns]
        if common_pairs:
            weights_subset = frac_weights.loc[common_pairs]
            weights_subset = weights_subset / weights_subset.sum()
            frac_portfolio = frac_returns_test.loc[:, common_pairs].dot(
                weights_subset)

    half_life_portfolio = pd.Series(0, index=unfiltered_portfolio.index)
    if not half_life_weights.empty and not half_life_returns_test.empty:
        common_pairs = [
            p for p in half_life_weights.index if p in half_life_returns_test.columns]
        if common_pairs:
            weights_subset = half_life_weights.loc[common_pairs]
            weights_subset = weights_subset / weights_subset.sum()
            half_life_portfolio = half_life_returns_test.loc[:, common_pairs].dot(
                weights_subset)

    hurst_portfolio = pd.Series(0, index=unfiltered_portfolio.index)
    if not hurst_weights.empty and not hurst_returns_test.empty:
        common_pairs = [
            p for p in hurst_weights.index if p in hurst_returns_test.columns]
        if common_pairs:
            weights_subset = hurst_weights.loc[common_pairs]
            weights_subset = weights_subset / weights_subset.sum()
            hurst_portfolio = hurst_returns_test.loc[:, common_pairs].dot(
                weights_subset)

    vol_portfolio = pd.Series(0, index=unfiltered_portfolio.index)
    if not vol_weights.empty and not vol_returns_test.empty:
        common_pairs = [
            p for p in vol_weights.index if p in vol_returns_test.columns]
        if common_pairs:
            weights_subset = vol_weights.loc[common_pairs]
            weights_subset = weights_subset / weights_subset.sum()
            vol_portfolio = vol_returns_test.loc[:, common_pairs].dot(
                weights_subset)

    # Benchmark returns
    benchmark_returns = test_benchmark.pct_change().dropna()

    # Compile results
    strategy_returns = {
        'Unfiltered': unfiltered_portfolio,
        'SSD Filter': ssd_portfolio,
        'Fractional Cointegration': frac_portfolio,
        'Half-Life': half_life_portfolio,
        'Hurst Exponent': hurst_portfolio,
        'Volatility Regime': vol_portfolio
    }

    # Compare performance
    print("\nComparing performance metrics:")
    metrics_table = print_performance_table(
        strategy_returns, benchmark_returns)
    print(metrics_table)

    # Plot cumulative returns
    plt.figure(figsize=(12, 8))
    for name, returns in strategy_returns.items():
        if len(returns) > 0:
            cumulative = (1 + returns).cumprod()
            plt.plot(cumulative.index, cumulative, label=name)

    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    plt.plot(cumulative_benchmark.index, cumulative_benchmark,
             label='Benchmark', linestyle='--')

    plt.title('Strategy Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/performance_comparison.png')

    # Save metrics to CSV
    metrics_table.to_csv('results/performance_metrics.csv')

    return {
        'unfiltered_pairs': unfiltered_pairs,
        'filtered_pairs': {
            'ssd': filtered_pairs_ssd,
            'fractional': filtered_pairs_frac,
            'half_life': filtered_pairs_half_life,
            'hurst': filtered_pairs_hurst,
            'volatility': filtered_pairs_vol
        },
        'weights': {
            'unfiltered': unfiltered_weights,
            'ssd': ssd_weights,
            'fractional': frac_weights,
            'half_life': half_life_weights,
            'hurst': hurst_weights,
            'volatility': vol_weights
        },
        'portfolio_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'metrics': metrics_table
    }


def main():
    parser = argparse.ArgumentParser(description='Pairs Trading Backtesting')
    parser.add_argument('--data', type=str,
                        default=config.DATA_PATH, help='Path to data file')
    parser.add_argument('--benchmark', type=str,
                        default='^IXIC', help='Benchmark column name')
    parser.add_argument(
        '--train', type=int, default=config.TRAIN_YEARS, help='Training period in years')
    parser.add_argument(
        '--total', type=int, default=config.TOTAL_YEARS, help='Total period in years')

    args = parser.parse_args()

    # Set up directories
    setup()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)

    # Split into training and testing
    print(
        f"Splitting data into training ({args.train} years) and testing ({args.total - args.train} years)...")
    train_data, test_data = split_data(
        data, train_years=args.train, total_years=args.total)

    # Run backtest
    print("Running backtest...")
    results = run_backtest(train_data, test_data,
                           benchmark_name=args.benchmark)

    print("\nBacktest complete!")
    print(f"Results saved to 'results/' directory")
    print("\nTo generate a complete LaTeX report, run:")
    print("python generate_report.py")
    print("This will create LaTeX tables in report/tables/ that can be included in your research paper")

    return results


if __name__ == "__main__":
    main()
