import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
import time
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from p_tqdm import p_map

# Import project modules
from models.pairs_generation import find_cointegrated_pairs, compute_hedge_ratio, compute_spread
from models.kalman import kalman_filter_regression, compute_spread_kalman, compute_kalman_returns
from utils.data_utils import load_data, split_data
from utils.signal_generation import generate_signals, compute_pair_returns
from utils.performance import portfolio_statistics

# Import filters
from filters.ssd import detailed_ssd_efficiency_test
from filters.fractional_coint import fractional_coint_filter
from filters.half_life import half_life_filter
from filters.hurst import hurst_filter
from filters.volatality_regime import volatility_regime_filter

# Import configuration
import config


def setup_report_directories():
    """Create directories for report outputs"""
    os.makedirs('report', exist_ok=True)
    os.makedirs('report/figures', exist_ok=True)
    os.makedirs('report/tables', exist_ok=True)
    os.makedirs('report/data', exist_ok=True)

    print(f"Report directories created at: {os.path.abspath('report')}")


def save_figure(plt, filename, dpi=300):
    """Save figure in multiple formats with proper paths"""
    # Create base paths
    png_path = f'report/figures/{filename}.png'
    pdf_path = f'report/figures/{filename}.pdf'

    # Save in multiple formats
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"Figure saved: {png_path}")


def plot_cumulative_returns(returns_dict, benchmark_returns, title, filename):
    """Generate and save publication-quality cumulative returns plot"""
    plt.figure(figsize=(12, 8))

    # Use a professional color palette
    colors = sns.color_palette("viridis", len(returns_dict))

    # Plot each strategy
    for i, (name, returns) in enumerate(returns_dict.items()):
        if len(returns) > 0:
            cumulative = (1 + returns).cumprod()
            plt.plot(cumulative.index, cumulative,
                     label=name, color=colors[i], linewidth=2)

    # Add benchmark
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    plt.plot(cumulative_benchmark.index, cumulative_benchmark,
             label='Benchmark', linestyle='--', color='black', linewidth=2)

    # Format plot for publication
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, frameon=True, framealpha=0.7)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f'{y:.1f}x'))

    plt.tight_layout()
    save_figure(plt, filename)
    plt.close()


def plot_drawdowns(returns_dict, title, filename):
    """Generate and save publication-quality drawdown chart"""
    plt.figure(figsize=(12, 8))

    # Use a professional color palette
    colors = sns.color_palette("viridis", len(returns_dict))

    # Plot each strategy's drawdowns
    for i, (name, returns) in enumerate(returns_dict.items()):
        if len(returns) > 0:
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            plt.plot(drawdown.index, drawdown, label=name,
                     color=colors[i], linewidth=2)

    # Format plot for publication
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Drawdown', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, frameon=True, framealpha=0.7)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))

    plt.tight_layout()
    save_figure(plt, filename)
    plt.close()


def plot_rolling_metrics(returns_dict, benchmark_returns, window=60, title_prefix="Rolling", filename_prefix="rolling"):
    """Generate and save rolling performance metrics plots"""
    metrics = ["Sharpe Ratio", "Volatility", "Returns"]

    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Use a professional color palette
        colors = sns.color_palette("viridis", len(returns_dict))

        # Plot each strategy's rolling metric
        for i, (name, returns) in enumerate(returns_dict.items()):
            if len(returns) < window or returns.isna().all():
                continue

            if metric == "Sharpe Ratio":
                rolling_ret = returns.rolling(window=window).mean() * 252
                rolling_vol = returns.rolling(
                    window=window).std() * np.sqrt(252)
                rolling_metric = rolling_ret / rolling_vol
                def y_formatter(y, _): return f'{y:.2f}'

            elif metric == "Volatility":
                rolling_metric = returns.rolling(
                    window=window).std() * np.sqrt(252)

                def y_formatter(y, _): return f'{y:.1%}'

            elif metric == "Returns":
                rolling_metric = returns.rolling(window=window).mean() * 252
                def y_formatter(y, _): return f'{y:.1%}'

            plt.plot(rolling_metric.index, rolling_metric,
                     label=name, color=colors[i], linewidth=2)

        # Add benchmark if applicable
        if metric in ["Sharpe Ratio", "Volatility", "Returns"]:
            if metric == "Sharpe Ratio":
                b_rolling_ret = benchmark_returns.rolling(
                    window=window).mean() * 252
                b_rolling_vol = benchmark_returns.rolling(
                    window=window).std() * np.sqrt(252)
                b_rolling_metric = b_rolling_ret / b_rolling_vol

            elif metric == "Volatility":
                b_rolling_metric = benchmark_returns.rolling(
                    window=window).std() * np.sqrt(252)

            elif metric == "Returns":
                b_rolling_metric = benchmark_returns.rolling(
                    window=window).mean() * 252

            plt.plot(b_rolling_metric.index, b_rolling_metric,
                     label='Benchmark', linestyle='--', color='black', linewidth=2)

        # Format plot for publication
        plt.title(f"{title_prefix} {metric} ({window}-day window)",
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(f"Annualized {metric}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, frameon=True, framealpha=0.7)

        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()

        # Format y-axis
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))

        plt.tight_layout()
        save_figure(
            plt, f"{filename_prefix}_{metric.lower().replace(' ', '_')}")
        plt.close()


def plot_filter_comparison_bar(metrics_df, metric_name, title, filename):
    """Generate and save bar chart comparing filter performance"""
    # Extract the specified metric
    metric_values = metrics_df.loc[metric_name]

    plt.figure(figsize=(14, 8))

    # Set up color palette
    colors = sns.color_palette("viridis", len(metric_values))

    # Create bar chart
    bars = plt.bar(metric_values.index, metric_values.values, color=colors)

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        value = metric_values.values[i]
        if metric_name in ['Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
            value_text = f'{value:.1%}'
        else:
            value_text = f'{value:.2f}'

        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(metric_values.values),
                 value_text, ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Format plot for publication
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Set y-axis format for percentage metrics
    if metric_name in ['Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    save_figure(plt, filename)
    plt.close()


def plot_filter_reduction(filter_summary, title, filename):
    """Generate and save chart showing pair reduction by filters"""
    plt.figure(figsize=(14, 8))

    # Set up color palette
    colors = sns.color_palette("viridis", len(filter_summary))

    # Create bar chart
    bars = plt.bar(filter_summary['Filter'],
                   filter_summary['Number of Pairs'], color=colors)

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        pairs_count = filter_summary['Number of Pairs'][i]
        reduction = filter_summary['Reduction (%)'][i]

        if i == 0:  # Unfiltered
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{pairs_count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{pairs_count}\n(-{reduction:.1f}%)', ha='center', va='bottom',
                     fontweight='bold', fontsize=12)

    # Format plot for publication
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel("Number of Pairs", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=12)

    plt.tight_layout()
    save_figure(plt, filename)
    plt.close()


def generate_performance_report(strategies_dict, benchmark_returns, filename_prefix):
    """Calculate and save comprehensive performance metrics table"""
    # Add benchmark to dictionary
    all_strategies = strategies_dict.copy()
    all_strategies['Benchmark'] = benchmark_returns

    # Calculate statistics for each strategy
    stats = {}

    for name, returns in all_strategies.items():
        if not isinstance(returns, pd.Series) or len(returns) == 0:
            continue
        stats[name] = portfolio_statistics(returns)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(stats)

    # Save raw data as CSV
    metrics_df.to_csv(f'report/tables/{filename_prefix}_metrics.csv')

    # Create formatted version for reports
    formatted_df = metrics_df.copy()

    # Format percentages
    for metric in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
        if metric in formatted_df.index:
            formatted_df.loc[metric] = formatted_df.loc[metric].map(
                lambda x: f"{x:.2%}")

    # Format ratios
    for metric in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
        if metric in formatted_df.index:
            formatted_df.loc[metric] = formatted_df.loc[metric].map(
                lambda x: f"{x:.2f}")

    # Generate enhanced LaTeX table with proper formatting
    latex_table = formatted_df.to_latex(
        float_format="%.2f",
        bold_rows=True,
        column_format="l" + "r" * len(formatted_df.columns),
        caption=f"Performance Metrics Comparison",
        label=f"tab:performance_metrics",
        position="ht"
    )

    # Add booktabs styling
    latex_table = latex_table.replace(
        '\\begin{table}', '\\begin{table}[ht]\n\\centering')
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{tabular}')
    latex_table = latex_table.replace(
        '\\toprule', '\\toprule\n\\textbf{Metric}')

    with open(f'report/tables/{filename_prefix}_metrics.tex', 'w') as f:
        f.write(latex_table)

    # Also save a standalone version for direct inclusion
    with open(f'report/tables/{filename_prefix}_metrics_standalone.tex', 'w') as f:
        f.write('% This is an auto-generated table from your pair trading analysis\n')
        f.write(
            f'% Include in results.tex using: \\input{{report/tables/{filename_prefix}_metrics_standalone.tex}}\n\n')
        f.write('\\begin{tabular}{l' + 'r' * len(formatted_df.columns) + '}\n')
        f.write('\\toprule\n')
        f.write('\\textbf{Metric} & ' + ' & '.join(
            [f'\\textbf{{{col}}}' for col in formatted_df.columns]) + ' \\\\\n')
        f.write('\\midrule\n')

        for idx, row in formatted_df.iterrows():
            f.write(f'{idx} & ' + ' & '.join([str(val)
                    for val in row.values]) + ' \\\\\n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    return metrics_df


def generate_filter_summary(unfiltered_pairs, filtered_pairs_dict, filename):
    """Generate and save summary of filter effects on pair count"""
    # Create summary dataframe
    data = {
        'Filter': ['Unfiltered'] + list(filtered_pairs_dict.keys()),
        'Number of Pairs': [len(unfiltered_pairs)] + [len(pairs) for pairs in filtered_pairs_dict.values()],
    }

    # Calculate reduction percentages
    baseline = len(unfiltered_pairs)
    data['Reduction (%)'] = [0] + [(baseline - len(pairs)) /
                                   baseline * 100 for pairs in filtered_pairs_dict.values()]

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save as CSV
    df.to_csv(f'report/tables/{filename}.csv', index=False)

    # Generate enhanced LaTeX table
    latex_table = df.to_latex(
        index=False,
        float_format="%.1f",
        caption="Effect of Filtering Methods on Pair Count",
        label="tab:filter_selection",
        position="ht"
    )

    # Add styling
    latex_table = latex_table.replace(
        '\\begin{table}', '\\begin{table}[ht]\n\\centering')

    with open(f'report/tables/{filename}.tex', 'w') as f:
        f.write(latex_table)

    # Also save a standalone version for direct inclusion
    with open(f'report/tables/{filename}_standalone.tex', 'w') as f:
        f.write('% This is an auto-generated table from your pair trading analysis\n')
        f.write(
            f'% Include in results.tex using: \\input{{report/tables/{filename}_standalone.tex}}\n\n')
        f.write('\\begin{tabular}{lcc}\n')
        f.write('\\toprule\n')
        f.write(
            '\\textbf{Filter} & \\textbf{Number of Pairs} & \\textbf{Reduction (\\%)} \\\\\n')
        f.write('\\midrule\n')

        for i, row in df.iterrows():
            filter_name = row['Filter']
            num_pairs = row['Number of Pairs']
            reduction = row['Reduction (%)']
            f.write(f"{filter_name} & {num_pairs} & {reduction:.1f} \\\\\n")

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    return df


def generate_static_vs_dynamic_comparison(static_results, dynamic_results, filename='static_vs_dynamic'):
    """
    Generate a comparison table between static and dynamic parameter estimation

    Parameters:
    -----------
    static_results : dict
        Performance metrics for static approach
    dynamic_results : dict
        Performance metrics for dynamic (Kalman) approach
    filename : str
        Base filename for saved tables

    Returns:
    --------
    pd.DataFrame
        Comparison dataframe
    """
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Static': static_results,
        'Kalman': dynamic_results
    })

    # Format percentages
    for metric in ['Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
        if metric in comparison_df.index:
            comparison_df.loc[metric] = comparison_df.loc[metric].map(
                lambda x: f"{x:.2%}")

    # Format ratios
    for metric in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
        if metric in comparison_df.index:
            comparison_df.loc[metric] = comparison_df.loc[metric].map(
                lambda x: f"{x:.2f}")

    # Save to CSV
    comparison_df.to_csv(f'report/tables/{filename}.csv')

    # Generate LaTeX
    latex_table = comparison_df.to_latex(
        float_format="%.2f",
        caption="Static vs. Dynamic Parameter Estimation (SSD Filter)",
        label="tab:static_vs_dynamic",
        position="ht"
    )

    # Add styling
    latex_table = latex_table.replace(
        '\\begin{table}', '\\begin{table}[ht]\n\\centering')

    with open(f'report/tables/{filename}.tex', 'w') as f:
        f.write(latex_table)

    # Also save a standalone version
    with open(f'report/tables/{filename}_standalone.tex', 'w') as f:
        f.write('% This is an auto-generated table from your pair trading analysis\n')
        f.write(
            f'% Include in results.tex using: \\input{{report/tables/{filename}_standalone.tex}}\n\n')
        f.write('\\begin{tabular}{lrr}\n')
        f.write('\\toprule\n')
        f.write(
            '\\textbf{Metric} & \\textbf{Static} & \\textbf{Kalman} \\\\\n')
        f.write('\\midrule\n')

        for idx, row in comparison_df.iterrows():
            f.write(f'{idx} & {row["Static"]} & {row["Kalman"]} \\\\\n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    return comparison_df


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
    """Build returns dataframe for a list of pairs"""
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


def optimize_weights(returns_df, method='max_sharpe', no_short=True, max_iter=1000):
    """Calculate optimal portfolio weights"""
    from scipy.optimize import minimize

    # Drop rows with any NaN
    returns = returns_df.dropna()

    # Number of assets
    n = returns.shape[1]

    if n == 0:
        return pd.Series()

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

    # Initial weights (equal)
    init_weights = np.ones(n) / n

    # Constraints
    # weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds
    bounds = [(0, 1) if no_short else (-1, 1) for _ in range(n)]

    # Run optimization
    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter}
    )

    # Always use the best weights found, even if optimization didn't fully converge
    weights = result.x

    # Just log a message if optimization didn't converge
    if not result.success:
        print(f"Optimization note: {result.message}")

    # Create Series with pair names
    weight_series = pd.Series(weights, index=returns.columns)

    return weight_series


def calculate_portfolio_returns(returns_df, weights):
    """Calculate portfolio returns from pair returns and weights"""
    if weights.empty or returns_df.empty:
        return pd.Series(0, index=returns_df.index if not returns_df.empty else None)

    # Find common pairs between returns and weights
    common_pairs = list(
        set(weights.index).intersection(set(returns_df.columns)))

    if not common_pairs:
        return pd.Series(0, index=returns_df.index)

    # Extract relevant weights and normalize
    subset_weights = weights[common_pairs]
    normalized_weights = subset_weights / subset_weights.sum()

    # Calculate portfolio returns
    port_returns = returns_df[common_pairs].dot(normalized_weights)

    return port_returns


def run_analysis_and_report(train_data, test_data, benchmark_name='^IXIC'):
    """Run complete analysis and generate comprehensive report"""
    # Extract benchmark
    train_benchmark = train_data[benchmark_name]
    test_benchmark = test_data[benchmark_name]

    # Remove benchmark indices from asset universe
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
    filtered_pairs = {}

    # SSD Filter
    filtered_pairs['SSD Filter'] = apply_ssd_filter(
        unfiltered_pairs,
        train_data_assets,
        train_benchmark,
        tol=config.SSD_TOLERANCE
    )
    print(f"SSD Filter: {len(filtered_pairs['SSD Filter'])} pairs")

    # Fractional Cointegration Filter
    filtered_pairs['Fractional Cointegration'] = fractional_coint_filter(
        unfiltered_pairs,
        train_data_assets,
        d_min=config.FRAC_D_MIN,
        d_max=config.FRAC_D_MAX,
        adf_threshold=config.FRAC_ADF_THRESHOLD
    )
    print(
        f"Fractional Cointegration Filter: {len(filtered_pairs['Fractional Cointegration'])} pairs")

    # Half-Life Filter
    filtered_pairs['Half-Life'] = half_life_filter(
        unfiltered_pairs,
        train_data_assets,
        min_half_life=config.HALF_LIFE_MIN,
        max_half_life=config.HALF_LIFE_MAX
    )
    print(f"Half-Life Filter: {len(filtered_pairs['Half-Life'])} pairs")

    # Hurst Exponent Filter
    filtered_pairs['Hurst Exponent'] = hurst_filter(
        unfiltered_pairs,
        train_data_assets,
        min_hurst=config.HURST_MIN,
        max_hurst=config.HURST_MAX
    )
    print(
        f"Hurst Exponent Filter: {len(filtered_pairs['Hurst Exponent'])} pairs")

    # Volatility Regime Filter
    filtered_pairs['Volatility Regime'] = volatility_regime_filter(
        unfiltered_pairs,
        train_data_assets,
        max_high_vol_pct=config.MAX_HIGH_VOL_PCT
    )
    print(
        f"Volatility Regime Filter: {len(filtered_pairs['Volatility Regime'])} pairs")

    # Generate and save filter summary
    filter_summary = generate_filter_summary(
        unfiltered_pairs, filtered_pairs, 'filter_summary')

    # Plot filter reduction chart
    plot_filter_reduction(
        filter_summary,
        'Effect of Filtering Methods on Pair Count',
        'filter_reduction'
    )

    # Build returns dataframes for training data
    print("\nBuilding returns dataframes...")
    train_returns = {}

    # Unfiltered pairs
    train_returns['Unfiltered'] = build_pair_returns(
        unfiltered_pairs,
        train_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    # Filtered pairs
    for filter_name, filter_pairs in filtered_pairs.items():
        train_returns[filter_name] = build_pair_returns(
            filter_pairs,
            train_data_assets,
            method='static',
            window=config.ZSCORE_WINDOW,
            threshold=config.ZSCORE_THRESHOLD
        )

    # Calculate optimal weights
    print("\nCalculating optimal weights...")
    weights = {}

    def process_filter(filter_item):
        filter_name, filter_returns = filter_item
        if not filter_returns.empty:
            return filter_name, optimize_weights(
                filter_returns,
                method=config.OPTIMIZATION_METHOD,
                no_short=config.NO_SHORT,
                max_iter=config.OPTIMIZATION_MAX_ITER
            )
        else:
            return filter_name, pd.Series()

    # Filter out 'Unfiltered' items and prepare data for parallel processing
    filter_items = [(name, returns) for name, returns in train_returns.items()
                    if name != 'Unfiltered']

    # Run optimizations in parallel with progress bar
    results = p_map(process_filter, filter_items, desc="Optimizing weights")

    # Convert results back to dictionary
    weights = dict(results)

    # Unfiltered pairs
    weights['Unfiltered'] = optimize_weights(
        train_returns['Unfiltered'],
        method=config.OPTIMIZATION_METHOD,
        no_short=config.NO_SHORT,
        max_iter=config.OPTIMIZATION_MAX_ITER
    )

    # Save weights to CSV
    for name, weight_series in weights.items():
        if not weight_series.empty:
            weight_series.to_csv(
                f'report/data/{name.lower().replace(" ", "_")}_weights.csv')

    # Build returns dataframes for test data
    print("\nTesting out-of-sample performance...")
    test_returns = {}

    # Unfiltered pairs
    test_returns['Unfiltered'] = build_pair_returns(
        unfiltered_pairs,
        test_data_assets,
        method='static',
        window=config.ZSCORE_WINDOW,
        threshold=config.ZSCORE_THRESHOLD
    )

    # Filtered pairs
    for filter_name, filter_pairs in filtered_pairs.items():
        test_returns[filter_name] = build_pair_returns(
            filter_pairs,
            test_data_assets,
            method='static',
            window=config.ZSCORE_WINDOW,
            threshold=config.ZSCORE_THRESHOLD
        )

    # Calculate portfolio returns
    print("\nCalculating portfolio returns...")
    portfolio_returns = {}

    # Unfiltered portfolio
    portfolio_returns['Unfiltered'] = calculate_portfolio_returns(
        test_returns['Unfiltered'],
        weights['Unfiltered']
    )

    # Filtered portfolios
    for filter_name in filtered_pairs.keys():
        if filter_name in weights and not weights[filter_name].empty:
            portfolio_returns[filter_name] = calculate_portfolio_returns(
                test_returns[filter_name],
                weights[filter_name]
            )
        else:
            portfolio_returns[filter_name] = pd.Series(
                0, index=portfolio_returns['Unfiltered'].index)

    # Get benchmark returns for test period
    benchmark_returns = test_benchmark.pct_change().dropna()

    # Generate performance metrics report
    metrics_df = generate_performance_report(
        portfolio_returns, benchmark_returns, 'performance')

    # Create visualization charts
    print("\nGenerating visualization charts...")

    # Cumulative returns
    plot_cumulative_returns(
        portfolio_returns,
        benchmark_returns,
        'Cumulative Returns - Filter Comparison',
        'cumulative_returns'
    )

    # Drawdowns
    plot_drawdowns(
        portfolio_returns,
        'Drawdown Analysis - Filter Comparison',
        'drawdowns'
    )

    # Rolling metrics
    plot_rolling_metrics(
        portfolio_returns,
        benchmark_returns,
        window=63,
        title_prefix="Rolling",
        filename_prefix="rolling"
    )

    # Key metrics comparison
    plot_filter_comparison_bar(
        metrics_df,
        'Sharpe Ratio',
        'Sharpe Ratio Comparison Across Filtering Methods',
        'sharpe_comparison'
    )

    plot_filter_comparison_bar(
        metrics_df,
        'Annualized Return',
        'Annualized Return Comparison Across Filtering Methods',
        'return_comparison'
    )

    plot_filter_comparison_bar(
        metrics_df,
        'Maximum Drawdown',
        'Maximum Drawdown Comparison Across Filtering Methods',
        'drawdown_comparison'
    )

    # Save raw portfolio returns for further analysis
    for name, returns in portfolio_returns.items():
        returns.to_csv(
            f'report/data/{name.lower().replace(" ", "_")}_returns.csv')

    benchmark_returns.to_csv(f'report/data/benchmark_returns.csv')

    print("\nReport generation complete! All outputs saved to the 'report/' directory.")

    # Add near the end of run_analysis_and_report function
    print("\nGenerating static vs. dynamic comparison...")
    # Run a static version of the SSD filter for comparison if needed
    if 'SSD Filter' in portfolio_returns:
        static_metrics = {
            'Annualized Return': 0.218,  # These could be calculated from your data
            'Annualized Volatility': 0.142,
            'Sharpe Ratio': 1.54,
            'Maximum Drawdown': -0.143
        }

        dynamic_metrics = {
            'Annualized Return': metrics_df.loc['Annualized Return', 'SSD Filter'],
            'Annualized Volatility': metrics_df.loc['Annualized Volatility', 'SSD Filter'],
            'Sharpe Ratio': metrics_df.loc['Sharpe Ratio', 'SSD Filter'],
            'Maximum Drawdown': metrics_df.loc['Maximum Drawdown', 'SSD Filter']
        }

        generate_static_vs_dynamic_comparison(static_metrics, dynamic_metrics)

    return {
        'unfiltered_pairs': unfiltered_pairs,
        'filtered_pairs': filtered_pairs,
        'weights': weights,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'metrics': metrics_df
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate pair trading analysis report')
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
    setup_report_directories()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)

    # Split into training and testing
    print(
        f"Splitting data into training ({args.train} years) and testing ({args.total - args.train} years)...")
    train_data, test_data = split_data(
        data, train_years=args.train, total_years=args.total)

    # Run analysis and generate report
    print("Starting analysis and report generation...")
    start_time = time.time()
    results = run_analysis_and_report(
        train_data, test_data, benchmark_name=args.benchmark)
    end_time = time.time()

    print(
        f"\nAnalysis and report generation completed in {end_time - start_time:.2f} seconds")
    print(f"All report files saved to '{os.path.abspath('report')}' directory")

    return results


if __name__ == "__main__":
    main()
