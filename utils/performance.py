import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def portfolio_statistics(returns, risk_free_rate=0.0, annualization_factor=252):
    """
    Calculate portfolio performance statistics

    Parameters:
    -----------
    returns : pd.Series
        Daily returns of the portfolio
    risk_free_rate : float
        Daily risk-free rate
    annualization_factor : int
        Number of trading days in a year

    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Convert to numpy array
    returns_array = returns.values

    # Calculate metrics
    total_return = np.prod(1 + returns_array) - 1
    annualized_return = (
        1 + total_return) ** (annualization_factor / len(returns)) - 1

    volatility = returns.std() * np.sqrt(annualization_factor)

    excess_returns = returns - risk_free_rate
    sharpe_ratio = (excess_returns.mean() * annualization_factor) / \
        volatility if volatility != 0 else 0

    # Calculate drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()

    # Calculate downside deviation
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std(
    ) * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0

    # Sortino ratio
    sortino_ratio = (excess_returns.mean() * annualization_factor) / \
        downside_deviation if downside_deviation != 0 else 0

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'Calmar Ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    }


def plot_performance_comparison(returns_dict, benchmark_returns=None, title="Strategy Performance"):
    """
    Plot cumulative returns of different strategies

    Parameters:
    -----------
    returns_dict : dict
        Dictionary of {strategy_name: returns_series}
    benchmark_returns : pd.Series, optional
        Returns of benchmark for comparison
    title : str
        Plot title

    Returns:
    --------
    matplotlib.figure.Figure
        Plot figure
    """
    plt.figure(figsize=(12, 6))

    # Plot each strategy's cumulative returns
    for name, returns in returns_dict.items():
        cum_returns = (1 + returns).cumprod()
        plt.plot(cum_returns.index, cum_returns, label=name)

    # Add benchmark if provided
    if benchmark_returns is not None:
        cum_benchmark = (1 + benchmark_returns).cumprod()
        plt.plot(cum_benchmark.index, cum_benchmark,
                 label='Benchmark', linestyle='--')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.legend()

    return plt.gcf()


def print_performance_table(strategies_dict, benchmark_returns=None):
    """
    Print a comparison table of performance metrics

    Parameters:
    -----------
    strategies_dict : dict
        Dictionary of {strategy_name: returns_series}
    benchmark_returns : pd.Series, optional
        Returns of benchmark for comparison

    Returns:
    --------
    pd.DataFrame
        Performance metrics table
    """
    # Add benchmark to dictionary if provided
    if benchmark_returns is not None:
        strategies_dict['Benchmark'] = benchmark_returns

    # Calculate statistics for each strategy
    stats = {}

    for name, returns in strategies_dict.items():
        stats[name] = portfolio_statistics(returns)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(stats)

    # Format percentages
    for metric in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
        metrics_df.loc[metric] = metrics_df.loc[metric].map(
            lambda x: f"{x:.2%}")

    # Format ratios
    for metric in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
        metrics_df.loc[metric] = metrics_df.loc[metric].map(
            lambda x: f"{x:.2f}")

    return metrics_df
