import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import os

from models.pairs_generation import find_cointegrated_pairs

# Function to calculate half-life from an AR(1) model of the spread


def calculate_half_life(spread):
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


def calculate_spreads(price_data, pairs_data):
    """Calculate spreads for each pair using price data and pair information"""
    spreads = {}
    for row in pairs_data:
        asset1, asset2, _ = row

        # Calculate hedge ratio using OLS regression
        model = OLS(price_data[asset1], sm.add_constant(price_data[asset2]))
        result = model.fit()
        hedge_ratio = result.params[1]

        # Calculate spread
        spread = price_data[asset1] - hedge_ratio * price_data[asset2]

        # Store with pair name
        pair_name = f"{asset1}_{asset2}"
        spreads[pair_name] = spread

    return spreads

# Main function to generate half-life distribution plot


def plot_half_life_distribution(data_file, pairs_file=None, output_file=None):
    """
    Generate histogram showing distribution of half-life values

    Parameters:
    data_file: CSV file with price data (or pandas DataFrame)
    pairs_file: CSV file with pairs information (optional)
    output_file: Path to save the output plot
    """
    # Load market data
    if isinstance(data_file, str):
        price_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        price_data = data_file

    price_data = price_data.dropna(axis=1).drop(
        columns=['^DJI', '^GSPC', '^IXIC'])

    half_lives = []
    _, pairs_file = find_cointegrated_pairs(price_data)

    # If pairs_file is provided, calculate spreads and half-lives for those pairs
    if pairs_file is not None:
        if isinstance(pairs_file, str):
            pairs_data = pd.read_csv(pairs_file)
        else:
            pairs_data = pairs_file

        spreads = calculate_spreads(price_data, pairs_data)

        # Calculate half-life for each spread
        for pair_name, spread in spreads.items():
            half_life = calculate_half_life(spread)
            if not np.isnan(half_life):
                half_lives.append(half_life)
    else:
        # If no pairs file, test all possible pairs for cointegration
        assets = price_data.columns

        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1, asset2 = assets[i], assets[j]

                # Calculate spread
                model = OLS(price_data[asset1],
                            sm.add_constant(price_data[asset2]))
                result = model.fit()
                hedge_ratio = result.params.iloc[1]
                spread = price_data[asset1] - hedge_ratio * price_data[asset2]

                # Calculate half-life
                half_life = calculate_half_life(spread)
                if not np.isnan(half_life):
                    half_lives.append(half_life)

    # Create the histogram
    plt.figure(figsize=(12, 8))

    # Create bins and plot histogram
    bins = np.linspace(0, 100, 26)  # 0 to 50 days with 25 bins
    n, bins, patches = plt.hist(half_lives, bins=bins, alpha=0.7,
                                color='skyblue', edgecolor='black')

    # Highlight bins in the 20-50 day range
    for i, patch in enumerate(patches):
        if bins[i] >= 20 and bins[i+1] <= 50:
            patch.set_facecolor('green')
            patch.set_alpha(0.8)

    # Add vertical lines at the filter boundaries
    plt.axvline(x=20, color='red', linestyle='--',
                label='Min Threshold (1 day)')
    plt.axvline(x=50, color='red', linestyle='--',
                label='Max Threshold (20 days)')

    # Add labels and title
    plt.xlabel('Half-Life (Days)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Half-Life Values for Cointegrated Pairs', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    # Add statistics in a text box
    total_pairs = len(half_lives)
    eligible_pairs = sum(1 for hl in half_lives if 1 <= hl <= 20)
    reduction_pct = (total_pairs - eligible_pairs) / \
        total_pairs * 100 if total_pairs > 0 else 0

    stats_text = (
        f'Total Pairs: {total_pairs}\n'
        f'Eligible Pairs (20-50 days): {eligible_pairs}\n'
        f'Reduction: {reduction_pct:.1f}%'
    )

    plt.figtext(0.75, 0.8, stats_text, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    return half_lives


# Example usage
if __name__ == "__main__":
    # Replace with your actual data file paths
    market_data_file = "data/stock_data.csv"

    # Generate and display the plot
    plot_half_life_distribution(
        market_data_file, output_file="halflife_distribution.png")
