
# Pairs Trading Framework with Enhanced Filtering Techniques

This project implements a comprehensive pairs trading framework with multiple advanced filtering techniques to enhance performance. The framework incorporates:

1. Second-Order Stochastic Dominance (SSD) Filter
2. Fractional Cointegration Filter
3. Half-Life Filter
4. Hurst Exponent Filter
5. Volatility Regime Filter

## Project Structure

```

pair_trading/
├── data/                      \# Data directory
├── filters/                   \# Filtering algorithms
├── models/                    \# Trading models
├── utils/                     \# Utility functions
├── config.py                  \# Configuration parameters
├── main.py                    \# Main entry point
└── README.md                  \# Project documentation

```

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Matplotlib
- Statsmodels
- Scikit-learn
- Joblib
- tqdm

### Installation

```


# Clone the repository

git clone https://github.com/yourusername/pair-trading.git
cd pair-trading

# Create a virtual environment

python -m venv venv
source venv/bin/activate  \# On Windows: venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

```

### Usage

1. Place your data in the `data/` directory
2. Configure parameters in `config.py`
3. Run the backtest:

```

python main.py --data data/stock_data.csv --benchmark ^IXIC --train 3 --total 5

```

## Filters Explanation

### Second-Order Stochastic Dominance (SSD) Filter

Filters pairs based on whether their return distribution is SSD efficient relative to a benchmark, ensuring the strategy provides a risk-reward profile at least as good as the benchmark.

### Fractional Cointegration Filter

Extends traditional cointegration to allow for long memory processes, better capturing the mean-reversion properties of certain pairs where the mean-reversion speed varies over time.

### Half-Life Filter

Selects pairs with optimal mean-reversion speed, avoiding pairs that revert too quickly (noisy) or too slowly (inefficient capital usage).

### Hurst Exponent Filter

Uses the Hurst exponent to identify truly mean-reverting series. H &lt; 0.5 indicates mean-reversion, H = 0.5 indicates random walk, and H &gt; 0.5 indicates trend persistence.

### Volatility Regime Filter

Dynamically adjusts strategy parameters based on detected volatility regimes, and filters out pairs that perform poorly in high volatility environments.

## Performance Evaluation

The framework automatically evaluates the performance of each filtering method against an unfiltered baseline and a benchmark index. Performance metrics include:

- Total Return
- Annualized Return
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio

Results are saved to the `results/` directory as both CSV files and plots.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```
