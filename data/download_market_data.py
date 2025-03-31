import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import argparse
import os


def download_stock_data(tickers, start_date, end_date, output_file):
    """
    Download historical stock data for given tickers and save to CSV.
    """
    print(
        f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")

    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)

    # Use Adjusted Close prices
    if 'Adj Close' in data.columns.levels[0]:
        prices = data['Adj Close']
    else:
        prices = data['Close']

    # Forward fill missing values (weekends, holidays)
    prices = prices.ffill()

    # Remove columns with all NaN values
    prices = prices.dropna(axis=1, how='all')

    # Save to CSV
    prices.to_csv(output_file)
    print(f"Data saved to {output_file}")

    return prices


def split_data(data, train_years, test_years, train_file, test_file):
    """
    Split data into training and testing periods.
    """
    # Convert index to datetime if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Calculate split point (end date - test_years)
    split_date = data.index[-1] - timedelta(days=test_years*365)

    # Split data
    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]

    # Save to CSV
    train_data.to_csv(train_file)
    test_data.to_csv(test_file)

    print(
        f"Data split into {train_file} ({train_years} years) and {test_file} ({test_years} years)")


tickers = [
    'IBM', 'BSX', 'KKR', 'QCOM', 'MRK', 'ICE', 'RY', 'ISRG', 'AXP', 'ORCL', 'AMAT', 'MSFT', 'SYK', 'RCL', 'TXN', 'CSCO', 'MCO', 'RDDT', 'PINS', 'BMY', 'TMUS', 'BAC', 'ZTS', 'CRM', 'AMZN', 'WMG', 'OMC', 'MA', 'HLT', 'LULU', 'JNJ', 'ADI', 'IBKR', 'META', 'CB', 'DHR', 'ABNB', 'BKNG', 'NWSA', 'EA', 'MELI', 'BLK', 'FWONA', 'LLY', 'GS', 'PFE', 'TD', 'SCHW', 'AMGN', 'CMCSA', 'FLUT', 'MCK', 'SPGI', 'ADP', 'CPNG', 'GILD', 'TJX', 'BRK-B', 'NOW', 'CMG', 'WBD', 'BX', 'CVS', 'UNH', 'AZO', 'TMO', 'MS', 'MMC', 'AVGO', 'T', 'V', 'ACN', 'AMD', 'BDX', 'DASH', 'NVDA', 'CI', 'CME', 'TTWO', 'ANET', 'YUM', 'SPOT', 'MCD', 'DHI', 'ELV', 'GSAT', 'FOXA', 'LYV', 'TSLA', 'AON', 'INTU', 'ABT', 'ORLY', 'PGR', 'CHTR', 'C', 'HD', 'AAPL', 'MU', 'ABBV', 'UBER', 'VZ', 'WFC', 'LOW', 'PLTR', 'LRCX', 'RBLX', 'GM', 'MDT', 'ZG', 'F', 'KLAC', 'JPM', 'DIS', 'NKE', 'FI', 'NFLX', 'HCA', 'MAR', 'SBUX', 'REGN', 'ADBE', 'ROST', 'GOOG', 'VRTX'
]


def main():
    parser = argparse.ArgumentParser(
        description='Download stock data for pairs trading')

    parser.add_argument('--stocks', type=str, nargs='+',
                        default=[
                            'IBM', 'BSX', 'KKR', 'QCOM', 'MRK', 'ICE', 'RY', 'ISRG', 'AXP', 'ORCL', 'AMAT', 'MSFT', 'SYK', 'RCL', 'TXN', 'CSCO', 'MCO', 'RDDT', 'PINS', 'BMY', 'TMUS', 'BAC', 'ZTS', 'CRM', 'AMZN', 'WMG', 'OMC', 'MA', 'HLT', 'LULU', 'JNJ', 'ADI', 'IBKR', 'META', 'CB', 'DHR', 'ABNB', 'BKNG', 'NWSA', 'EA', 'MELI', 'BLK', 'FWONA', 'LLY', 'GS', 'PFE', 'TD', 'SCHW', 'AMGN', 'CMCSA', 'FLUT', 'MCK', 'SPGI', 'ADP', 'CPNG', 'GILD', 'TJX', 'BRK-B', 'NOW', 'CMG', 'WBD', 'BX', 'CVS', 'UNH', 'AZO', 'TMO', 'MS', 'MMC', 'AVGO', 'T', 'V', 'ACN', 'AMD', 'BDX', 'DASH', 'NVDA', 'CI', 'CME', 'TTWO', 'ANET', 'YUM', 'SPOT', 'MCD', 'DHI', 'ELV', 'GSAT', 'FOXA', 'LYV', 'TSLA', 'AON', 'INTU', 'ABT', 'ORLY', 'PGR', 'CHTR', 'C', 'HD', 'AAPL', 'MU', 'ABBV', 'UBER', 'VZ', 'WFC', 'LOW', 'PLTR', 'LRCX', 'RBLX', 'GM', 'MDT', 'ZG', 'F', 'KLAC', 'JPM', 'DIS', 'NKE', 'FI', 'NFLX', 'HCA', 'MAR', 'SBUX', 'REGN', 'ADBE', 'ROST', 'GOOG', 'VRTX'
                        ],
                        help='List of stock tickers')

    parser.add_argument('--benchmarks', type=str, nargs='+',
                        default=['^DJI', '^GSPC', '^IXIC'],
                        help='List of benchmark indices (^DJI=Dow Jones, ^GSPC=S&P 500, ^IXIC=NASDAQ)')

    parser.add_argument('--years', type=int, default=5,
                        help='Total number of years of data to download')

    parser.add_argument('--train_years', type=int, default=3,
                        help='Number of years for training data')

    parser.add_argument('--output', type=str, default='stock_data.csv',
                        help='Output file name')

    args = parser.parse_args()

    # Combine stocks and benchmarks - benchmarks should be last columns
    tickers = args.stocks + args.benchmarks

    # Calculate dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.years*365)
                  ).strftime('%Y-%m-%d')

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Download data
    data = download_stock_data(tickers, start_date, end_date, args.output)

    # Split data if train_years is specified
    if args.train_years > 0:
        train_file = 'train_data.csv'
        test_file = 'test_data.csv'
        split_data(data, args.train_years, args.years -
                   args.train_years, train_file, test_file)

    print("\nData ready for pairs trading analysis")


if __name__ == "__main__":
    main()
