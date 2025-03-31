import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_data(filename):
    """
    Load and preprocess price data from CSV file

    Parameters:
    -----------
    filename : str
        Path to CSV file

    Returns:
    --------
    pd.DataFrame
        Processed price data
    """
    # Load data with datetime index
    df = pd.read_csv(filename, index_col=0, parse_dates=True)

    # Sort by date
    df.sort_index(inplace=True)

    # Remove columns with NaN values
    df.dropna(axis=1, inplace=True)

    return df


def split_data(data, train_years=3, total_years=5):
    """
    Split data into training and testing periods

    Parameters:
    -----------
    data : pd.DataFrame
        Full price data
    train_years : int
        Number of years for training
    total_years : int
        Total number of years in the data

    Returns:
    --------
    tuple
        (training data, testing data)
    """
    split_date = (datetime.now() - timedelta(days=(total_years -
                  train_years)*365)).strftime('%Y-%m-%d')

    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]

    return train_data, test_data


def align_series(series_list):
    """
    Align multiple time series to have the same dates

    Parameters:
    -----------
    series_list : list
        List of pandas Series or DataFrames

    Returns:
    --------
    list
        List of aligned Series or DataFrames
    """
    return [s.reindex(series_list[0].index) for s in series_list]
