"""
Time series data cleaner for preprocessing time series data.
"""

import pandas as pd
import numpy as np

def clean_time_series(data_path):
    """
    Clean and preprocess time series data.
    
    Args:
        data_path: Path to time series data file
        
    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(data_path)
    # Cleaning logic: handle missing values, outliers, etc.
    df = df.fillna(method='ffill')
    return df
