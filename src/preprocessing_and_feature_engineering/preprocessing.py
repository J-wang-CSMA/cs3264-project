import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess(filepaths, required_rows, date_format='%Y %b', reverse_columns=True):
    """Load & preprocess CSV, returning a DF with a 'Date' col & one col per required row."""
    df = pd.concat([pd.read_csv(fp, header=0, index_col=0) for fp in filepaths], axis=0, join='inner')
    df.columns = df.columns.str.strip()
    if reverse_columns:
        df = df[df.columns[::-1]]
    missing = [r for r in required_rows if r not in df.index]
    if missing:
        raise ValueError(f"Rows not found: {missing}")
    out_df = df.loc[required_rows].T.reset_index().rename(columns={'index': 'Date'})
    out_df['Date'] = pd.to_datetime(out_df['Date'], format=date_format)
    out_df = out_df.sort_values('Date')
    for col in [c for c in out_df.columns if c != 'Date']:
        out_df[col] = pd.to_numeric(out_df[col], errors='coerce')
    return out_df.dropna(subset=[c for c in out_df.columns if c != 'Date'])

def split_data_by_date(df, date_col='Date',
                       train_start='1990-01-01', train_end='2009-12-31',
                       test_start='2010-01-01', test_end='2025-12-31'):
    """Split df by date range into train & test."""
    train = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)]
    test = df[(df[date_col] >= test_start) & (df[date_col] <= test_end)]
    return train, test


def add_month_sin_cos(df, date_col='Date'):
    """Add cyclical month features."""
    out = df.copy()
    out['month'] = out[date_col].dt.month
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    return out


def scale_features(df, exclude_cols=['Date']):
    """Scale all columns except those in exclude_cols; return scaled df & scaler."""
    out = df.copy()
    feature_cols = [c for c in out.columns if c not in exclude_cols]
    scaler = MinMaxScaler()
    out[feature_cols] = scaler.fit_transform(out[feature_cols])
    return out, scaler


def set_series_monthly_frequency(series):
    series.index = pd.DatetimeIndex(series.index).to_period('M').to_timestamp()
    series.index.freq = 'MS'
    return series


def main_preprocessing(filepaths, required_rows, date_format='%Y %b', reverse_columns=True,
                       apply_month_encoding=True, apply_scaling=True,
                       train_start='1980-01-01', train_end='2000-12-31',
                       test_start='2001-01-01', test_end='2025-12-31',
                       return_raw=False):
    # Load and preprocess the CSV file (raw)
    df = load_and_preprocess(filepaths, required_rows, date_format, reverse_columns)
    # Optionally add cyclical month features
    if apply_month_encoding:
        df = add_month_sin_cos(df, date_col='Date')

    # Save raw copy if needed
    raw_df = df.copy() if return_raw else None

    # Optionally scale features (all columns except 'Date')
    scaler = None
    if apply_scaling:
        df, scaler = scale_features(df, exclude_cols=['Date'])

    # Split the DataFrame by date into train/test sets
    train_df, test_df = split_data_by_date(df, date_col='Date',
                                           train_start=train_start, train_end=train_end,
                                           test_start=test_start, test_end=test_end)
    if return_raw:
        raw_train_df, raw_test_df = split_data_by_date(raw_df, date_col='Date',
                                                       train_start=train_start, train_end=train_end,
                                                       test_start=test_start, test_end=test_end)
        return train_df, test_df, scaler, raw_train_df, raw_test_df
    return train_df, test_df, scaler