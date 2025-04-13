#%%
import pandas as pd
import numpy as np
import re

def preprocess_visitor_arrivals(filepath):
    df = pd.read_csv(filepath)
    df = df.melt(id_vars=['Data Series'], var_name='Date', value_name='Visitors')
    df = df[df['Date'].notna()]
    df = df[df['Date'] != ""]
    df['Date'] = df['Date'].str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b', errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.groupby(['Date', 'Data Series'])['Visitors'].sum().reset_index()
    df = df.pivot(index='Date', columns='Data Series', values='Visitors')
    df = df.reset_index()
    df = df.rename_axis(None, axis=1)
    df.rename(columns=dict(zip(df.columns, map(str.strip, df.columns))), inplace=True)

    return df

def preprocess_exchange_rates(filepath):
    df = pd.read_csv(filepath)
    df = df.melt(id_vars=['Data Series'], var_name='Date', value_name='ExchangeRate')
    df = df[df['Date'].notna()]
    df = df[df['Date'] != ""]
    df['Date'] = df['Date'].str.strip()
    df['ExchangeRate'] = df['ExchangeRate'].replace(r'[^0-9.]+', '', regex=True)
    df['ExchangeRate'] = pd.to_numeric(df['ExchangeRate'], errors='coerce')
    df = df[df['ExchangeRate'].notna()]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b', errors='coerce')
    df = df.groupby(['Date', 'Data Series'])['ExchangeRate'].mean().reset_index()
    df = df.pivot(index='Date', columns='Data Series', values='ExchangeRate')
    df = df.reset_index()
    df = df.rename_axis(None, axis=1)
    df.rename(columns=dict(zip(df.columns, map(str.strip, df.columns))), inplace=True)
    return df

def preprocess_hotel_data(filepath):
    df = pd.read_csv(filepath)
    df = df.melt(id_vars=['Data Series'], var_name='Date', value_name='Value')
    df = df[df['Date'].notna()]
    df = df[df['Date'] != ""]
    df['Date'] = df['Date'].str.strip()
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df[df['Value'].notna()]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b', errors='coerce')
    df = df.groupby(['Date', 'Data Series'])['Value'].mean().reset_index()
    df = df.pivot(index='Date', columns='Data Series', values='Value')
    df = df.reset_index()
    df = df.rename_axis(None, axis=1)
    df.rename(columns=dict(zip(df.columns, map(str.strip, df.columns))), inplace=True)
    return df

def preprocess_airport_data(filepath):
    df = pd.read_csv(filepath)
    df = df.melt(id_vars=['Data Series'], var_name='Date', value_name='Value')
    df = df[df['Date'].notna()]
    df = df[df['Date'] != ""]
    df['Date'] = df['Date'].str.strip()
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df[df['Value'].notna()]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b', errors='coerce')
    df = df.groupby(['Date', 'Data Series'])['Value'].mean().reset_index()
    df = df.pivot(index='Date', columns='Data Series', values='Value')
    df = df.reset_index()
    df = df.rename_axis(None, axis=1)
    df.rename(columns=dict(zip(df.columns, map(str.strip, df.columns))), inplace=True)
    return df


def preprocess_cpi_data(filepath):
    """Preprocesses CPI data from CSV to a usable DataFrame."""
    df = pd.read_csv(filepath)

    # Transpose the DataFrame to make months as rows
    df = df.T
    df.reset_index(inplace=True)

    # Remove the first row (headers) and set the correct column names
    new_header = df.iloc[0]
    new_header.iloc[1:] = new_header.iloc[1:].apply(lambda x: "cpi_" + x.strip())
    df = df[1:]
    df.columns = new_header
    df.rename(columns={'Data Series': 'Date'}, inplace=True)

    # Clean up the 'Date' column
    df.loc[:, 'Date'] = df.loc[:, 'Date'].str.strip()
    df = df[df['Date'].notna()]
    df = df[df['Date'] != ""]

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b', errors='coerce')

    # Convert value columns to numeric
    for col in df.columns[1:]:  # Skip the 'Date' column
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in the data columns
    df = df.dropna(subset=df.columns[1:])

    # Rename the date column to ds for consistency
    df.rename(columns={'Date': 'ds'}, inplace=True)

    return df


def merge_dataframes(visitor_arrivals_df, exchange_rates_df, hotel_df, airport_arrivals_df, airport_departures_df, cpi_df):
    exchange_rates_df = exchange_rates_df.add_prefix('exchange_')
    hotel_df = hotel_df.add_prefix('hotel_')
    airport_arrivals_df = airport_arrivals_df.add_prefix('airport_arrivals_')
    airport_departures_df = airport_departures_df.add_prefix('airport_departures_')

    exchange_rates_df = exchange_rates_df.rename(columns={'exchange_Date': 'Date'})
    hotel_df = hotel_df.rename(columns={'hotel_Date': 'Date'})
    airport_arrivals_df = airport_arrivals_df.rename(columns={'airport_arrivals_Date': 'Date'})
    airport_departures_df = airport_departures_df.rename(columns={'airport_departures_Date': 'Date'})
    cpi_df = cpi_df.rename(columns={'ds': 'Date'})

    merged_df = pd.merge(visitor_arrivals_df, exchange_rates_df, on='Date', how='left')
    merged_df = pd.merge(merged_df, hotel_df, on='Date', how='left')
    merged_df = pd.merge(merged_df, airport_arrivals_df, on='Date', how='left')
    merged_df = pd.merge(merged_df, airport_departures_df, on='Date', how='left')
    merged_df = pd.merge(merged_df, cpi_df, on='Date', how='left')

    # Strip whitespace from column names
    merged_df.columns = [col.strip() for col in merged_df.columns]

    return merged_df

# Example usage:
visitor_arrivals_df = preprocess_visitor_arrivals('./Datasets/international_visitor_arrivals_by_country.csv')
exchange_rates_df = preprocess_exchange_rates('./Datasets/exchange_rates_per_currency_unit_to_sgd.csv')
hotel_df = preprocess_hotel_data('./Datasets/hotel_bookings.csv')
airport_departures_df = preprocess_airport_data('./Datasets/airport_departures_by_country_and_region.csv')
airport_arrivals_df = preprocess_airport_data('./Datasets/airport_arrivals_by_country_and_region.csv')

cpi_df = preprocess_cpi_data('./Datasets/consumer_price_index_base_year_2024.csv')


merged_data = merge_dataframes(visitor_arrivals_df, exchange_rates_df, hotel_df, airport_arrivals_df, airport_departures_df, cpi_df)

print(merged_data.head())
merged_data.to_csv('./Datasets/merged_data.csv', index=False)

print("Merged data saved to 'merged_data.csv'")