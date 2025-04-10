import matplotlib.pyplot as plt
from preprocessing_and_feature_engineering.preprocessing import main_preprocessing, set_series_monthly_frequency
from training.training_loop import train_model, train_model_deseasonalized, test_model
import torch
import torch.nn as nn
from src.models.seasonal_sarima import deseasonalize_data
from src.utils.visualize_predictions import (
    plot_seasonal_vs_actual,
    plot_hybrid_vs_actual,
    plot_combined_forecasts,
    plot_proportional_difference
)
import pandas as pd
import numpy as np


def main():
    # 1. Choose the CSV file and required rows.
    filepaths = [
        '../Datasets/international_visitor_arrivals_by_country.csv',
        '../Datasets/exchange_rates_per_currency_unit_to_sgd.csv',
        '../Datasets/consumer_price_index_base_year_2024.csv',
        '../Datasets/food_beverage_services_index_base_year_2017.csv',
        '../Datasets/retail_sales_index_base_year_2017.csv',
    ]
    required_rows = ['Total International Visitor Arrivals By Inbound Tourism Markets']

    # 2. Preprocess the data.
    # main_preprocessing returns both scaled and raw DataFrames.
    train_df, test_df, scaler, raw_train_df, raw_test_df = main_preprocessing(
        filepaths,
        required_rows,
        date_format='%Y %b',
        apply_month_encoding=False,
        apply_scaling=False,
        train_start='1988-01-01', train_end='2000-12-31',
        test_start='2001-01-01', test_end='2025-12-31',
        return_raw=True
    )

    # 3. Choose feature columns and target.
    feature_cols = ['Value', 'month', 'month_sin', 'month_cos']
    target_col = 'Value'
    lookback = 36

    # 4. Set hyperparameters.
    hyperparams = {
        'input_size': len(feature_cols),
        'hidden_size': 64,
        'num_layers': 1,
        'output_size': 1,
        'num_epochs': 300,
        'learning_rate': 0.005,
        'optimizer_class': torch.optim.Adam,
        'criterion_class': nn.HuberLoss,
        'criterion_params': {'delta': 0.001}
    }

    deseasonalize_training = True
    if deseasonalize_training:
        model, scaler_deseason, seasonal_fit = train_model_deseasonalized(train_df, target_col, lookback, hyperparams)
        y_pred, y_test, trend_pred = test_model(model, test_df, feature_cols, target_col, lookback,
                                                scaler=scaler_deseason, deseasonalize=True, seasonal_fit=seasonal_fit)
    else:
        model = train_model(train_df, feature_cols, target_col, lookback, hyperparams)
        y_pred, y_test, _ = test_model(model, test_df, feature_cols, target_col, lookback,
                                       scaler=None, deseasonalize=False)


    # Convert raw test target series to monthly frequency.
    raw_y_test_series = set_series_monthly_frequency(raw_test_df.set_index('Date')[target_col])

    # Compute seasonal forecast on the raw test target series using seasonal_fit.
    steps = len(raw_y_test_series)
    seasonal_forecast_obj = seasonal_fit.get_forecast(steps=steps)
    seasonal_test_pred = seasonal_forecast_obj.predicted_mean
    seasonal_series = pd.Series(seasonal_test_pred, index=seasonal_test_pred.index)

    # Align actual series and seasonal forecast by discarding the first 'lookback' points.
    actual = raw_y_test_series.iloc[lookback:]
    seasonal_aligned = seasonal_series.iloc[lookback:]
    test_dates = actual.index

    # Plot Seasonal vs Actual.
    plot_seasonal_vs_actual(test_dates, actual.values, seasonal_aligned,
                            title="Seasonal Forecast vs Actual (Test)")
    # Plot Hybrid vs Actual.
    plot_hybrid_vs_actual(test_dates, y_test, y_pred, title="Hybrid Forecast vs Actual")
    # Plot Combined Forecasts.
    plot_combined_forecasts(test_dates, actual.values, seasonal_aligned, trend_pred, y_pred,
                            title="All Components: Actual, Seasonal, Trend, and Hybrid Forecasts")

    # Compute proportional difference breakdown and plot.
    breakdown_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actual,
        'Seasonal': seasonal_aligned,
        'Trend': trend_pred,
        'Hybrid': y_pred
    })
    breakdown_df['Difference'] = breakdown_df['Actual'] - breakdown_df['Hybrid']
    breakdown_df['Proportional_Difference'] = breakdown_df['Difference'] / breakdown_df['Actual']
    plot_proportional_difference(breakdown_df, title="Proportional Difference (Actual - Hybrid) Over Time")
    low = breakdown_df['Proportional_Difference'].quantile(0.05)
    high = breakdown_df['Proportional_Difference'].quantile(0.95)
    filtered = breakdown_df[(breakdown_df['Proportional_Difference'] >= low) &
                            (breakdown_df['Proportional_Difference'] <= high)]
    print("Mean Proportional Difference (filtered):", filtered['Proportional_Difference'].mean())


if __name__ == '__main__':
    main()
