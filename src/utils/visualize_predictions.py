import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_seasonal_vs_actual(dates, actual, seasonal_forecast, title="Seasonal Forecast vs Actual (Test)"):
    plt.figure(figsize=(20, 6))
    plt.plot(dates, actual, label='Actual (Test)')
    plt.plot(dates, seasonal_forecast, label='Seasonal-Only Forecast', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_hybrid_vs_actual(dates, actual, hybrid_forecast, title="Hybrid Forecast vs Actual"):
    plt.figure(figsize=(20, 6))
    plt.plot(dates, actual, label='Actual (Aligned)')
    plt.plot(dates, hybrid_forecast, label='Final Hybrid (Seasonal + LSTM Trend)', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_combined_forecasts(dates, actual, seasonal, trend, hybrid,
                            title="All Components: Actual, Seasonal, Trend, and Hybrid Forecasts"):
    plt.figure(figsize=(20, 8))
    plt.plot(dates, actual, label='Actual', marker='o', color='blue')
    plt.plot(dates, seasonal, label='Seasonal Forecast (SARIMA)', color='red', linestyle='--')
    plt.plot(dates, trend, label='Trend Forecast (LSTM)', color='green', linestyle=':')
    plt.plot(dates, hybrid, label='Hybrid (Seasonal + Trend)', linewidth=2, color='orange', marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_proportional_difference(breakdown_df, title="Proportional Difference (Actual - Hybrid) Over Time"):
    plt.figure(figsize=(20, 12))
    plt.plot(breakdown_df['Date'], breakdown_df['Proportional_Difference'], marker='o', linestyle='-', color='purple')
    print(breakdown_df['Proportional_Difference'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Proportional Difference')
    plt.tight_layout()
    plt.show()
