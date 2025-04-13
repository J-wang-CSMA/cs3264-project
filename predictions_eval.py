#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import re

csv_files = [
    './Predictions/predictions_covariates_win24_fore6.csv',
    './Predictions/predictions_nocovariates_win24_fore6.csv',
    './Predictions/predictions_covariates_win48_fore2.csv',
    './Predictions/predictions_covariates_win48_fore3.csv',
    './Predictions/predictions_covariates_win48_fore12.csv',
    './Predictions/predictions_covariates_win48_fore6.csv',
    './Predictions/predictions_nocovariates_win48_fore6.csv'
]

all_results = []  # List to store results from each CSV

def extract_metadata(filename):
    """Extracts covariates, window size, and forecast horizon from filename."""
    covariates = "covariates" in filename
    window_match = re.search(r'win(\d+)', filename)
    horizon_match = re.search(r'fore(\d+)', filename)

    window = int(window_match.group(1)) if window_match else None
    horizon = int(horizon_match.group(1)) if horizon_match else None

    return covariates, window, horizon

for csv_file in csv_files:
    try:
        predictions_df = pd.read_csv(csv_file)
        predictions_df['ds'] = pd.to_datetime(predictions_df['ds'])

        # Extract metadata
        covariates, window, horizon = extract_metadata(csv_file)

        # Time Periods
        period1_start = pd.to_datetime('2014-01-01')
        period1_end = pd.to_datetime('2019-12-31')
        period2_start = pd.to_datetime('2020-01-01')
        period2_end = pd.to_datetime('2025-12-31')

        # Filter DataFrames for Each Period
        period1_df = predictions_df[(predictions_df['ds'] >= period1_start) & (predictions_df['ds'] <= period1_end)]
        period2_df = predictions_df[(predictions_df['ds'] >= period2_start) & (predictions_df['ds'] <= period2_end)]

        # Models
        models = ['NHITS', 'NBEATS', 'iTransformer']

        # Calculate MAE, Mean % Diff, Mean % Increase, Mean % Decrease
        results = {}

        for period, df in [('2014-2019', period1_df), ('2020-2025', period2_df)]:
            results[period] = {}
            for model in models:
                actuals = df['Actuals'].values
                predictions = df[model].values

                mae = mean_absolute_error(actuals, predictions)

                # Handle potential division by zero
                percentage_diff = (predictions - actuals) / (actuals + 1e-10) * 100

                # Calculate Mean % Increase
                increase_diffs = percentage_diff[percentage_diff > 0]
                mean_increase = np.mean(increase_diffs) if len(increase_diffs) > 0 else 0

                # Calculate Mean % Decrease
                decrease_diffs = percentage_diff[percentage_diff < 0]
                mean_decrease = np.mean(np.abs(decrease_diffs)) if len(decrease_diffs) > 0 else 0

                results[period][model] = {
                    'MAE': mae,
                    'Mean % Diff': np.mean(np.abs(percentage_diff)),
                    'Mean % Increase': mean_increase,
                    'Mean % Decrease': mean_decrease,
                }

        # Store results for this CSV
        all_results.append({
            'filename': csv_file,
            'results': results,
            'covariates': covariates,
            'window': window,
            'horizon': horizon
        })

        # Print Results for this CSV
        print(f"\nResults for {csv_file}:")
        print(f"  Covariates: {covariates}, Window: {window}, Horizon: {horizon}")
        for period, model_results in results.items():
            print(f"\nResults for {period}:")
            for model, metrics in model_results.items():
                print(f"  {model}:")
                print(f"    MAE: {metrics['MAE']:.4f}")
                print(f"    Mean % Diff: {metrics['Mean % Diff']:.4f}%")
                print(f"    Mean % Increase: {metrics['Mean % Increase']:.4f}%")
                print(f"    Mean % Decrease: {metrics['Mean % Decrease']:.4f}%")

        # Plotting for this CSV
        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df['ds'], predictions_df['Actuals'], label='Actual', color='black')
        plt.plot(predictions_df['ds'], predictions_df['NHITS'], label='NHITS', linestyle='--')
        plt.plot(predictions_df['ds'], predictions_df['NBEATS'], label='NBEATS', linestyle='--')
        plt.plot(predictions_df['ds'], predictions_df['iTransformer'], label='iTransformer', linestyle='--')
        plt.legend()
        plt.title(f'Predictions vs Actual ({csv_file})')
        plt.xlabel('Date')
        plt.ylabel('Visitor Arrivals')
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred while processing '{csv_file}': {e}")

# Create and save a summary DataFrame
summary_data = []
for result in all_results:
    filename = result['filename']
    covariates = result['covariates']
    window = result['window']
    horizon = result['horizon']
    for period, model_results in result['results'].items():
        for model, metrics in model_results.items():
            summary_data.append({
                'filename': filename,
                'period': period,
                'model': model,
                'MAE': metrics['MAE'],
                'Mean % Diff': metrics['Mean % Diff'],
                'Mean % Increase': metrics['Mean % Increase'],
                'Mean % Decrease': metrics['Mean % Decrease'],
                'covariates': covariates,
                'window': window,
                'horizon': horizon
            })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('summary_results.csv', index=False)
print("\nSummary results saved to 'summary_results.csv'")
# %%
