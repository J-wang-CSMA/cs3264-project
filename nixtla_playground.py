import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS
from neuralforecast.losses.pytorch import MAE
import warnings
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
FILEPATH = '../Datasets/international_visitor_arrivals_by_country.csv'
TARGET_ROW_LABEL = "Total International Visitor Arrivals By Inbound Tourism Markets"
UNIQUE_ID_NAME = "Singapore_Total_Arrivals"
FORECAST_HORIZON = 4
LOOKBACK_WINDOW = 12
MAX_EPOCHS = 300
FREQ = 'MS'

def load_format_for_nixtla(filepath, row_label, unique_id):
    """Load and format the data for Nixtla."""
    try:
        df_full = pd.read_csv(filepath, header=0, index_col=0)
        df_full.columns = df_full.columns.str.strip()

        try:
            date_cols = pd.to_datetime(df_full.columns[:5], format='%Y %b', errors='coerce')
            if date_cols.is_monotonic_decreasing:
                print("Detected descending dates, reversing columns.")
                df_full = df_full[df_full.columns[::-1]]
        except Exception:
            print("Could not automatically determine date order, assuming ascending.")

        if row_label not in df_full.index:
            raise ValueError(f"Row '{row_label}' not found in the CSV index.")

        total_series = df_full.loc[row_label]
        total_df = total_series.reset_index()
        total_df.columns = ['DateStr', 'y']
        total_df['ds'] = pd.to_datetime(total_df['DateStr'], format='%Y %b')
        total_df['y'] = pd.to_numeric(total_df['y'], errors='coerce')
        total_df = total_df.dropna(subset=['ds', 'y']).sort_values('ds')
        total_df['unique_id'] = unique_id
        nixtla_df = total_df[['unique_id', 'ds', 'y']].reset_index(drop=True)
        nixtla_df['y'] = nixtla_df['y'].astype(np.float32)

        print(f"Successfully loaded and formatted data for '{unique_id}'.")
        return nixtla_df

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except ValueError as ve:
        print(f"Error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

def main():
    df = load_format_for_nixtla(FILEPATH, TARGET_ROW_LABEL, UNIQUE_ID_NAME)
    if df is None:
        print("Exiting due to data loading error.")
        exit()

    print("\nLoaded Data Head:")
    print(df.head())
    print("\nLoaded Data Tail:")
    print(df.tail())
    print(f"\nData Frequency: {pd.infer_freq(df['ds'])}")

    if len(df) < FORECAST_HORIZON + LOOKBACK_WINDOW:
        print(f"Error: Not enough data ({len(df)} points) for the specified lookback ({LOOKBACK_WINDOW}) and horizon ({FORECAST_HORIZON}).")
        exit()

    train_df = df[:-FORECAST_HORIZON]
    test_df = df[-FORECAST_HORIZON:]
    print(f"\nTraining data length: {len(train_df)}")
    print(f"Test data length (horizon): {len(test_df)}")

    models = [
        NHITS(h=FORECAST_HORIZON,
              input_size=LOOKBACK_WINDOW,
              max_steps=MAX_EPOCHS,
              loss=MAE(),
              scaler_type='minmax'),
        NBEATS(h=FORECAST_HORIZON,
               input_size=LOOKBACK_WINDOW,
               max_steps=MAX_EPOCHS,
               loss=MAE(),
               scaler_type='minmax')
    ]

    nf = NeuralForecast(models=models, freq=FREQ)

    print("\n--- Training Models ---")
    nf.fit(df=train_df)
    print("--- Training Complete ---")

    print("\n--- Generating Forecast ---")
    predictions_df = nf.predict()
    print("--- Forecast Generated ---")
    print("\nPredictions DataFrame Head:")
    print(predictions_df.head())

    eval_df = pd.merge(test_df[['ds', 'y']], predictions_df, on='ds', how='left')
    print("\nEvaluation DataFrame Head:")
    print(eval_df.head())
    eval_df = eval_df.dropna()

    if len(eval_df) != FORECAST_HORIZON:
        print(f"Warning: Evaluation data length ({len(eval_df)}) doesn't match forecast horizon ({FORECAST_HORIZON}). Check predictions.")

    if not eval_df.empty:
        mae_nhits = mean_absolute_error(eval_df['y'], eval_df['NHITS'])
        mae_nbeats = mean_absolute_error(eval_df['y'], eval_df['NBEATS'])
        print(f"\n--- Evaluation Metrics (MAE over {len(eval_df)} steps) ---")
        print(f"NHITS MAE:  {mae_nhits:.4f}")
        print(f"NBEATS MAE: {mae_nbeats:.4f}")
    else:
        print("\nEvaluation DataFrame is empty. Cannot calculate metrics.")
        mae_nhits, mae_nbeats = np.nan, np.nan

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))
    plt.plot(df['ds'], df['y'], label='Actual Full Series', color='black', linewidth=1.5)
    plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='dodgerblue', linewidth=1.5)
    if 'NHITS' in eval_df.columns:
        plt.plot(eval_df['ds'], eval_df['NHITS'], label=f'NHITS Forecast (MAE: {mae_nhits:.2f})', color='red', linestyle='--', linewidth=1.5)
    if 'NBEATS' in eval_df.columns:
        plt.plot(eval_df['ds'], eval_df['NBEATS'], label=f'NBEATS Forecast (MAE: {mae_nbeats:.2f})', color='limegreen', linestyle='--', linewidth=1.5)
    plt.plot(test_df['ds'], test_df['y'], label='Actual Test Data', color='darkorange', marker='o', linestyle='None', markersize=6)
    plt.title(f'Multi-Step Forecast (H={FORECAST_HORIZON}) vs Actual - {UNIQUE_ID_NAME}')
    plt.xlabel('Date')
    plt.ylabel('Visitor Arrivals')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if not eval_df.empty:
        fig, axs = plt.subplots(len(models), 1, figsize=(12, 5 * len(models)), sharex=True)
        if len(models) == 1:
            axs = [axs]
        for i, model_obj in enumerate(models):
            model_name = type(model_obj).__name__
            if model_name in eval_df.columns:
                mae_val = mean_absolute_error(eval_df["y"], eval_df[model_name])
                axs[i].plot(df['ds'], df['y'], label='Actual Full Series', color='black', linewidth=1, alpha=0.6)
                axs[i].plot(train_df['ds'], train_df['y'], label='Training Data', color='dodgerblue')
                axs[i].plot(eval_df['ds'], eval_df[model_name], label=f'{model_name} Forecast', color='red', linestyle='--')
                axs[i].plot(test_df['ds'], test_df['y'], label='Actual Test Data', color='darkorange', marker='.', linestyle='None')
                axs[i].set_title(f'{model_name} Forecast vs Actual (MAE: {mae_val:.2f})')
                axs[i].set_ylabel('Visitor Arrivals')
                axs[i].legend()
                axs[i].grid(True)
            else:
                axs[i].set_title(f'{model_name} - No predictions found')
        axs[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
