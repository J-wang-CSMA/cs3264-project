import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, iTransformer
from neuralforecast.losses.pytorch import MAE
import warnings
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
FILEPATH = './Datasets/international_visitor_arrivals_by_country.csv'
TARGET_ROW_LABEL = "Total International Visitor Arrivals By Inbound Tourism Markets"
UNIQUE_ID_NAME = "Singapore_Total_Arrivals"
FORECAST_HORIZON = 6
LOOKBACK_WINDOW = 48
MAX_EPOCHS = 300
FREQ = 'MS'


def prepare_merged_df_for_nixtla(merged_df, covariates=None, unique_id="Merged_Data"):
    """Prepares merged_df for Nixtla format with optional covariates."""
    df = merged_df.copy()
    df = df.reset_index()
    df.rename(columns={'Date': 'ds', 'Total International Visitor Arrivals By Inbound Tourism Markets': 'y'}, inplace=True)
    # Convert 'ds' to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    df['unique_id'] = unique_id

    # Include covariates if specified
    if covariates:
        for covariate in covariates:
            if covariate in df.columns:
                df[covariate] = df[covariate].astype(np.float32)
            else:
                print(f"Warning: Covariate '{covariate}' not found in DataFrame.")

        # Ensure correct column order
        cols = ['unique_id', 'ds', 'y'] + covariates
        df = df[cols]
    else:
        df = df[['unique_id', 'ds', 'y']]

    df['y'] = df['y'].astype(np.float32)
    return df


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


def rolling_forecast(df, models, lookback, horizon, freq, covariates=None):
    """Performs rolling forecast with yearly retraining."""
    predictions = []
    actuals = []
    dates = []
    min_max_values = []

    #Ensure the dataframe is sorted by date.
    df = df.sort_values(by = 'ds')
    df = df.reset_index(drop = True)

    yearly_increment = 12
    start_index = lookback + horizon

    start_index =  df[df['ds'].dt.year == 2014].index[-1] + 1  # +1 to include the last day of 1990
    start_index = max(start_index, lookback + horizon)  # Ensure it's not less than the lookback + horizon

    while start_index < len(df) - horizon + 1:
        end_index = min(start_index + yearly_increment, len(df) - horizon + 1)
        for i in range(start_index, end_index):
            train_df = df.iloc[:i].copy()
            test_df = df.iloc[i:i + horizon].copy()

            # Print the training and prediction time frames
            print(f"\nTraining Time Frame: {train_df['ds'].min()} to {train_df['ds'].max()}")
            print(f"Prediction Time Frame: {test_df['ds'].min()} to {test_df['ds'].max()}")

            # Impute missing values within train_df
            for col in train_df.columns:
                if train_df[col].isnull().any():
                    train_df[col] = train_df[col].ffill() # or any other imputation method
                    train_df[col] = train_df[col].bfill() # or any other imputation method
                    test_df[col] = test_df[col].ffill() #apply the same imputation to the test set.
                    test_df[col] = test_df[col].bfill() #apply the same imputation to the test set.
            # train_df.fillna(0, inplace=True) # Fill remaining NaNs with 0
            # test_df.fillna(0, inplace=True) # Fill remaining NaNs with 0
            # Preprocess train_df here (e.g., scaling)
            min_y = train_df['y'].min()
            max_y = train_df['y'].max()
            for col in train_df.select_dtypes(include=np.number).columns:
                min_val = train_df[col].min()
                max_val = train_df[col].max()
                if max_val - min_val == 0:
                    continue
                train_df[col] = (train_df[col] - min_val) / (max_val - min_val)
                test_df[col] = (test_df[col] - min_val) / (max_val - min_val)
            # Store min and max for 'y'
            min_max_values.append((min_y, max_y))

            nf = NeuralForecast(models=models, freq=freq)
            if covariates:
                nf.fit(df=train_df)
            else:
                nf.fit(df=train_df)
            forecast = nf.predict(df=test_df)

            predictions.append(forecast)
            actuals.append(test_df['y'].values)
            dates.append(test_df['ds'].values)

        start_index = end_index

    return predictions, actuals, dates, min_max_values


def main():
    # df = load_format_for_nixtla(FILEPATH, TARGET_ROW_LABEL, UNIQUE_ID_NAME)
    # if df is None:
    #     print("Exiting due to data loading error.")
    #     exit()
    merged_df = pd.read_csv("./Datasets/merged_data.csv")
    covariates = [
        'exchange_US Dollar (Singapore Dollar Per US Dollar)',
        'exchange_Renminbi (Singapore Dollar Per Renminbi)',
        'exchange_Euro (Singapore Dollar Per Euro)', 
        'exchange_Japanese Yen (Singapore Dollar Per 100 Japanese Yen)', 'exchange_Australian Dollar (Singapore Dollar Per Australian Dollar)', 'exchange_Sterling Pound (Singapore Dollar Per Pound Sterling)', 'exchange_Malaysian Ringgit (Singapore Dollar Per Malaysian Ringgit)', 'exchange_Thai Baht (Singapore Dollar Per 100 Thai Baht)', 'exchange_Hong Kong Dollar (Singapore Dollar Per Hong Kong Dollar)', 'exchange_Korean Won (Singapore Dollar Per 100 Korean Won)',
        'hotel_Standard Average Hotel Occupancy Rate (Per Cent)', 'hotel_Standard Average Room Rate (Dollar)',
        'airport_arrivals_Number Of Air Passenger Arrivals', 'airport_departures_Number Of Air Passenger Departures',
        'cpi_Food'
    ]
    merged_df = merged_df.fillna(0, axis=1)
    print("Num null: ", merged_df.describe(), "!")
    # merged_df.fill(axis=0, inplace=True)
    # merged_df.dropna(axis=1, inplace=True)
    df = prepare_merged_df_for_nixtla(merged_df, covariates=covariates, unique_id=UNIQUE_ID_NAME)
    if df is None:
        print("Error: Data preparation failed.")
        exit()
    missing_values = merged_df[covariates].isnull().sum()
    print("Missing:", missing_values)

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
               scaler_type='minmax'),
        iTransformer(h=FORECAST_HORIZON,
                     input_size=LOOKBACK_WINDOW,
                     max_steps=MAX_EPOCHS,
                     loss=MAE(),
                     scaler_type='minmax',
                     n_series=4,),
        # ARIMAWrapper(h=FORECAST_HORIZON)
    ]
    predictions, actuals, dates, min_max_values = rolling_forecast(df, models, LOOKBACK_WINDOW, FORECAST_HORIZON, FREQ, covariates)
    unscaled_predictions = []
    unscaled_actuals = []
    for i, pred in enumerate(predictions):
        min_y, max_y = min_max_values[i]
        unscaled_pred = {}
        for model_name in [type(x).__name__ for x in models]:
            unscaled_pred[model_name] = pred[model_name].values * (max_y - min_y) + min_y
        unscaled_predictions.append(unscaled_pred)
    
    for i, actual in enumerate(actuals):
        min_y, max_y = min_max_values[i]
        unscaled_actual = actual * (max_y - min_y) + min_y
        unscaled_actuals.append(unscaled_actual)

    # Calculate MAE for each forecast
    mae_results = {}
    for model_obj in models:
        model_name = type(model_obj).__name__
        model_predictions = [pred[model_name] for pred in unscaled_predictions]
        mae_values = [mean_absolute_error(unscaled_actuals[i], model_predictions[i]) for i in range(len(unscaled_actuals))]
        mae_results[model_name] = mae_values

    # Calculate overall MAE
    overall_mae = {}
    for model_name, mae_values in mae_results.items():
        overall_mae[model_name] = np.mean(mae_values)
        print(f"{model_name} Overall MAE: {overall_mae[model_name]:.4f}")

    # Plotting (example for NHITS)
    plt.figure(figsize=(15, 6))
    plt.plot(df['ds'], df['y'], label='Actual', color='black')

    nhits_predictions = [pred['NHITS'] for pred in unscaled_predictions]
    nhits_dates = [date[0] for date in dates]

    # Plot each prediction window
    for i, pred_values in enumerate(nhits_predictions):
        plt.plot(dates[i], pred_values, linestyle='--', label=f'NHITS Pred Window {i + 1}')

    plt.legend()
    plt.title('Rolling Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Visitor Arrivals')
    plt.show()

    mae_df = pd.DataFrame(overall_mae.items(), columns=['Model', 'Overall MAE'])
    mae_df.to_csv('mae_results.csv', index=False)

    # Save predictions to a csv file.
    predictions_df = pd.DataFrame()

    for i, date_array in enumerate(dates):
        temp_df = pd.DataFrame()
        temp_df['ds'] = date_array
        temp_df['Actuals'] = unscaled_actuals[i]
        for model_obj in models:
            model_name = type(model_obj).__name__
            temp_df[model_name] = unscaled_predictions[i][model_name]
        predictions_df = pd.concat([predictions_df, temp_df])
    cov_name = "nocovariates" if len(covariates) == 0 else "covariates"
    predictions_df.to_csv(f'./Predictions/predictions_{cov_name}_win{LOOKBACK_WINDOW}_fore{FORECAST_HORIZON}.csv', index=False)

    # Calculate MAE for each forecast
    # mae_results = {}
    # mae_df = pd.DataFrame()  # create empty dataframe.
    # for i, model_obj in enumerate(models):
    #     model_name = type(model_obj).__name__
    #     model_predictions = [pred[model_name] for pred in unscaled_predictions]
    #     mae_values = [mean_absolute_error(unscaled_actuals[i], model_predictions[i]) for i in range(len(unscaled_actuals))]
    #     mae_results[model_name] = mae_values
    #     mae_df[model_name] = mae_values  # add the mae values to the dataframe.
    # mae_df['ds'] = [date[0] for date in dates]  # add the dates to the dataframe.

    # # Calculate overall MAE
    # overall_mae = {}
    # for model_name, mae_values in mae_results.items():
    #     overall_mae[model_name] = np.mean(mae_values)
    #     print(f"{model_name} Overall MAE: {overall_mae[model_name]:.4f}")

    # mae_df.to_csv('mae_over_time.csv', index=False)  # save the mae dataframe to a csv file.

if __name__ == "__main__":
    main()
