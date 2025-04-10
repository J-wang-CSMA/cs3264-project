import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.lstm import TrendLSTM  # or your multivariate LSTM model if named differently
from preprocessing_and_feature_engineering.feature_engineering import create_sequences
from preprocessing_and_feature_engineering.preprocessing import set_series_monthly_frequency
from models.seasonal_sarima import deseasonalize_data

def train_model(train_df, feature_cols, target_col, lookback, hyperparams):
    # Extract the training data for the chosen features
    train_values = train_df[feature_cols].values
    print("train_values shape:", train_values.shape)
    # Create sequences; Y will be the row at time t+lookback (all features)
    X_train, Y_train = create_sequences(train_values, lookback)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    # Extract the target column from Y_train (prediction target)
    target_index = feature_cols.index(target_col)

    y_train = Y_train[:, target_index].reshape(-1, 1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Use hyperparameter values (or defaults)
    input_size = hyperparams.get('input_size', len(feature_cols))
    hidden_size = hyperparams.get('hidden_size', 70)
    num_layers = hyperparams.get('num_layers', 1)
    output_size = hyperparams.get('output_size', 1)
    num_epochs = hyperparams.get('num_epochs', 500)
    learning_rate = hyperparams.get('learning_rate', 0.005)
    optimizer_class = hyperparams.get('optimizer_class', optim.Adam)
    criterion_class = hyperparams.get('criterion_class', nn.MSELoss)
    criterion_params = hyperparams.get('criterion_params', {})

    # Instantiate the model, loss function, and optimizer
    model = TrendLSTM(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=output_size)
    criterion = criterion_class(**criterion_params)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.10f}")

    return model


def test_model(model, test_df, feature_cols, target_col, lookback, scaler=None,
               deseasonalize=True, seasonal_fit=None, alpha=1.25):
    if not deseasonalize:
        test_values = test_df[feature_cols].values
        X_test, Y_test = create_sequences(test_values, lookback)
        target_index = feature_cols.index(target_col)
        y_test = Y_test[:, target_index].reshape(-1, 1)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()

        if scaler is not None:
            dummy_pred = np.zeros((y_pred.shape[0], len(feature_cols)))
            dummy_actual = np.zeros((y_test.shape[0], len(feature_cols)))
            dummy_pred[:, target_index] = y_pred.flatten()
            dummy_actual[:, target_index] = y_test.flatten()
            inv_pred = scaler.inverse_transform(dummy_pred)
            inv_actual = scaler.inverse_transform(dummy_actual)
            y_pred = inv_pred[:, target_index].reshape(-1, 1)
            y_test = inv_actual[:, target_index].reshape(-1, 1)

        return y_pred, y_test

    else:
        if seasonal_fit is None or scaler is None:
            raise ValueError("For deseasonalized testing, both seasonal_fit and scaler must be provided.")

        y_test_series = test_df.set_index('Date')[target_col]

        steps = len(y_test_series)
        seasonal_forecast_obj = seasonal_fit.get_forecast(steps=steps)
        seasonal_test_pred = seasonal_forecast_obj.predicted_mean

        deseason_test = y_test_series - seasonal_test_pred
        deseason_test_scaled = scaler.transform(deseason_test.values.reshape(-1, 1))

        X_test, _ = create_sequences(deseason_test_scaled, lookback)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            trend_pred_scaled = model(X_test_tensor).numpy()
        trend_pred = scaler.inverse_transform(trend_pred_scaled).flatten()

        seasonal_test_aligned = seasonal_test_pred.iloc[lookback:]
        final_forecast = seasonal_test_aligned.values + alpha * trend_pred

        # Actual test values (aligned to final forecast)
        actual_test = y_test_series.iloc[lookback:]
        return final_forecast, actual_test.values, trend_pred


def train_model_deseasonalized(train_df, target_col, lookback, hyperparams):
    y_train_series = set_series_monthly_frequency(train_df.set_index('Date')[target_col])

    seasonal_results = deseasonalize_data(y_train_series, y_train_series)
    seasonal_fit, seasonal_train_pred, deseason_train, _, _ = seasonal_results

    scaler_deseason = MinMaxScaler(feature_range=(0, 1))
    deseason_train_scaled = scaler_deseason.fit_transform(deseason_train.values.reshape(-1, 1))
    X_train, y_train_seq = create_sequences(deseason_train_scaled, lookback)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

    hidden_size = hyperparams.get('hidden_size', 70)
    num_layers = hyperparams.get('num_layers', 1)
    output_size = hyperparams.get('output_size', 1)
    num_epochs = hyperparams.get('num_epochs', 500)
    learning_rate = hyperparams.get('learning_rate', 0.005)
    optimizer_class = hyperparams.get('optimizer_class', optim.Adam)
    criterion_class = hyperparams.get('criterion_class', nn.MSELoss)
    criterion_params = hyperparams.get('criterion_params', {})

    model = TrendLSTM(input_size=1, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=output_size)
    criterion = criterion_class(**criterion_params)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Deseasonalized Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.10f}")

    return model, scaler_deseason, seasonal_fit