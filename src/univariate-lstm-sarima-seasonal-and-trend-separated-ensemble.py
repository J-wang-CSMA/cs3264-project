import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from captum.attr import IntegratedGradients

hidden_states = []


def hook_fn(module, input, output):
    hidden_states.append(output[0].detach().cpu().numpy())


# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, header=0, index_col=0)
    df.columns = df.columns.str.strip()  # Clean column names
    df = df[df.columns[::-1]]  # Reverse columns so dates are ascending
    row_label = "Total International Visitor Arrivals By Inbound Tourism Markets"
    if row_label not in df.index:
        raise ValueError(f"Row '{row_label}' not found.")
    total_series = df.loc[row_label]
    total_df = total_series.reset_index()
    total_df.columns = ['Date', 'TotalVisitors']
    total_df['Date'] = pd.to_datetime(total_df['Date'], format='%Y %b')
    total_df = total_df.sort_values('Date')
    total_df['TotalVisitors'] = pd.to_numeric(total_df['TotalVisitors'], errors='coerce')
    return total_df.dropna(subset=['TotalVisitors'])


# Split data into training and test sets
def split_data(total_df):
    train = total_df[(total_df['Date'] >= '1980-01-01') & (total_df['Date'] <= '2000-12-31')]
    test = total_df[(total_df['Date'] >= '2001-01-01') & (total_df['Date'] <= '2024-12-31')]
    return train, test


# Create sequences for model training
def create_sequences(data, lookback=12):
    xs, ys = [], []
    for i in range(len(data) - lookback):
        xs.append(data[i:i + lookback])
        ys.append(data[i + lookback])
    return np.array(xs), np.array(ys)


# Define an LSTM model for the deseasonalized (long range trend) series
class TrendLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(TrendLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64)
        c0 = torch.zeros(1, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def main():
    # Load and split the data
    filepath = '../Datasets/international_visitor_arrivals_by_country.csv'
    total_df = load_and_preprocess_data(filepath)
    train_df, test_df = split_data(total_df)

    # Convert series and set monthly frequency
    y_train = train_df.set_index('Date')['TotalVisitors']
    y_test = test_df.set_index('Date')['TotalVisitors']
    y_train.index = pd.DatetimeIndex(y_train.index).to_period('M').to_timestamp();
    y_train.index.freq = 'MS'
    y_test.index = pd.DatetimeIndex(y_test.index).to_period('M').to_timestamp();
    y_test.index.freq = 'MS'
    print(test_df)
    # Fit a purely seasonal SARIMA model (captures annual seasonality)
    seasonal_model = SARIMAX(y_train, order=(0, 0, 0), seasonal_order=(1, 1, 1, 12),
                             enforce_stationarity=True, enforce_invertibility=True)
    seasonal_fit = seasonal_model.fit(disp=False)
    seasonal_train_pred = seasonal_fit.predict(start=y_train.index[0],
                                               end=y_train.index[-1], dynamic=False)

    # Deseasonalize training data (trend + residual)
    deseason_train = y_train - seasonal_train_pred

    # Scale deseasonalized training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    deseason_train_scaled = scaler.fit_transform(deseason_train.values.reshape(-1, 1))
    print("Deseasonalized training range:",
          deseason_train.min(), deseason_train.max())
    print("Scaled range:",
          scaler.data_min_, scaler.data_max_)
    # Create training sequences with a 36-month lookback
    lookback = 36
    X_train, y_train_seq = create_sequences(deseason_train_scaled, lookback)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

    # Train the LSTM on deseasonalized (trend) training data
    lstm_model = TrendLSTM(input_size=1, hidden_size=64, num_layers=1, output_size=1)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
    criterion = nn.HuberLoss(delta=0.001)
    epochs = 300
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        output = lstm_model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"LSTM Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.10f}")

    # Forecast seasonal component on the test period using SARIMA
    steps = len(y_test)
    seasonal_forecast_obj = seasonal_fit.get_forecast(steps=steps)
    seasonal_test_pred = seasonal_forecast_obj.predicted_mean

    # Deseasonalize test data and scale it
    deseason_test = y_test - seasonal_test_pred
    deseason_test_scaled = scaler.transform(deseason_test.values.reshape(-1, 1))

    # Create test sequences from deseasonalized test data
    X_test, _ = create_sequences(deseason_test_scaled, lookback)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    lstm_model.eval()
    with torch.no_grad():
        deseason_test_pred_scaled = lstm_model(X_test_tensor).numpy()
    deseason_test_pred = scaler.inverse_transform(deseason_test_pred_scaled).flatten()
    print(deseason_test_pred)
    # Align forecasts: discard first 'lookback' points for seasonal and trend forecasts
    seasonal_test_aligned = seasonal_test_pred.iloc[lookback:]
    alpha = 1.25  # scaling factor for the trend component
    final_forecast = seasonal_test_aligned.values + alpha * deseason_test_pred
    actual_test_aligned = y_test.iloc[lookback:]

    # Plot Seasonal-Only Forecast vs. Actual
    plt.figure(figsize=(20, 6))
    plt.plot(y_test.index, y_test, label='Actual (Test)')
    plt.plot(y_test.index, seasonal_test_pred, label='Seasonal-Only Forecast', linestyle='--')
    plt.title('Seasonal-Only SARIMA vs. Actual (Test)')
    plt.legend()
    plt.show()

    # Plot Hybrid Forecast vs. Actual
    plt.figure(figsize=(20, 6))
    plt.plot(actual_test_aligned.index, actual_test_aligned, label='Actual (Aligned)')
    plt.plot(actual_test_aligned.index, final_forecast, label='Final Hybrid (Seasonal + LSTM Trend)', linestyle='--')
    plt.title('Hybrid Forecast vs. Actual')
    plt.legend()
    plt.show()

    # Combined 4-line plot: Actual, Seasonal, Trend, and Hybrid
    trend_full = np.full(len(y_test), np.nan)
    trend_full[lookback:] = deseason_test_pred
    hybrid_full = np.full(len(y_test), np.nan)
    hybrid_full[lookback:] = final_forecast
    plt.figure(figsize=(20, 8))
    plt.plot(y_test.index, y_test, label='Actual', marker='o', color='blue')
    plt.plot(y_test.index, seasonal_test_pred, label='Seasonal Forecast (SARIMA)', color='red', linestyle='--')
    plt.plot(y_test.index, trend_full, label='Trend Forecast (LSTM)', color='green', linestyle=':')
    plt.plot(y_test.index, hybrid_full, label='Hybrid (Seasonal + Trend)', linewidth=2, color='orange', marker='o')
    plt.title('All Components: Actual, Seasonal, Trend, and Hybrid Forecasts')
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compute breakdown DataFrame with proportional difference (exclude outliers)
    breakdown_df = pd.DataFrame({
        'Date': actual_test_aligned.index,
        'Actual': actual_test_aligned.values,
        'Seasonal': seasonal_test_aligned.values,
        'Deseasonal_LSTM': deseason_test_pred,
        'Hybrid': final_forecast
    })
    breakdown_df['Difference'] = breakdown_df['Actual'] - breakdown_df['Hybrid']
    breakdown_df['Proportional_Difference'] = breakdown_df['Difference'] / breakdown_df['Actual']
    low = breakdown_df['Proportional_Difference'].quantile(0.01)
    high = breakdown_df['Proportional_Difference'].quantile(0.99)
    filtered = breakdown_df[(breakdown_df['Proportional_Difference'] >= low) &
                            (breakdown_df['Proportional_Difference'] <= high)]
    print("Mean Proportional Difference (filtered):", filtered['Proportional_Difference'].mean())

    # Plot proportional difference over time
    plt.figure(figsize=(20, 6))
    plt.plot(breakdown_df['Date'], breakdown_df['Proportional_Difference'], marker='o', linestyle='-', color='purple')
    plt.title('Proportional Difference (Actual - Hybrid) Over Time')
    plt.xlabel('Date')
    plt.ylabel('Proportional Difference')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
