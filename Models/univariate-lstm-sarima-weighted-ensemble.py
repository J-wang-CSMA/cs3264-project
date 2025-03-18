import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler


#  Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, header=0, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[df.columns[::-1]]  # reverse columns so dates ascend
    row_label = "Total International Visitor Arrivals By Inbound Tourism Markets"
    if row_label not in df.index:
        raise ValueError(f"Row '{row_label}' not found.")
    total_series = df.loc[row_label]
    print(total_series)
    total_df = total_series.reset_index()
    total_df.columns = ['Date', 'TotalVisitors']
    total_df['Date'] = pd.to_datetime(total_df['Date'], format='%Y %b')
    total_df = total_df.sort_values('Date')
    total_df['TotalVisitors'] = pd.to_numeric(total_df['TotalVisitors'], errors='coerce')
    total_df.dropna(subset=['TotalVisitors'], inplace=True)
    return total_df


# Split data into train/test sets
def split_data(total_df):
    train = total_df[(total_df['Date'] >= '1990-01-01') & (total_df['Date'] <= '2014-12-31')]
    test = total_df[(total_df['Date'] >= '2015-01-01') & (total_df['Date'] <= '2019-12-31')]
    return train, test


# Create sequences for LSTM
def create_sequences(data, lookback):
    xs, ys = [], []
    for i in range(len(data) - lookback):
        xs.append(data[i:i + lookback])
        ys.append(data[i + lookback])
    return np.array(xs), np.array(ys)


# Define LSTM model for raw data
class RawLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(RawLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out)


def main():
    # Load data and split
    filepath = '../Datasets/international_visitor_arrivals_by_country.csv'
    total_df = load_and_preprocess_data(filepath)
    train_df, test_df = split_data(total_df)

    # Create time series for SARIMA
    y_train = train_df.set_index('Date')['TotalVisitors']
    y_test = test_df.set_index('Date')['TotalVisitors']
    y_train.index = pd.DatetimeIndex(y_train.index).to_period('M').to_timestamp()
    y_train.index.freq = 'MS'
    y_test.index = pd.DatetimeIndex(y_test.index).to_period('M').to_timestamp()
    y_test.index.freq = 'MS'

    # Fit SARIMA on training data
    sarima_model = SARIMAX(y_train,
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False,
                           trend='t')
    sarima_fit = sarima_model.fit(disp=False)
    steps = len(y_test)
    sarima_forecast_obj = sarima_fit.get_forecast(steps=steps)
    sarima_forecast = sarima_forecast_obj.predicted_mean

    # LSTM on raw data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_raw = train_df['TotalVisitors'].values.reshape(-1, 1)
    test_raw = test_df['TotalVisitors'].values.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    lookback = 12
    X_train, y_train_seq = create_sequences(train_scaled, lookback)
    X_test, y_test_seq = create_sequences(test_scaled, lookback)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    lstm_model = RawLSTM(input_size=1, hidden_size=64, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
    epochs = 500
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        output = lstm_model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"LSTM Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # LSTM forecast on test data
    lstm_model.eval()
    with torch.no_grad():
        lstm_forecast_scaled = lstm_model(X_test_tensor).numpy()
    lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled).flatten()

    # Align LSTM forecast with test set
    test_dates = test_df['Date'].iloc[lookback:]
    sarima_forecast_aligned = sarima_forecast.iloc[lookback:]

    # Weighted ensemble of SARIMA and LSTM
    alpha = 0.7
    hybrid_forecast = alpha * sarima_forecast_aligned.values + (1 - alpha) * lstm_forecast

    # Plot results
    plt.figure(figsize=(20, 8))
    plt.plot(test_df['Date'], test_df['TotalVisitors'], label='Actual', marker='o')
    plt.plot(test_df['Date'], sarima_forecast, label='SARIMA', marker='x')
    plt.plot(test_dates, lstm_forecast, label='LSTM', marker='d')
    plt.plot(test_dates, hybrid_forecast, label='Hybrid', linestyle='--', linewidth=2)
    plt.title('Hybrid Forecast (Weighted Ensemble) vs. Actual')
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.legend()
    plt.show()

    # Print breakdown
    breakdown_df = pd.DataFrame({
        'Date': test_dates,
        'SARIMA': sarima_forecast_aligned.values,
        'LSTM': lstm_forecast,
        'Hybrid': hybrid_forecast
    })
    print(breakdown_df.head(10))


if __name__ == '__main__':
    main()
