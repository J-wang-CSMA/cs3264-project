import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, header=0, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[df.columns[::-1]]  # reverse columns so dates are ascending
    row_label = "Total International Visitor Arrivals By Inbound Tourism Markets"
    if row_label not in df.index:
        raise ValueError(f"Row '{row_label}' not found.")
    total_series = df.loc[row_label]
    print(total_series)
    total_df = total_series.reset_index()
    total_df.columns = ['Date', 'TotalVisitors']
    total_df['Date'] = pd.to_datetime(total_df['Date'], format='%Y %b')
    total_df = total_df.sort_values('Date')
    # Add cyclical month encoding
    total_df['month'] = total_df['Date'].dt.month
    total_df['month_sin'] = np.sin(2 * np.pi * total_df['month'] / 12)
    total_df['month_cos'] = np.cos(2 * np.pi * total_df['month'] / 12)
    return total_df


# Split and scale data
def split_and_scale_data(total_df):
    train = total_df[(total_df['Date'] >= '1980-01-01') & (total_df['Date'] <= '2012-12-31')]
    test = total_df[(total_df['Date'] >= '2013-01-01') & (total_df['Date'] <= '2019-12-31')]
    feature_cols = ['TotalVisitors', 'month_sin', 'month_cos']
    train_vals = train[feature_cols].values
    test_vals = test[feature_cols].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled = scaler.transform(test_vals)
    return train, test, train_scaled, test_scaled, scaler


# Create sequences for [TotalVisitors, month_sin, month_cos]
def create_sequences(data, lookback=12):
    xs, ys = [], []
    for i in range(len(data) - lookback):
        x = data[i: i + lookback]
        y = data[i + lookback, 0]  # first column = 'TotalVisitors'
        xs.append(x)
        ys.append(y)
    X = np.array(xs)
    Y = np.array(ys).reshape(-1, 1)
    return X, Y


# LSTM model (handles 3 input features)
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def main():
    # Load data
    filepath = '../Datasets/international_visitor_arrivals_by_country.csv'
    total_df = load_and_preprocess_data(filepath)
    # Split and scale
    train_df, test_df, train_scaled, test_scaled, scaler = split_and_scale_data(total_df)
    # Create sequences (using 24-month lookback, for example)
    lookback = 24
    X_train_np, y_train_np = create_sequences(train_scaled, lookback)
    X_test_np, y_test_np = create_sequences(test_scaled, lookback)
    # Convert to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    # Define and train LSTM
    model = LSTMModel(input_size=3, hidden_size=70, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)

    # Inverse transform predictions
    test_pred_only = test_pred.numpy().reshape(-1, 1)
    dummy_pred = np.zeros((len(test_pred_only), 3))
    dummy_pred[:, 0] = test_pred_only[:, 0]
    test_pred_inverse = scaler.inverse_transform(dummy_pred)[:, 0:1]
    # Also invert y_test
    y_test_only = y_test.numpy().reshape(-1, 1)
    dummy_y = np.zeros((len(y_test_only), 3))
    dummy_y[:, 0] = y_test_only[:, 0]
    y_test_inverse = scaler.inverse_transform(dummy_y)[:, 0:1]

    # Plot
    test_dates = test_df['Date'].iloc[lookback:]
    plt.figure(figsize=(20, 12))
    plt.plot(test_dates, y_test_inverse, label='Actual')
    plt.plot(test_dates, test_pred_inverse, label='Predicted')
    plt.title('Visitors Prediction (with Month Encoding)')
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
