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


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, header=0, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[df.columns[::-1]]
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


def split_data(total_df):
    train = total_df[(total_df['Date'] >= '1980-01-01') & (total_df['Date'] <= '2017-12-31')]
    test = total_df[(total_df['Date'] >= '2018-01-01') & (total_df['Date'] <= '2025-12-31')]
    return train, test


def create_sequences(data, lookback=12):
    xs, ys = [], []
    for i in range(len(data) - lookback):
        xs.append(data[i:i + lookback])
        ys.append(data[i + lookback])
    return np.array(xs), np.array(ys)


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
    filepath = '../Datasets/international_visitor_arrivals_by_country.csv'
    total_df = load_and_preprocess_data(filepath)
    train_df, test_df = split_data(total_df)

    y_train = train_df.set_index('Date')['TotalVisitors']
    y_test = test_df.set_index('Date')['TotalVisitors']
    y_train.index = pd.DatetimeIndex(y_train.index).to_period('M').to_timestamp()
    y_train.index.freq = 'MS'
    y_test.index = pd.DatetimeIndex(y_test.index).to_period('M').to_timestamp()
    y_test.index.freq = 'MS'

    print(test_df)

    seasonal_model = SARIMAX(y_train, order=(0, 0, 0), seasonal_order=(1, 1, 1, 12),
                             enforce_stationarity=True, enforce_invertibility=True)
    seasonal_fit = seasonal_model.fit(disp=False)
    seasonal_train_pred = seasonal_fit.predict(start=y_train.index[0],
                                               end=y_train.index[-1], dynamic=False)
    deseason_train = y_train - seasonal_train_pred

    scaler = MinMaxScaler(feature_range=(0, 1))
    deseason_train_scaled = scaler.fit_transform(deseason_train.values.reshape(-1, 1))

    lookback = 24
    X_train, y_train_seq = create_sequences(deseason_train_scaled, lookback)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

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

    hook_handle = lstm_model.lstm.register_forward_hook(hook_fn)
    lstm_model.eval()
    hidden_states.clear()
    with torch.no_grad():
        _ = lstm_model(X_train_tensor)
    hook_handle.remove()

    hs = np.concatenate(hidden_states, axis=0)
    hs_reshaped = hs.reshape(hs.shape[0], -1)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(hs_reshaped)

    # Simple PCA scatter plot (uncolored)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title("PCA of LSTM Hidden States (Uncolored)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    sequence_dates = y_train.index[lookback:]
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Date': sequence_dates.values
    })
    pca_df['Year'] = pca_df['Date'].dt.year
    pca_df['Month'] = pca_df['Date'].dt.month

    # Color-code by Month
    plt.figure(figsize=(20, 12))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Month', palette='viridis')
    plt.title("PCA of LSTM Hidden States by Month (Training Data)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()

    # 2) t-SNE on hidden states (alternative to PCA)
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(hs_reshaped)
    pca_df['TSNE1'] = tsne_result[:, 0]
    pca_df['TSNE2'] = tsne_result[:, 1]
    plt.figure(figsize=(20, 12))
    sns.scatterplot(data=pca_df, x='TSNE1', y='TSNE2', hue='Month', palette='coolwarm')
    plt.title("t-SNE of LSTM Hidden States by Month (Training Data)")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()

    # 3d PCA
    pca3d = PCA(n_components=3)
    pca3d_result = pca3d.fit_transform(hs_reshaped)
    pca_df_3d = pd.DataFrame({
        'PC1': pca3d_result[:, 0],
        'PC2': pca3d_result[:, 1],
        'PC3': pca3d_result[:, 2],
        'Date': sequence_dates.values
    })
    pca_df_3d['Year'] = pca_df_3d['Date'].dt.year
    pca_df_3d['Month'] = pca_df_3d['Date'].dt.month

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pca_df_3d['PC1'], pca_df_3d['PC2'], pca_df_3d['PC3'],
                    c=pca_df_3d['Month'], cmap='viridis', s=50)
    ax.set_title("3D PCA of LSTM Hidden States")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.colorbar(sc, label='Month')
    plt.show()

    # Additional separate 2D PCA chart colored by Year
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Year', palette='coolwarm')
    plt.title("2D PCA of LSTM Hidden States by Year (Training Data)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()

    # Additional separate 3D PCA chart colored by Year
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pca_df_3d['PC1'], pca_df_3d['PC2'], pca_df_3d['PC3'],
                    c=pca_df_3d['Year'], cmap='coolwarm', s=50)
    ax.set_title("3D PCA of LSTM Hidden States by Year")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.colorbar(sc, label='Year')
    plt.show()

    import shap

    X_train_flat = X_train_tensor.numpy().reshape(X_train_tensor.shape[0], -1)


    background_size = min(200, X_train_flat.shape[0])
    background_data = X_train_flat[:background_size]


    steps = len(y_test)
    seasonal_forecast_obj = seasonal_fit.get_forecast(steps=steps)
    seasonal_test_pred = seasonal_forecast_obj.predicted_mean
    deseason_test = y_test - seasonal_test_pred
    deseason_test_scaled = scaler.transform(deseason_test.values.reshape(-1, 1))
    X_test, _ = create_sequences(deseason_test_scaled, lookback)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_flat = X_test_tensor.numpy().reshape(X_test_tensor.shape[0], -1)

    def model_predict(data_np):
        data_torch = torch.tensor(data_np.reshape(-1, lookback, 1), dtype=torch.float32)
        with torch.no_grad():
            preds = lstm_model(data_torch).numpy().flatten()
        return preds

    explainer = shap.KernelExplainer(model_predict, background_data)
    test_size = min(24, X_test_flat.shape[0])
    test_subset = X_test_flat[:test_size]

    shap_values = explainer.shap_values(test_subset)

    shap.summary_plot(shap_values, test_subset, plot_type="bar", feature_names=[f"t{i}" for i in range(lookback)])
    shap.summary_plot(shap_values, test_subset, feature_names=[f"t{i}" for i in range(lookback)])
    steps = len(y_test)
    seasonal_forecast_obj = seasonal_fit.get_forecast(steps=steps)
    seasonal_test_pred = seasonal_forecast_obj.predicted_mean

    deseason_test = y_test - seasonal_test_pred
    deseason_test_scaled = scaler.transform(deseason_test.values.reshape(-1, 1))

    X_test, _ = create_sequences(deseason_test_scaled, lookback)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    lstm_model.eval()
    with torch.no_grad():
        deseason_test_pred_scaled = lstm_model(X_test_tensor).numpy()
    deseason_test_pred = scaler.inverse_transform(deseason_test_pred_scaled).flatten()

    seasonal_test_aligned = seasonal_test_pred.iloc[lookback:]
    alpha = 1.25
    final_forecast = seasonal_test_aligned.values + alpha * deseason_test_pred
    actual_test_aligned = y_test.iloc[lookback:]

    plt.figure(figsize=(20, 6))
    plt.plot(y_test.index, y_test, label='Actual (Test)')
    plt.plot(y_test.index, seasonal_test_pred, label='Seasonal-Only Forecast', linestyle='--')
    plt.title('Seasonal-Only SARIMA vs. Actual (Test)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 6))
    plt.plot(actual_test_aligned.index, actual_test_aligned, label='Actual (Aligned)')
    plt.plot(actual_test_aligned.index, final_forecast, label='Final Hybrid (Seasonal + LSTM Trend)', linestyle='--')
    plt.title('Hybrid Forecast vs. Actual')
    plt.legend()
    plt.show()

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

    plt.figure(figsize=(20, 6))
    plt.plot(breakdown_df['Date'], breakdown_df['Proportional_Difference'], marker='o', linestyle='-', color='purple')
    plt.title('Proportional Difference (Actual - Hybrid) Over Time')
    plt.xlabel('Date')
    plt.ylabel('Proportional Difference')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
