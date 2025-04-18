from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------------- configurable constants ---------------------------
DATA_PATH          = '../Datasets/international_visitor_arrivals_by_country.csv'
FORECAST_HORIZON   = 3
LOOKBACK           = 24
ALPHA              = 1.25
LR                 = 0.005
EPOCHS             = 300
HIDDEN_SIZE        = 64
TRIM_Q_LOW, TRIM_Q_HIGH = 0.05, 0.01   # for trimmed mean % diff
# -----------------------------------------------------------------------


def load_and_preprocess_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[df.columns[::-1]]

    label = "Total International Visitor Arrivals By Inbound Tourism Markets"
    if label not in df.index:
        raise KeyError(label)

    s = df.loc[label]
    out = (
        s.reset_index()
         .rename(columns={'index': 'Date', label: 'TotalVisitors'})
    )
    out['Date'] = pd.to_datetime(out['Date'], format='%Y %b')
    out.sort_values('Date', inplace=True)
    out['TotalVisitors'] = pd.to_numeric(out['TotalVisitors'], errors='coerce')
    return out.dropna()


def split_data(df: pd.DataFrame):
    train = df[(df.Date >= '2004-01-01') & (df.Date <= '2013-12-31')]
    test  = df[(df.Date >= '2014-01-01') & (df.Date <= '2019-12-31')]
    return train, test


def create_sequences(arr: np.ndarray, lookback: int):
    xs, ys = [], []
    for i in range(len(arr) - lookback):
        xs.append(arr[i:i + lookback])
        ys.append(arr[i + lookback])
    return np.array(xs), np.array(ys)


class TrendLSTM(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE)
        c0 = torch.zeros(1, x.size(0), HIDDEN_SIZE)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def fit_seasonal_model(series: pd.Series):
    model = SARIMAX(series,
                    order=(0, 0, 0),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=True,
                    enforce_invertibility=True)
    return model.fit(disp=False)


def main():
    # 1. data ----------------------------------------------------------------
    df_total = load_and_preprocess_data(DATA_PATH)
    train_df, test_df = split_data(df_total)

    y_train = train_df.set_index('Date')['TotalVisitors']
    y_test  = test_df.set_index('Date')['TotalVisitors']
    y_train.index = y_train.index.to_period('M').to_timestamp(); y_train.index.freq = 'MS'
    y_test.index  = y_test.index.to_period('M').to_timestamp();  y_test.index.freq  = 'MS'

    # 2. initial seasonal fit & trend‑LSTM training --------------------------
    seasonal_fit = fit_seasonal_model(y_train)
    seasonal_pred_train = seasonal_fit.predict(start=y_train.index[0],
                                               end=y_train.index[-1])
    deseason_train = y_train - seasonal_pred_train
    scaler = MinMaxScaler()
    deseason_train_scaled = scaler.fit_transform(
        deseason_train.values.reshape(-1, 1))

    X_tr, y_tr = create_sequences(deseason_train_scaled, LOOKBACK)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    lstm_model = TrendLSTM()
    opt = torch.optim.Adam(lstm_model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss(delta=0.001)

    for _ in range(EPOCHS):
        lstm_model.train()
        opt.zero_grad()
        loss = loss_fn(lstm_model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

    # 3. walk‑forward --------------------------------------------------------
    history = y_train.copy()
    pred_cols = [f"Forecast_t+{h}" for h in range(1, FORECAST_HORIZON + 1)]
    pred_dict = {col: [] for col in pred_cols}
    actuals, seasonal_1, trend_1, hybrid_1 = [], [], [], []
    seasonal_pred_test = seasonal_fit.predict(start=y_test.index[0],
                                              end=y_test.index[-1])
    for idx, actual_val in y_test.items():
        #   seasonal
        seasonal_forecast = np.array([seasonal_pred_test.loc[idx]])

        #   latest deseasonalised history
        seasonal_hist = seasonal_fit.predict(start=history.index[0],
                                             end=history.index[-1])
        deseason_hist = history - seasonal_hist
        deseason_hist_scaled = scaler.transform(
            deseason_hist.values.reshape(-1, 1)).flatten()

        #   recursive LSTM
        seq = list(deseason_hist_scaled[-LOOKBACK:])
        trend_scaled = []
        lstm_model.eval()
        with torch.no_grad():
            for _ in range(FORECAST_HORIZON):
                x_inp = torch.tensor(
                    np.array(seq[-LOOKBACK:]).reshape(1, LOOKBACK, 1),
                    dtype=torch.float32)
                pred_s = lstm_model(x_inp).item()
                trend_scaled.append(pred_s)
                seq.append(pred_s)

        trend_preds = scaler.inverse_transform(
            np.array(trend_scaled).reshape(-1, 1)).flatten()

        hybrid = seasonal_forecast + ALPHA * trend_preds

        # save horizon‑specific predictions
        for h, col in enumerate(pred_cols, start=1):
            pred_dict[col].append(hybrid[h - 1])

        # save 1‑step components for “original” plots
        seasonal_1.append(seasonal_forecast[0])
        trend_1.append(trend_preds[0])
        hybrid_1.append(hybrid[0])

        actuals.append(actual_val)
        history.loc[idx] = actual_val  # roll forward


    pred_df = pd.DataFrame(
        index=y_test.index,
        columns=["Actual", "Seasonal", "Trend", "Hybrid"] + pred_cols,
        dtype=float
    )
    pred_df["Actual"]   = actuals
    pred_df["Seasonal"] = seasonal_1
    pred_df["Trend"]    = trend_1
    pred_df["Hybrid"]   = hybrid_1
    for col in pred_cols:
        pred_df[col] = pred_dict[col]


    os.makedirs("plots_multistep", exist_ok=True)
    print("\n=== Metrics by horizon ===")
    for h in range(1, FORECAST_HORIZON + 1):
        col = f"Forecast_t+{h}"
        valid_df = pred_df.iloc[:-(h - 1)] if h > 1 else pred_df

        actual = valid_df["Actual"].values
        forecast = valid_df[col].values


        abs_err = np.abs(forecast - actual)
        cut_err = np.quantile(abs_err, 1 - TRIM_Q_HIGH)
        trimmed_mae = abs_err[abs_err <= cut_err].mean()
        mae = abs_err.mean()


        pct_diff = (forecast - actual) / actual  
        abs_pct = np.abs(pct_diff)
        cut_pct = np.quantile(abs_pct, 1 - TRIM_Q_HIGH) 
        mask_pct = abs_pct <= cut_pct 
        raw_mean = pct_diff.mean() * 100  
        trimmed_mean = pct_diff[mask_pct].mean() * 100  

 
        print(f"{h:2d}‑step | MAE: {mae:,.0f} | "
              f"Trim‑MAE: {trimmed_mae:,.0f} | "
              f"Mean % diff: {raw_mean:.3f}% | "
              f"Trimmed % diff: {trimmed_mean:.3f}%")

        # Actual vs. Forecast_t+h
        dates = valid_df.index
        plt.figure(figsize=(12, 5))
        plt.plot(dates, actual, label="Actual")
        plt.plot(dates, forecast, label=col, linestyle='--')
        plt.title(f"Actual vs. {h}-step‑ahead Forecast")
        plt.xlabel("Date"); plt.ylabel("Visitors"); plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots_multistep/actual_vs_forecast_h{h}.png")
        plt.close()

        # Proportional difference curve
        plt.figure(figsize=(12, 4))
        plt.plot(dates, pct_diff * 100, marker='o')
        plt.axhline(0, linewidth=1)
        plt.title(f"Proportional Difference (Forecast‑Actual)/Actual ·100  —  {h}-step")
        plt.xlabel("Date"); plt.ylabel("% difference")
        plt.tight_layout()
        plt.savefig(f"plots_multistep/prop_diff_h{h}.png")
        plt.close()


    # Seasonal‑only vs Actual
    plt.figure(figsize=(20, 6))
    plt.plot(pred_df.index, pred_df["Actual"], label="Actual (Test)")
    plt.plot(pred_df.index, pred_df["Seasonal"], "--", label="Seasonal‑only Forecast")
    plt.title("Seasonal‑Only SARIMA vs. Actual (Test)")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots_multistep/seasonal_vs_actual.png"); plt.close()

    # Hybrid vs Actual
    plt.figure(figsize=(20, 6))
    plt.plot(pred_df.index, pred_df["Actual"], label="Actual (Test)")
    plt.plot(pred_df.index, pred_df["Hybrid"], "--", label="Hybrid Forecast")
    plt.title("Hybrid Forecast vs. Actual")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots_multistep/hybrid_vs_actual.png"); plt.close()

    # 4‑line decomposition
    plt.figure(figsize=(20, 8))
    plt.plot(pred_df.index, pred_df["Actual"],   label="Actual",  marker='o', color='blue')
    plt.plot(pred_df.index, pred_df["Seasonal"], label="Seasonal (SARIMA)", linestyle='--', color='red')
    plt.plot(pred_df.index, pred_df["Trend"],    label="Trend (LSTM)",      linestyle=':', color='green')
    plt.plot(pred_df.index, pred_df["Hybrid"],   label="Hybrid",            linewidth=2,   color='orange', marker='o')
    plt.title("All Components: Actual, Seasonal, Trend, Hybrid")
    plt.xlabel("Date"); plt.ylabel("Total Visitors")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots_multistep/decomposition_4lines.png"); plt.close()

    # Proportional diff (Actual − Hybrid)/Actual
    prop_diff = (pred_df["Actual"] - pred_df["Hybrid"]) / pred_df["Actual"]
    plt.figure(figsize=(20, 6))
    plt.plot(pred_df.index, prop_diff * 100, marker='o', color='purple')
    plt.axhline(0, linewidth=1)
    plt.title("Proportional Difference (Actual − Hybrid) Over Time")
    plt.xlabel("Date"); plt.ylabel("% difference")
    plt.tight_layout()
    plt.savefig("plots_multistep/prop_diff_hybrid.png"); plt.close()

    print("\nPlots saved in the 'plots_multistep' directory.")
    pred_df.to_csv(f"hybrid_multistep_H{FORECAST_HORIZON}.csv")
    print(f"Predictions → hybrid_multistep_H{FORECAST_HORIZON}.csv")


if __name__ == "__main__":
    main()