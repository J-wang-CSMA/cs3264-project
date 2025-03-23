import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def deseasonalize_data(y_train, y_test, seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=True, enforce_invertibility=True):
    seasonal_model = SARIMAX(
        y_train,
        order=(0, 0, 0),
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility
    )
    seasonal_fit = seasonal_model.fit(disp=False)
    seasonal_train_pred = seasonal_fit.predict(start=y_train.index[0],
                                               end=y_train.index[-1],
                                               dynamic=False)
    deseason_train = y_train - seasonal_train_pred

    steps = len(y_test)
    seasonal_forecast_obj = seasonal_fit.get_forecast(steps=steps)
    seasonal_test_pred = seasonal_forecast_obj.predicted_mean
    deseason_test = y_test - seasonal_test_pred

    return seasonal_fit, seasonal_train_pred, deseason_train, seasonal_test_pred, deseason_test
