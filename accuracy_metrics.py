import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

def load_data(ticker):
    file_path = f"stock_data/{ticker}_stock_prices.csv"
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
    df = df.sort_values(by='Date')
    return df

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def calculate_metrics(model, scaler, data, time_step=60):
    # Prepare data for prediction
    X, y = create_dataset(data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Predict
    predictions = model.predict(X)

    # Inverse transform to get actual prices
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    y_pred = scaler.inverse_transform(predictions)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mae, mse, mape

def evaluate_model(ticker):
    model_path = f"models/{ticker}.keras"
    scaler_path = f"scalers/{ticker}.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler for ticker {ticker} not found.")
        return

    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading model or scaler for ticker {ticker}: {e}")
        return

    df = load_data(ticker)
    data = df['LTP'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = 60
    mae, mse, mape = calculate_metrics(model, scaler, scaled_data, time_step)
    print(f"Model metrics for {ticker} - MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%")

def main():
    with open('stock_data/valid_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f]

    for ticker in tickers:
        evaluate_model(ticker)

if __name__ == "__main__":
    main()