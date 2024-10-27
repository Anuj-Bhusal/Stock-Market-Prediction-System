import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

def load_data(ticker):
    file_path = f"stock_data/{ticker}_stock_prices.csv"
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
    df = df.sort_values(by='Date')
    return df

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def train_model(ticker):
    df = load_data(ticker)
    data = df['LTP'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, y, validation_split=0.1, epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('scalers'):
        os.makedirs('scalers')

    model.save(f"models/{ticker}.keras")
    with open(f"scalers/{ticker}.pkl", 'wb') as f:
        pickle.dump(scaler, f)

def main():
    with open('stock_data/valid_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f]

    for ticker in tickers:
        train_model(ticker)
        print(f"Model trained and saved for ticker: {ticker}")

if __name__ == "__main__":
    main()