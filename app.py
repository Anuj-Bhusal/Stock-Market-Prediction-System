import numpy as np
from flask import Flask, render_template, request, jsonify
import os
from keras.models import load_model
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load valid tickers from file
tickers_path = os.path.join(BASE_DIR, 'stock_data', 'valid_tickers.txt')
if not os.path.exists(tickers_path):
    raise FileNotFoundError(f"No such file or directory: '{tickers_path}'")

with open(tickers_path, 'r') as f:
    trained_tickers = [line.strip() for line in f]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/predict')
def predict():
    return render_template('predict.html', tickers=trained_tickers)

# Define the calculate_uptrend_downtrend_accuracy function
def calculate_uptrend_downtrend_accuracy(comparison):
    correct_predictions = 0
    total_predictions = 0  # Track the total number of valid comparisons
    
    for i in range(1, len(comparison)):
        if comparison[i]["actual_price"] is None or comparison[i-1]["actual_price"] is None:
            continue  # Skip comparisons where actual_price is None
        
        actual_change = comparison[i]["actual_price"] - comparison[i-1]["actual_price"]
        predicted_change = comparison[i]["predicted_price"] - comparison[i-1]["predicted_price"]
        
        if (actual_change > 0 and predicted_change > 0) or (actual_change < 0 and predicted_change < 0):
            correct_predictions += 1
        
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return f"{accuracy:.2f}%"

@app.route('/predict_stock', methods=['GET'])
def predict_stock():
    ticker = request.args.get('ticker')

    if not ticker:
        return jsonify({"error": "Ticker parameter is required."}), 400

    if ticker not in trained_tickers:
        return jsonify({"error": f"Ticker '{ticker}' is not a valid trained ticker."}), 400

    model_path = os.path.join(BASE_DIR, "models", f"{ticker}.keras")
    scaler_path = os.path.join(BASE_DIR, "scalers", f"{ticker}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return jsonify({"error": "Model or scaler for the specified ticker does not exist."}), 400

    try:
        model = load_model(model_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to load scaler: {e}"}), 500

    try:
        # Load the latest stock data
        data_path = os.path.join(BASE_DIR, "stock_data", f"{ticker}_stock_prices.csv")
        if not os.path.exists(data_path):
            return jsonify({"error": "No data available for the specified ticker."}), 400

        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
        df = df.sort_values(by='Date')

        # Handle duplicate dates by aggregating data (taking mean)
        df = df.groupby('Date').mean().reset_index()

        # Get all the available data
        data = df['LTP'].values.reshape(-1, 1)
        actual_price_now = float(df['LTP'].values[-1])

        # Predict prices for the next three days
        predictions = []
        for i in range(3):
            scaled_data = scaler.transform(data[-60:])  # Use only the last 60 days for prediction
            X = scaled_data.reshape(1, 60, 1)
            predicted_price = model.predict(X)
            predicted_price = scaler.inverse_transform(predicted_price).flatten()[0]
            predictions.append(float(predicted_price))  # Convert to float

            # Update data with the next day's actual LTP value (shift window)
            data = np.append(data, predicted_price).reshape(-1, 1)

        # Calculate accuracy (MAPE)
        past_actual_prices = df['LTP'].values[-4:-1]  # Exclude today's data
        mape = np.mean(np.abs((past_actual_prices - predictions[:3]) / past_actual_prices)) * 100

        # Compare past three days (excluding today's date)
        comparison = []
        past_dates = df['Date'].values[-4:-1]  # Exclude today's date
        for i in range(3):
            date = past_dates[i].astype('M8[ms]').astype(datetime)
            comparison.append({
                "date": date.strftime("%Y-%m-%d"),
                "actual_price": float(past_actual_prices[i]),  # Convert to float
                "predicted_price": predictions[i]  # Already converted to float
            })

        # Add today's and tomorrow's data
        today = pd.Timestamp('today').strftime("%Y-%m-%d")
        tomorrow = (pd.Timestamp('today') + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        comparison.append({
            "date": today,
            "actual_price": actual_price_now,  # Convert to float if needed
            "predicted_price": predictions[0]  # Already converted to float
        })
        comparison.append({
            "date": tomorrow,
            "actual_price": None,
            "predicted_price": predictions[1]  # Already converted to float
        })

        # Generate plot
        plt.figure()
        plt.plot(df['Date'].values, df['LTP'].values, label="Actual Prices")
        future_dates = pd.date_range(start=df['Date'].values[-1], periods=4, freq='B')[1:]
        plt.plot(future_dates, predictions[:3], 'ro', label="Predicted Prices")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"Stock Prices for {ticker}")
        plt.xticks(rotation=45)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        response = {
            "plot_url": f"data:image/png;base64,{plot_url}",
            "actual_price_now": actual_price_now,
            "next_three_days_predictions": predictions,
            "comparison": comparison,
            "accuracy": f"{100 - mape:.2f}%",  # Model accuracy
            "uptrend_downtrend_accuracy": calculate_uptrend_downtrend_accuracy(comparison)  # Uptrend/Downtrend accuracy
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
