# app_fixed.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

# --- User Input ---
tickers_input = st.text_input(
    "Enter stock ticker(s) separated by commas (e.g., AAPL, MSFT, GOOGL):"
).upper()

start_date = st.date_input("Start date", value=pd.to_datetime("2009-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2023-01-01"))

if not tickers_input:
    st.warning("Please enter at least one valid stock ticker!")
    st.stop()

tickers = [ticker.strip() for ticker in tickers_input.split(",")]

# --- Load trained model ---
try:
    model = load_model("keras_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Process each ticker ---
for ticker_symbol in tickers:
    st.header(f"Ticker: {ticker_symbol}")

    # --- Download stock data ---
    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        continue

    if df.empty:
        st.error(f"No data found for ticker '{ticker_symbol}'")
        continue

    st.subheader(f"Data from {start_date} to {end_date}")
    st.write(df.describe())

    # --- Prepare data for LSTM ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    look_back = 60
    X_test = []
    for i in range(look_back, len(scaled_data)):
        X_test.append(scaled_data[i - look_back:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # --- Make predictions ---
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # --- Plot actual vs predicted ---
    st.subheader("Actual vs Predicted Prices")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'].values[look_back:], color='blue', label='Actual Price')
    plt.plot(predicted_prices, color='red', label='Predicted Price')
    plt.title(f"{ticker_symbol} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)
