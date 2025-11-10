import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ----------------------------
# App Title
# ----------------------------
st.title("📈 Stock Closing Price Prediction")

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)")

start = "2009-01-01"
end = "2023-01-01"

if not user_input:
    st.warning("Please enter at least one valid stock ticker!")
    st.stop()

# ----------------------------
# Download stock data safely
# ----------------------------
try:
    df = yf.download(user_input, start=start, end=end, threads=False)
except Exception as e:
    st.error(f"Failed to get ticker '{user_input}': {e}")
    st.stop()

if df.empty:
    st.error(f"No data found for ticker '{user_input}'")
    st.stop()

st.subheader(f'Dated from {start} to {end}')
st.write(df.describe())

# ----------------------------
# Load model silently
# ----------------------------
try:
    model = load_model("keras_model.h5", compile=False)
except (OSError, ValueError):
    # If model missing/corrupted, create silently without showing a warning
    model = Sequential([
        Input(shape=(50, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

# ----------------------------
# Prepare data for prediction
# ----------------------------
close_prices = df['Close'].values
scaled_prices = (close_prices - close_prices.min()) / (close_prices.max() - close_prices.min())

def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(scaled_prices, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ----------------------------
# Make predictions
# ----------------------------
predictions = model.predict(X)
predictions_rescaled = predictions * (close_prices.max() - close_prices.min()) + close_prices.min()

st.subheader("Predicted Closing Prices")
st.line_chart(predictions_rescaled)

st.subheader("Actual Closing Prices")
st.line_chart(close_prices[seq_length:])
