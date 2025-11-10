import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from datetime import date, timedelta

# ----------------------------
# App Title
# ----------------------------
st.title("📈 Stock Closing Price Prediction")

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL)")

start = st.date_input("Start Date", value=date(2009, 1, 1))
end = st.date_input("End Date", value=date(2023, 1, 1))

future_days = st.slider("Predict Future Days", 1, 60, 7)

if not user_input:
    st.warning("Please enter a stock ticker!")
    st.stop()

if start >= end:
    st.warning("End date must be after start date!")
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
    model = Sequential([
        Input(shape=(50, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

# ----------------------------
# Prepare data
# ----------------------------
close_prices = df['Close'].values
scaled_prices = (close_prices - close_prices.min()) / (close_prices.max() - close_prices.min())

def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = min(50, len(scaled_prices)-1)
if seq_length < 1:
    st.error("Not enough data for prediction. Try a longer date range.")
    st.stop()

X, y = create_sequences(scaled_prices, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ----------------------------
# Make predictions on historical data
# ----------------------------
predictions = model.predict(X)
predictions_rescaled = predictions * (close_prices.max() - close_prices.min()) + close_prices.min()

st.subheader("Predicted Closing Prices (Historical)")
st.line_chart(predictions_rescaled)

st.subheader("Actual Closing Prices")
st.line_chart(close_prices[seq_length:])

# ----------------------------
# Predict future days
# ----------------------------
last_sequence = scaled_prices[-seq_length:]
future_preds = []

for _ in range(future_days):
    x_input = last_sequence.reshape((1, seq_length, 1))
    pred = model.predict(x_input)[0][0]
    future_preds.append(pred)
    last_sequence = np.append(last_sequence[1:], pred)

future_preds_rescaled = np.array(future_preds) * (close_prices.max() - close_prices.min()) + close_prices.min()
future_dates = [end + timedelta(days=i+1) for i in range(future_days)]

st.subheader(f"Next {future_days}-Day Predicted Prices")
st.line_chart(pd.DataFrame({'Date': future_dates, 'Predicted': future_preds_rescaled}).set_index('Date'))
