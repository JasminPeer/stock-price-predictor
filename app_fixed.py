# app_fixed.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Closing Price Prediction")

# --- User input ---
user_input = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL):").upper()

start = "2009-01-01"
end = "2023-01-01"

if not user_input:
    st.warning("Please enter at least one valid stock ticker!")
    st.stop()

# --- Download stock data ---
try:
    df = yf.download(user_input, start=start, end=end, threads=False, auto_adjust=True)
except Exception as e:
    st.error(f"Error fetching data for {user_input}: {e}")
    st.stop()

if df.empty:
    st.error(f"No data found for ticker '{user_input}'")
    st.stop()

st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

# --- Load trained model ---
try:
    model = load_model('keras_model.h5')  # Make sure model was saved without 'time_major'
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Prepare data for prediction ---
scaler = MinMaxScaler(feature_range=(0,1))
close_prices = df['Close'].values.reshape(-1,1)
scaled_data = scaler.fit_transform(close_prices)

# Predict last 60 days to forecast next day
look_back = 60
X_test = []
for i in range(look_back, len(scaled_data)):
    X_test.append(scaled_data[i-look_back:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

st.subheader(f"Predicted Closing Prices (last {len(predicted_prices)} days)")
st.line_chart(predicted_prices)

st.subheader("Actual Closing Prices")
st.line_chart(close_prices[look_back:])
