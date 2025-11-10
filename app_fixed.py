# app_fixed.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor")
st.write("Enter a stock ticker and get its predicted closing prices.")

# Input ticker
user_input = st.text_input("Enter Stock Ticker", value="AAPL").upper()

# Date range
start = "2009-01-01"
end = "2023-01-01"

if user_input.strip() == "":
    st.warning("Please enter at least one valid stock ticker!")
else:
    # Load trained model safely
    try:
        model = load_model("keras_model.h5", compile=False)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Fetch data with yfinance
    try:
        df = yf.download(user_input, start=start, end=end, threads=False)
        if df.empty:
            st.error(f"No data found for ticker '{user_input}'")
            st.stop()
    except Exception as e:
        st.error(f"Failed to download ticker '{user_input}': {e}")
        st.stop()

    # Show basic data
    st.subheader(f"Data for {user_input} from {start} to {end}")
    st.write(df.describe())
    st.line_chart(df["Close"])

    # Preprocess data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    # Prepare input for model (last 60 days)
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y_true = create_dataset(scaled_close)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # reshape for LSTM

    # Make predictions
    y_pred = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1))

    # Display predictions
    st.subheader("Predicted vs Actual Closing Prices")
    pred_df = pd.DataFrame({
        "Actual": y_true.flatten(),
        "Predicted": y_pred.flatten()
    })
    st.line_chart(pred_df)
