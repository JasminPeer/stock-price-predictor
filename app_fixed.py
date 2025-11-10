# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# define start and end dates
start = "2009-01-01"
end = "2023-01-01"

# Streamlit title
st.title('📈 Stock Closing Price Prediction')

# input multiple tickers
user_input = st.text_input('Enter Stock Ticker(s) separated by commas', 'GOOGL,AAPL,MSFT')
tickers = [t.strip().upper() for t in user_input.split(",") if t.strip() != ""]

if not tickers:
    st.warning("Please enter at least one valid stock ticker!")
else:
    # load trained model
    model = load_model('keras_model.h5')

    for ticker_symbol in tickers:
        st.header(f"Ticker: {ticker_symbol}")
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)

            if df.empty:
                st.error(f"No data found for ticker '{ticker_symbol}'. Skipping.")
                continue

            # display data
            st.subheader('Data Summary')
            st.write(df.describe())

            # first plot
            st.subheader('Closing Price Vs Time Chart')
            fig1 = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'])
            st.pyplot(fig1)

            # moving averages
            ma100 = df['Close'].rolling(100).mean()
            ma200 = df['Close'].rolling(200).mean()

            # second plot
            st.subheader('Closing Price Vs Time Chart with 100 days Moving Average')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], 'r', label="Per Day Closing")
            plt.plot(ma100, 'g', label="MA 100")
            plt.legend()
            st.pyplot(fig2)

            # third plot
            st.subheader('Closing Price Vs Time Chart with 100 and 200 days Moving Average')
            fig3 = plt.figure(figsize=(12, 6))
            plt.plot(ma200, 'b', label="MA 200")
            plt.plot(ma100, 'g', label="MA 100")
            plt.legend()
            st.pyplot(fig3)

            # data training
            train_df = pd.DataFrame(df['Close'][0:int(len(df)*0.85)])
            test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):int(len(df))])

            scaler = MinMaxScaler(feature_range=(0, 1))
            train_df_arr = scaler.fit_transform(train_df)

            # prepare test data
            past_100_days = train_df.tail(100)
            final_df = pd.concat([past_100_days, test_df], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])
            x_test, y_test = np.array(x_test), np.array(y_test)

            # predictions
            y_pred = model.predict(x_test)
            scale = scaler.scale_
            scale_factor = 1 / scale[0]
            y_pred = y_pred * scale_factor
            y_test = y_test * scale_factor

            # final plot
            st.subheader('Predicted Vs Original')
            fig4 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'g', label="Original Price")
            plt.plot(y_pred, 'r', label="Predicted Price")
            plt.legend()
            st.pyplot(fig4)

        except Exception as e:
            st.error(f"Failed to fetch data for '{ticker_symbol}': {e}")
