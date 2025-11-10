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

# user input
user_input = st.text_input('Enter Stock Ticker', 'AAPL')  # default to AAPL

# fetch data safely
try:
    df = yf.download(user_input, start=start, end=end, threads=False)
except Exception as e:
    st.error(f"Failed to download ticker '{user_input}': {e}")
    st.stop()

if df.empty:
    st.warning(f"No data found for ticker '{user_input}'")
    st.stop()

# display data
st.subheader('Dated from 1st Jan, 2009 to 1st Jan, 2023')
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
plt.plot(ma100, 'g', label="Moving Average 100")
plt.legend()
st.pyplot(fig2)

# third plot
st.subheader('Closing Price Vs Time Chart with 100 and 200 days Moving Average')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(ma200, 'b', label="Moving Average 200")
plt.plot(ma100, 'g', label="Moving Average 100")
plt.legend()
st.pyplot(fig3)

# split data
train_df = df['Close'][:int(len(df)*0.85)].to_frame()
test_df = df['Close'][int(len(df)*0.85):].to_frame()

scaler = MinMaxScaler(feature_range=(0, 1))
train_df_arr = scaler.fit_transform(train_df)

# load trained model safely
try:
    model = load_model('keras_model.h5', compile=False)
except Exception as e:
    st.error(f"Failed to load Keras model: {e}")
    st.stop()

# prepare test data
past_100_days = train_df.tail(100)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# predictions
y_pred = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# final plot
st.subheader('Predicted Vs Original')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")
plt.legend()
st.pyplot(fig4)
