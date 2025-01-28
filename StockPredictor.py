import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
import matplotlib.pyplot as plt

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)  # Only scale training data here
    x_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        x_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    return np.array(x_train), np.array(y_train), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

ticker = 'MSFT' 
data = fetch_data(ticker, '2015-01-01', '2024-10-28')
x_train, y_train, scaler = preprocess_data(data)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train_cnn = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

model = build_model((x_train_cnn.shape[1], x_train_cnn.shape[2]))
model.fit(x_train_cnn, y_train, epochs=50, batch_size=32)

data_test = fetch_data(ticker, '2023-01-01', '2024-10-28')
scaled_test_data = scaler.transform(data_test)
x_test = [scaled_test_data[i-60:i, 0] for i in range(60, len(scaled_test_data))]
x_test = np.array(x_test).reshape(-1, 60, 1) 

x_test_cnn = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test_cnn)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.figure(figsize=(10, 5))
plt.plot(data_test[60:], color='blue', label='Actual Price')
plt.plot(predicted_prices, color='red', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f"{ticker} Stock Prediction")
plt.legend()
plt.show()

# Predict Future Prices
# Fetch fresh data and separate scaler
def predict_future_prices(model, data, days=30):
    # Assuming `scaler` is defined from the actual data range
    last_data = data[-120:]
    scaled_data = scaler.transform(last_data)
    
    predictions = []
    for _ in range(days):
        X_test = scaled_data.reshape(1, scaled_data.shape[0], 1)
        predicted_price = model.predict(X_test)
        predictions.append(predicted_price[0, 0])

        # Update last_data for the next iteration with new prediction
        scaled_data = np.append(scaled_data, predicted_price[0, 0]).reshape(-1, 1)[-120:]

    # Transform predictions back to original scale
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

data_test = fetch_data(ticker, '2023-01-01', '2024-10-28')
future_prices = predict_future_prices(model, data_test, days=30)

# Visualization
plt.figure(figsize=(12, 6))
# Plot previous 60 days
x_actual = np.arange(len(data) - 120, len(data))
plt.plot(x_actual, data[-120:], color='blue', label='Last 60 Days Actual Price')

# Plot future predictions with a continuous x-axis
x_future = np.arange(len(data) - 1, len(data) + 30)
plt.plot(x_future, np.concatenate((data[-1:], future_prices)), color='orange', label='Predicted Future Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title(f'Predicted Prices for the Next 30 Days ({ticker})')
plt.legend()
plt.show()
