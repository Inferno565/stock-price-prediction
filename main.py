import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import logging

logging.basicConfig(level=logging.INFO)

# Fetch historical stock data
def fetch_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2023-01-01")
    return data['Close'].values

# Preprocess the data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Create the dataset for training
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Build the model
def build_model(input_shape):
    model = keras.Sequential()
    model.add(layers.LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict function
def predict(ticker):
    logging.info(f"Starting prediction for {ticker}")
    data = fetch_data(ticker)
    logging.info(f"Fetched data shape: {data.shape}")
    
    scaled_data, scaler = preprocess_data(data)
    logging.info(f"Scaled data shape: {scaled_data.shape}")
    
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    logging.info(f"X shape: {X.shape}, y shape: {y.shape}")

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    try:
        model = keras.models.load_model('model.h5')
        model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("Loaded existing model")
    except:
        model = build_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=1)
        model.save('model.h5')
        logging.info("Built and trained new model")

    predictions = model.predict(X_test)
    logging.info(f"Raw predictions shape: {predictions.shape}")
    predictions = scaler.inverse_transform(predictions)
    logging.info(f"Inverse transformed predictions shape: {predictions.shape}")

    original_data = data[-len(predictions):]
    logging.info(f"Original data shape: {original_data.shape}")

    original_list = original_data.flatten().tolist()
    predictions_list = predictions.flatten().tolist()
    
    logging.info(f"Original data (first 5): {original_list[:5]}")
    logging.info(f"Predictions (first 5): {predictions_list[:5]}")
    
    return original_list, predictions_list

# # Example usage
# if __name__ == "__main__":
#     ticker = "AAPL"  # Example: Apple Inc.
#     predictions = predict(ticker)
#     print(predictions)