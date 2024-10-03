import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

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
    data = fetch_data(ticker)
    scaled_data, scaler = preprocess_data(data)
    
    # Create training and testing datasets
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Load or build the model
    try:
        model = keras.models.load_model('model.h5')  # Load existing model
    except:
        model = build_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, batch_size=1, epochs=10)
        model.save('model.h5')  # Save the model

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

# Example usage
if __name__ == "__main__":
    ticker = "AAPL"  # Example: Apple Inc.
    predictions = predict(ticker)
    print(predictions)