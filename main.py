import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def fetch_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def fetch_news_sentiment(ticker):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table:
            logger.warning(f"No news table found for ticker {ticker}")
            return []

        news_data = []
        for row in news_table.findAll('tr'):
            title = row.a.text
            date_data = row.td.text.split(' ')
            
            if len(date_data) == 1:
                time = date_data[0]
                date = datetime.now().strftime('%Y-%m-%d')
            else:
                date = date_data[0]
                time = date_data[1]
            
            sentiment = TextBlob(title).sentiment.polarity
            news_data.append({'date': date, 'time': time, 'title': title, 'sentiment': sentiment})

        return news_data
    except Exception as e:
        logger.error(f"Error in fetch_news_sentiment for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def preprocess_data(data):
    # Add technical indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    
    # Remove NaN values
    data.dropna(inplace=True)
    
    return data

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # 0 index for 'Close' price
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict(ticker, forecast_days=30):
    try:
        # Load the pre-trained model and scaler
        model = load_model(f'{ticker}_lstm_model.h5')
        with open(f'{ticker}_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Fetch recent data for prediction
        data = fetch_data(ticker)
        data = preprocess_data(data)

        # Prepare data for LSTM
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        dataset = data[features].values
        scaled_data = scaler.transform(dataset)

        # Create the last sequence for prediction
        seq_length = 60
        last_sequence = scaled_data[-seq_length:]

        # Forecast future prices
        forecast = []
        for _ in range(forecast_days):
            next_pred = model.predict(last_sequence.reshape(1, seq_length, len(features)))
            forecast.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = next_pred

        # Inverse transform forecast
        forecast = scaler.inverse_transform(np.hstack([np.array(forecast).reshape(-1, 1), np.zeros((len(forecast), len(features)-1))]))[:, 0]

        # Fetch news sentiment
        news_data = fetch_news_sentiment(ticker)
        sentiment = np.mean([item['sentiment'] for item in news_data]) if news_data else 0

        # Add volatility and sentiment adjustment
        volatility = np.std(data['Close'].pct_change())
        noise = np.random.normal(0, volatility, forecast_days)
        sentiment_adjustment = sentiment * 0.01

        forecast = forecast * (1 + noise + sentiment_adjustment)

        historical_dates = data.index[-365:].strftime('%Y-%m-%d').tolist()
        historical_data = data['Close'].values[-365:].tolist()
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_days).strftime('%Y-%m-%d').tolist()

        return {
            'historical_dates': historical_dates,
            'historical_data': historical_data,
            'forecast_dates': forecast_dates,
            'forecast_data': forecast.tolist(),
            'sentiment': float(sentiment),
            'news_data': news_data[:10],
            'last_close_price': float(data['Close'].iloc[-1]),
            'next_day_prediction': float(forecast[0])
        }

    except Exception as e:
        logger.error(f"Error in predict function for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        return None
