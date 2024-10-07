import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import traceback
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch historical stock data
def fetch_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

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

def predict(ticker, forecast_days=30):
    try:
        data = fetch_data(ticker)
        if data.empty:
            logger.warning(f"No historical data found for ticker {ticker}")
            return None

        news_data = fetch_news_sentiment(ticker)
        
        if not news_data:
            logger.warning(f"No news data found for ticker {ticker}")
            sentiment = 0
        else:
            sentiment = np.mean([item['sentiment'] for item in news_data])
        
        # Convert data to numpy array
        data_array = np.array(data)
        
        # Train-Test split (90% train, 10% test)
        train_size = int(len(data_array) * 0.9)
        train, test = data_array[:train_size], data_array[train_size:]

        # Fit ARIMA model on training data
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()

        # Make predictions for the test set
        predictions = model_fit.forecast(steps=len(test))

        # Calculate accuracy metrics
        mse = np.mean((test - predictions)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test - predictions))
        mape = np.mean(np.abs((test - predictions) / test)) * 100

        # Print accuracy metrics
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse}")
        logger.info(f"Mean Absolute Error (MAE): {mae}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {mape}%")

        # Make predictions for the future (forecast)
        forecast = model_fit.forecast(steps=forecast_days)
        
        # Ensure forecast is a numpy array
        forecast = np.array(forecast)

        # Add volatility and sentiment adjustment
        volatility = np.std(np.diff(data_array) / data_array[:-1])
        noise = np.random.normal(0, volatility, forecast_days)
        sentiment_adjustment = sentiment * 0.01  # Adjust the impact of sentiment

        logger.info(f"Forecast shape: {forecast.shape}")
        logger.info(f"Noise shape: {noise.shape}")
        logger.info(f"Sentiment adjustment: {sentiment_adjustment}")

        # Perform element-wise multiplication
        forecast = forecast * (1 + noise + sentiment_adjustment)

        historical_dates = data.index[-365:].strftime('%Y-%m-%d').tolist()
        historical_data = data_array[-365:].tolist()
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_days).strftime('%Y-%m-%d').tolist()
        forecast_data = forecast.tolist()

        return {
            'historical_dates': historical_dates,
            'historical_data': historical_data,
            'forecast_dates': forecast_dates,
            'forecast_data': forecast_data,
            'sentiment': float(sentiment),
            'news_data': news_data[:10]  # Return only the 10 most recent news items
        }

    except Exception as e:
        logger.error(f"Error in predict function for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        return None