import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import logging
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

# Fetch historical stock data
def fetch_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Get sentiment from recent news
def get_sentiment(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_table = soup.find(id='news-table')
    if news_table:
        recent_news = news_table.find_all('tr')[:5]  # Get the 5 most recent news items
        sentiment_score = sum(['positive' in tr.text.lower() for tr in recent_news]) - sum(['negative' in tr.text.lower() for tr in recent_news])
        return sentiment_score / 5  # Normalize to [-1, 1]
    return 0

# Predict function
def predict(ticker, forecast_days=30):
    logging.info(f"Starting prediction for {ticker} with {forecast_days} days forecast")
    try:
        data = fetch_data(ticker)
        logging.info(f"Fetched data shape: {data.shape}")

        # Fit ARIMA model
        model = ARIMA(data, order=(5,1,0))
        model_fit = model.fit()

        # Make predictions
        forecast = model_fit.forecast(steps=forecast_days)
        
        # Add volatility
        volatility = data.pct_change().std()
        noise = np.random.normal(0, volatility, forecast_days)
        forecast = forecast * (1 + noise)

        # Adjust forecast based on sentiment
        sentiment = get_sentiment(ticker)
        sentiment_factor = 1 + (sentiment * 0.01)  # 1% adjustment per sentiment unit
        forecast = forecast * sentiment_factor

        historical_dates = data.index[-365:].strftime('%Y-%m-%d').tolist()
        historical_data = data.values[-365:].tolist()
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_days).strftime('%Y-%m-%d').tolist()
        forecast_data = forecast.tolist()

        logging.info(f"Historical data shape: {len(historical_data)}")
        logging.info(f"Forecasted prices shape: {len(forecast_data)}")

        return {
            'historical_dates': historical_dates,
            'historical_data': historical_data,
            'forecast_dates': forecast_dates,
            'forecast_data': forecast_data
        }
    except Exception as e:
        logging.error(f"Error in predict function: {str(e)}")
        return None

# # Example usage
# if __name__ == "__main__":
#     ticker = "AAPL"  # Example: Apple Inc.
#     predictions = predict(ticker)
#     print(predictions)