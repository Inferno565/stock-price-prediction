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