import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import traceback
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import yfinance as yf
from pmdarima import auto_arima

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of historical data
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

def find_best_arima_model(data):
    model = auto_arima(data, start_p=1, start_q=1, max_p=5, max_q=5, m=7,
                       start_P=0, seasonal=True, d=1, D=1, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)
    return model.order, model.seasonal_order

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
        
        # Preprocess data
        data = preprocess_data(data)
        
        # Split data into features (X) and target (y)
        X = data[['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']]
        y = data['Close']
        
        # Train-Test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Find best ARIMA model
        best_order, best_seasonal_order = find_best_arima_model(y_train)
        
        # Fit SARIMAX model
        model = SARIMAX(y_train, exog=X_train, order=best_order, seasonal_order=best_seasonal_order)
        model_fit = model.fit()

        # Make predictions for the test set
        predictions = model_fit.forecast(steps=len(y_test), exog=X_test)

        # Calculate accuracy metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # Print accuracy metrics
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse}")
        logger.info(f"Mean Absolute Error (MAE): {mae}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {mape}%")

        # Prepare data for future forecast
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        future_exog = pd.DataFrame(index=future_dates, columns=X.columns)
        
        # Fill future exogenous data with last known values (you may want to improve this)
        for col in future_exog.columns:
            future_exog[col] = X[col].iloc[-1]

        # Make predictions for the future (forecast)
        forecast = model_fit.forecast(steps=forecast_days, exog=future_exog)
        
        # Add volatility and sentiment adjustment
        volatility = np.std(y.pct_change())
        noise = np.random.normal(0, volatility, forecast_days)
        sentiment_adjustment = sentiment * 0.01  # Adjust the impact of sentiment

        logger.info(f"Forecast shape: {forecast.shape}")
        logger.info(f"Noise shape: {noise.shape}")
        logger.info(f"Sentiment adjustment: {sentiment_adjustment}")

        # Perform element-wise multiplication
        forecast = forecast * (1 + noise + sentiment_adjustment)

        historical_dates = y.index[-365:].strftime('%Y-%m-%d').tolist()
        historical_data = y.values[-365:].tolist()
        forecast_dates = future_dates.strftime('%Y-%m-%d').tolist()
        forecast_data = forecast.tolist()

        # Helper function to convert NaN to None
        def nan_to_none(value):
            return None if np.isnan(value) else float(value)

        return {
            'historical_dates': historical_dates,
            'historical_data': historical_data,
            'forecast_dates': forecast_dates,
            'forecast_data': [nan_to_none(x) for x in forecast_data],
            'sentiment': float(sentiment),
            'news_data': news_data[:10],  # Return only the 10 most recent news items
            'accuracy_metrics': {
                'mse': nan_to_none(mse),
                'rmse': nan_to_none(rmse),
                'mae': nan_to_none(mae),
                'mape': nan_to_none(mape)
            }
        }

    except Exception as e:
        logger.error(f"Error in predict function for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        return None