import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging
import time
from binance.client import Client

# ============================
# Configuration Parameters
# ============================

# Binance Testnet API keys
API_KEY = 'ZsKcRvWjcwsuQHvmtN0EZ8SU9PQvjcmycX7Eod17RByypw5GBDbr1rvHI5WGbCv4'
API_SECRET = 'pLOrAOfuyiwdOCEbwYBGxhjjjo6Z0LiFT3mB5eiEwhsPX4YkP3XlOOqBPpRMDkyL'

# Initialize Binance Client for Testnet
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

# Paths and Parameters
DATA_PATH = 'data_test.csv'  # Path to your CSV file
MODEL_PATH = 'doge_gb_model.joblib'  # Path to saved ML model
SYMBOL = 'DOGEUSDT'  # Trading pair
INTERVAL = '1m'  # Time interval for data
QUANTITY = 300  # Quantity of DOGE to trade

# Technical Indicators Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
EMA_SHORT = 12
EMA_LONG = 26

# Logging Configuration
LOG_FILE = 'doge_live_trading.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def log_and_print(message):
    print(message)
    logging.info(message)

# ============================
# Data Loading
# ============================

def load_data(file_path):
    """
    Loads and preprocesses historical market data from a CSV file.
    """
    try:
        log_and_print(f"Loading data from {file_path}...")
        data = pd.read_csv(file_path)

        # Convert Date to datetime and sort
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)

        # Remove $ and commas from numeric columns and convert to float
        for column in ['Market Cap', 'Volume', 'Open', 'Close']:
            data[column] = data[column].replace({'\$': '', ',': ''}, regex=True).astype(float)

        log_and_print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        log_and_print(f"Error loading data: {e}")
        raise

# ============================
# Indicator Calculation
# ============================

def add_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    """
    try:
        log_and_print("Calculating technical indicators...")
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=RSI_PERIOD).rsi()

        macd = ta.trend.MACD(df['Close'], window_slow=MACD_SLOW,
                             window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        bollinger = ta.volatility.BollingerBands(df['Close'], window=BOLLINGER_WINDOW,
                                                 window_dev=BOLLINGER_STD)
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()

        df['ema_short'] = ta.trend.EMAIndicator(df['Close'], window=EMA_SHORT).ema_indicator()
        df['ema_long'] = ta.trend.EMAIndicator(df['Close'], window=EMA_LONG).ema_indicator()

        df.dropna(inplace=True)  # Drop rows with NaN values after indicator calculation
        log_and_print(f"Indicators added successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        log_and_print(f"Error adding indicators: {e}")
        raise

# ============================
# Machine Learning Model
# ============================

def train_model(df):
    """
    Train the machine learning model on historical data.
    """
    try:
        log_and_print("Training machine learning model...")

        # Ensure there is enough data
        if df.empty or len(df) < 10:
            raise ValueError("Insufficient data for training.")

        # Add target variable
        df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

        # Features and target
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_short', 'ema_long']
        X = df[features]
        y = df['target']

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, MODEL_PATH)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        log_and_print(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")
        return model
    except Exception as e:
        log_and_print(f"Error training model: {e}")
        raise

def load_model():
    """
    Loads the trained machine learning model.
    """
    try:
        model = joblib.load(MODEL_PATH)
        log_and_print("Machine learning model loaded successfully.")
        return model
    except Exception:
        log_and_print("No pre-trained model found. Training a new model.")
        data = load_data(DATA_PATH)
        data = add_indicators(data)
        return train_model(data)

# ============================
# Live Trading Logic
# ============================

def predict_signal(df, model):
    """
    Predicts the trading signal based on the latest data point.
    """
    try:
        latest = df.iloc[-1]
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_short', 'ema_long']
        input_data = latest[features].values.reshape(1, -1)
        prediction = model.predict(input_data)
        confidence = model.predict_proba(input_data)[0][prediction[0]]
        return prediction[0], confidence
    except Exception as e:
        log_and_print(f"Error predicting signal: {e}")
        return 'hold', 0

def place_live_order(signal, symbol, quantity):
    """
    Places a live buy or sell order on Binance Testnet.
    """
    try:
        if signal == 1:  # Buy signal
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            log_and_print(f"Market Buy Order placed: {order}")
        elif signal == 0:  # Sell signal
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            log_and_print(f"Market Sell Order placed: {order}")
        else:
            log_and_print("Hold signal, no order placed.")
    except Exception as e:
        log_and_print(f"Error placing live order: {e}")

def live_trading():
    """
    Executes live trading on Binance Testnet based on real-time data and model predictions.
    """
    log_and_print("Starting live trading on Binance Testnet...")
    model = load_model()

    while True:
        try:
            # Fetch real-time data
            data = load_data(DATA_PATH)  # Reuse the same CSV file for testing
            data = add_indicators(data)

            # Predict signal
            signal, confidence = predict_signal(data, model)
            action = 'Buy' if signal == 1 else 'Sell' if signal == 0 else 'Hold'
            log_and_print(f"Signal: {action} | Confidence: {confidence * 100:.2f}%")

            # Place live order based on the signal
            place_live_order(signal, SYMBOL, QUANTITY)

            # Sleep for the next interval
            time.sleep(60)  # Adjust based on the interval
        except Exception as e:
            log_and_print(f"Error in live trading loop: {e}")

# ============================
# Main Script
# ============================

if __name__ == "__main__":
    live_trading()
