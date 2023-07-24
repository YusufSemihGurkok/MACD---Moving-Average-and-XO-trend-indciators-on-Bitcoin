import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime
from ta.trend import ema_indicator
import ta.trend as trend
import ta.momentum as momentum
from binance.exceptions import BinanceAPIException
import ta


def calculate_macd(btc_close_data):
    # Convert the input data to a pandas Series
    btc_series = pd.Series(btc_close_data)

    # Calculate the 12-period EMA
    ema12 = btc_series.ewm(span=18, adjust=False).mean()

    # Calculate the 26-period EMA
    ema26 = btc_series.ewm(span=32, adjust=False).mean()

    # Calculate the MACD line
    macd = ema12 - ema26

    # Calculate the signal line
    signal = macd.ewm(span=9, adjust=False).mean()

    # Return the MACD and signal lines as numpy arrays
    return macd.values, signal.values

# Set up Binance API client
api_key = "General-API-Here"
api_secret = "Secret-API-Here"
client = Client(api_key, api_secret)

# Define trading pair and timeframe
symbol = 'BTCUSDT'
interval = 'Client.KLINE_INTERVAL_5MINUTE'

# Set start and end time for historical data
end_time = datetime.now()
start_time = end_time - pd.DateOffset(months=6)

# Convert start and end times to milliseconds
start_time_ms = int(start_time.timestamp() * 1000)
end_time_ms = int(end_time.timestamp() * 1000)

# Get historical price data from Binance API
klines = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_5MINUTE, "16 January 2023", "16 July 2023")

# Create DataFrame from the klines data
data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                         'taker_buy_quote_asset_volume', 'ignore'])

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

# Convert numeric columns to float
numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
data[numeric_cols] = data[numeric_cols].astype(float)

# Set timestamp as the DataFrame index
data.set_index('timestamp', inplace=True)

# Calculate the 150-day Simple Moving Average (SMA)
data['sma_150'] = trend.SMAIndicator(data['close'], window=150).sma_indicator()

# Calculate the MACD indicator
macd_data, macd_signal = calculate_macd(data['close'])


# Assign MACD values to the 'macd' column in the DataFrame
data['macd'] = macd_data
data['macd_signal'] = macd_signal


# Create the XO Trend Indicator
short_ema = data['close'].ewm(span=16, adjust=False).mean()
long_ema = data['close'].ewm(span=30, adjust=False).mean()

data['xo_trend'] = np.where(short_ema > long_ema, 1, -1)


data['buy_signal'] = ((data['close'] > data['sma_150']) &
                      (data['macd'] > 0) &
                      (data['xo_trend'] > 0))

data['sell_signal'] = ((data['close'] < data['sma_150']) &
                       (data['macd'] < 0) &
                       (data['xo_trend'] < 0))

# Backtesting
position = 0  # 0: No position, 1: Long position, -1: Short position
trades = []  # Store trade details

for i in range(len(data)):
    if data['buy_signal'].iloc[i] and position != 1:
        # Enter long position
        position = 1
        trade = {'entry_price': data['close'].iloc[i], 'exit_price': None, 'profit': None}
        trades.append(trade)
    elif data['sell_signal'].iloc[i] and position != -1:
        # Enter short position
        position = -1
        trade = {'entry_price': data['close'].iloc[i], 'exit_price': None, 'profit': None}
        trades.append(trade)
    elif position == 1 and data['close'].iloc[i] < data['xo_trend'].iloc[i]:
        # Exit long position
        position = 0
        trades[-1]['exit_price'] = data['close'].iloc[i]
        trades[-1]['profit'] = trades[-1]['exit_price'] - trades[-1]['entry_price']
    elif position == -1 and data['close'].iloc[i] > data['xo_trend'].iloc[i]:
        # Exit short position
        position = 0
        trades[-1]['exit_price'] = data['close'].iloc[i]
        trades[-1]['profit'] = trades[-1]['entry_price'] - trades[-1]['exit_price']

# Calculate overall profit
total_profit = sum(trade['profit'] for trade in trades if trade['profit'] is not None)

# Print total profit
print('Total Profit: ', total_profit)

                                     