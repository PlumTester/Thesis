from alpha_vantage.foreignexchange import ForeignExchange
# from alpha_vantage.techindicators import TechIndicators

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib


def normalise(df):
    df_max = df.max()
    df_min = df.min()
    return (2*(df - df_min)/(df_max-df_min) - 1)


key = 'abcd'

fx = ForeignExchange(key, output_format='pandas')
# # ti = TechIndicators(key)

# pull available daily prices, this is until 2001-03-16
df = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')[0]

# rename alpha_vantage's weird column naming
df.rename(columns={"1. open": "Open", "2. high" : "High", "3. low" : "Low", "4. close" : "Close"}, inplace=True)

# next day's price
df['Close_t+1'] = df['Close'].shift(periods=1)

# Smoothing indicators
df['MA'] = talib.MA(df['Close'], timeperiod=30, matype=0)
df['EMA'] = talib.EMA(df['Close'], timeperiod=30)
df['TEMA'] = talib.TEMA(df['Close'], timeperiod=30)

# Trend indicators
df['ROCP'] = talib.ROCP(df['Close'], timeperiod=10)
df['MOM'] = talib.MOM(df['Close'], timeperiod=10)

# Oscillator indicators
df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
df['PPO'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26, matype=0)

# Volatility indicators
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

# Momentum indicators
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)

# drop any rows with NaN values
df.dropna(inplace=True)

# normalise the data to (-1, 1)
df_scaled = pd.DataFrame()
for (columnName, columnData) in df.iteritems():
    if not columnName == 'Close_t+1':
        df_scaled[columnName] = normalise(columnData)


# 1 year
window = 365
obs = df['Open'].count()
count = 0

# target feature extracted per graph, written to excel later
target = pd.DataFrame(columns=('Start', 'End', 'Days', 'Close_t+1', 'Class'))

# make graph for raw and normalised data, put in directory according to Class (should be easier with keras later)
# sliding window of moving forward in time, dataset indexed from most recent to oldest, go backwards
for i in range(window, obs):
    lb = obs - i
    ub = obs - count - 1

    targetClass = "Up"
    price_change = df['Close_t+1'][ub] - df['Close'][ub]
    if price_change < 0:
        targetClass = "Down"

    target.loc[count] = [df.index[lb], df.index[ub], window, df['Close_t+1'][ub], targetClass]

    df[lb:ub].plot(legend=False)
    plt.savefig('Graphs\\' + targetClass + "\\" + str(count) + '.png')
    plt.close('all')

    df_scaled[lb:ub].plot(legend=False)
    plt.savefig('Graphs_Normalised\\'  + targetClass + "\\" + str(count) + '.png')
    plt.close('all')

    count += 1



# write to separate sheets
writer = pd.ExcelWriter("EURUSD.xlsx", enginer='xlsxwriter')
df.to_excel(writer, sheet_name="raw_data")
df_scaled.to_excel(writer, sheet_name="normalised_data")
target.to_excel(writer, sheet_name="target_data")
writer.save()

# train.to_excel(writer, sheet_name="Train")
# test.to_excel(writer, sheet_name="Test")


