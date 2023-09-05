import yfinance as yf
import datetime
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Defining the directory to save scaler objects
SCALER_DIR = 'scalers'
if not os.path.exists(SCALER_DIR):
    os.makedirs(SCALER_DIR)

def fetch_and_preprocess_data(tickerSymbol, start_date='2022-1-1'):
    """
    Fetch stock data for a given ticker, preprocess it, and then split it for training and testing.
    """
    try:
        # Get the current date
        currentDate = datetime.date.today().strftime('%Y-%m-%d')
        print(f"Fetching data for {tickerSymbol} up to {currentDate}")

        # Getting data on this ticker
        tickerData = yf.Ticker(tickerSymbol)

        # Getting historic data up to dynamic enddate
        tickerDf = tickerData.history(period='1d', start=start_date, end=currentDate)

        # Drop NA values and forward fill any missing data
        tickerDf = tickerDf.dropna()
        tickerDf = tickerDf.fillna(method='ffill')  # Forward fill#

        # Store original 'Close' prices
        tickerDf['Original_Close'] = tickerDf['Close']

        # Scaling the 'Close' column
        local_scaler = MinMaxScaler()
        tickerDf['Close'] = local_scaler.fit_transform(tickerDf[['Close']])
        
        # Save the scaler object
        scaler_filepath = os.path.join(SCALER_DIR, f"{tickerSymbol}_scaler.save")
        joblib.dump(local_scaler, scaler_filepath)
        
        return tickerDf, local_scaler

    except Exception as e:
        print(f"Error fetching and preprocessing data: {str(e)}")
        return None

def createSequences(data, seqLength=90):
    """
    Convert the data into sequences for LSTM training.
    """
    xs, ys = [], []
    for i in range(len(data) - seqLength):
        x = data[i:(i + seqLength)]
        y = data[i + seqLength]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data_for_lstm(tickerDf, seqLength=90):
    """
    Prepare data for LSTM: Create sequences, split for training/testing, and reshape.
    """
    X, y = createSequences(tickerDf['Close'].values, seqLength)

    # Splitting Data into training and testing sets
    trainingSize = int(len(X) * 0.8)  # Using 80% of the data for training
    xTrain, yTrain = X[:trainingSize], y[:trainingSize]
    xTest, yTest = X[trainingSize:], y[trainingSize:]

    # Reshaping input for LSTM
    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))

    return xTrain, yTrain, xTest, yTest

# import yfinance as yf
# import datetime
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# # Initializing Scaler to properly train model
# scaler = MinMaxScaler()

# def fetch_and_preprocess_data(tickerSymbol, start_date='2020-1-1'):
#     """
#     Fetch stock data for a given ticker, preprocess it, and then split it for training and testing.
#     """
#     try:
#         # Get the current date
#         currentDate = datetime.date.today().strftime('%Y-%m-%d')
#         print(f"Fetching data for {tickerSymbol} up to {currentDate}")

#         # Getting data on this ticker
#         tickerData = yf.Ticker(tickerSymbol)

#         # Getting historic data up to dynamic enddate
#         tickerDf = tickerData.history(period='1d', start=start_date, end=currentDate)

#         # Drop NA values and forward fill any missing data
#         tickerDf = tickerDf.dropna()
#         tickerDf = tickerDf.fillna(method='ffill')  # Forward fill#

#         # Store original 'Close' prices
#         tickerDf['Original_Close'] = tickerDf['Close']

#         # Scaling the 'Close' column
#         local_scaler = MinMaxScaler()
#         tickerDf['Close'] = local_scaler.fit_transform(tickerDf[['Close']])
#         return tickerDf, local_scaler


#     except Exception as e:
#         print(f"Error fetching and preprocessing data: {str(e)}")
#         return None

# def createSequences(data, seqLength=90):
#     """
#     Convert the data into sequences for LSTM training.
#     """
#     xs, ys = [], []
#     for i in range(len(data) - seqLength):
#         x = data[i:(i + seqLength)]
#         y = data[i + seqLength]
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

# def prepare_data_for_lstm(tickerDf, seqLength=90):
#     """
#     Prepare data for LSTM: Create sequences, split for training/testing, and reshape.
#     """
#     X, y = createSequences(tickerDf['Close'].values, seqLength)

#     # Splitting Data into training and testing sets
#     trainingSize = int(len(X) * 0.8)  # Using 80% of the data for training
#     xTrain, yTrain = X[:trainingSize], y[:trainingSize]
#     xTest, yTest = X[trainingSize:], y[trainingSize:]

#     # Reshaping input for LSTM
#     xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
#     xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))

#     return xTrain, yTrain, xTest, yTest

