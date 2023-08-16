import yfinance as yf
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


#1 - Data Collection
currentDate = datetime.date.today().strftime('%Y-%m-%d')
print(currentDate)

#Defining Ticker Symbol
tickerSymbol = 'AAPL'

#Getting data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#Getting historic data up to dynamic enddate
tickerDf = tickerData.history(period='id', start='2010-1-1', end=currentDate)
tickerDf.to_csv('AAPL_stock_data.csv')

data = pd.read_csv('AAPL_stock_data.csv')
data = data.dropna()
data = data.fillna(method='ffill') #Forward fill, meaning that if a value is missing, it is replaced with the last valid value
#print(data.head())

#Initializing Scaler to properly train model
scaler = MinMaxScaler()
#Fitting Data to look at closing prices
data['Close'] = scaler.fit_transform(data[['Close']])

#Converting Data into sequences
def createSequences(data, seqLength):
    xs, ys = [], []
    for i in range(len(data) - seqLength):
        x = data[i:(i + seqLength)]
        y = data[i + seqLength]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
seqLength = 90
X, y = createSequences(data['Close'].values, seqLength)


#Splitting Data into training and testing sets
trainingSize = int(len(X) * 0.8) #Using 80% of the data for training
xTrain, yTrain = X[:trainingSize], y[:trainingSize]
xTest, yTest = X[trainingSize:], y[trainingSize:]

#Reshaping input for LSTM
xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))