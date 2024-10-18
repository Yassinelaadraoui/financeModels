import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(ticker+'.csv')
    return df

# Example usage
ticker = "AAPL"
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2023, 10, 1)


df = get_data(ticker, start, end)
#print(df.head())

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.trend import ADXIndicator, PSARIndicator
from sklearn.preprocessing import StandardScaler

def data_preprocessing(ticker):
    dataset = pd.read_csv('{}.csv'.format(ticker))
    dataset = dataset.dropna()
    dataset = dataset[['Date', 'Open', 'High', 'Low', 'Close']]  # Ensure the 'Date' column is present in the CSV

    # Custom features
    dataset['H-L'] = dataset['High'] - dataset['Low']
    dataset['O-C'] = dataset['Close'] - dataset['Open']
    
    # Moving averages
    dataset['ma_5'] = dataset['Close'].rolling(window=5).mean()
    dataset['ma_10'] = dataset['Close'].rolling(window=10).mean()
    
    # Exponential moving average
    dataset['EWMA_12'] = dataset['Close'].ewm(span=12).mean()

    # Standard deviation
    dataset['std_5'] = dataset['Close'].rolling(window=5).std()
    dataset['std_10'] = dataset['Close'].rolling(window=10).std()
    
    # RSI (Relative Strength Index) - replacing TA-Lib with ta library
    rsi_indicator = RSIIndicator(close=dataset['Close'], window=14)
    dataset['RSI'] = rsi_indicator.rsi()
    
    # Williams %R - replacing TA-Lib with ta library
    willr_indicator = WilliamsRIndicator(high=dataset['High'], low=dataset['Low'], close=dataset['Close'], lbp=7)
    dataset['Williams %R'] = willr_indicator.williams_r()

    # Parabolic SAR - replacing TA-Lib with ta library
    psar_indicator = PSARIndicator(high=dataset['High'], low=dataset['Low'], close=dataset['Close'], step=0.2, max_step=0.2)
    dataset['SAR'] = psar_indicator.psar()

    # ADX (Average Directional Index) - replacing TA-Lib with ta library
    adx_indicator = ADXIndicator(high=dataset['High'], low=dataset['Low'], close=dataset['Close'], window=10)
    dataset['ADX'] = adx_indicator.adx()
    
    # Binary feature indicating if the price will rise the next day
    dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)
    
    # Drop rows with NaN values
    dataset = dataset.dropna()
    
    # Features (X), target (y), and dates
    X = dataset.iloc[:, 5:-1]
    y = dataset.iloc[:, -1]
    dates = dataset['Date']
    
    # Split into training and testing sets
    split = int(len(dataset) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    test_dates = dates[split:]  # Dates for the testing set
    
    # Standardize the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test, test_dates



import pandas as pd
from sklearn import svm
from collections import Counter

def svm_linear(ticker):
    # Get preprocessed data and test dates
    X_train, X_test, y_train, y_test, test_dates = data_preprocessing(ticker)
    
    # Train SVM classifier with linear kernel
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    # Calculate accuracy
    confidence = clf.score(X_test, y_test)
    
    # Get predictions
    predictions = clf.predict(X_test)
    
    # Print accuracy
    print('Accuracy:', confidence)
    
    # Save output to a CSV file (date, actual, predicted)
    output = pd.DataFrame({
        'Date': test_dates.values,  # Dates for test set
        'Actual': y_test.values,    # Actual values
        'Predicted': predictions    # Predicted values
    })
    
    # Save the dataframe as CSV
    output.to_csv(f'{ticker}_svm_output.csv', index=False)
    
    print(f"Output saved to {ticker}_svm_output.csv")

# Example usage:
svm_linear('AAPL')
