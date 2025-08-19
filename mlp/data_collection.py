import yfinance as yf
import numpy as np
import pandas as pd
import sys

def getdata(ticker_symbol,period=None,start=None,end=None,interval="1d"):
    """Pulls data from the yFinance database and returns a numpy array of those values\n
    \n
    args:\n
    tickers = str or tuple of ticker symbols\n
    period = period for which to download data\n
    start = start of period of interest (YYYY-MM-DD)\n
    end = end of period of interest (YYYY-MM-DD)\n
    interval = timeframe in which the data will be collected 
    """

    data = yf.download(tickers=ticker_symbol, period=period, start=start, end=end, interval=interval, auto_adjust=False, threads=False)

    if (data == 0).all().all() or (data == None).all().all():
        print("Failed to download the data")
        sys.exit()
    else:
        print(f"Data collected for {ticker_symbol} with timeframe of {interval}")
        np.save(ticker_symbol,data.to_numpy())

    return data

def read_data(data_file_name):
    data = np.load(data_file_name)
    datadf = pd.DataFrame(data)

    return datadf