# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:52:35 2020

@author: tahoehigh
"""
import bs4 as bs
import datetime as dt
from datetime import datetime
import os
from pandas_datareader import data as pdr
import pickle
import requests
#import fix_yahoo_finance as yf
import yfinance as yf
import pandas as pd


#raw_input1, raw_input2, raw_input3 = input("Please provide a year, month, and start day").split()
#print(raw_input1,raw_input2,raw_input3)
yf.pdr_override

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    Ticker_Dataframe = pd.DataFrame(tickers,columns = ['Ticker_Name'])
    Ticker_Dataframe.to_excel("S&P500 Current Ticker List.xlsx", index=False)
    return tickers



# save_sp500_tickers()
   #start = dt.datetime(2019, 6, 8)
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    date_input = input("Please provide a date seperated by dashes (e.g., 12-30-1994)")
    time_in_datetime = datetime.strptime(date_input, "%m-%d-%Y")
    print(time_in_datetime)
    start = time_in_datetime
    end = dt.datetime.now()
    for ticker in tickers:
        try:
            print(ticker)
            #if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        except:
            print(f"Issue with Ticker: {ticker}")
            #else:
            #print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

save_sp500_tickers()
get_data_from_yahoo()
compile_data()
