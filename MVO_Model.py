#This is a my first crack at generator an Mean Variance Optimization model
#The user gets to decide what index the MVO is built from
#I have 3 seperate functions to pull the tickers for either the S&P500,
#NASDAQ, or DOW

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import datetime as dt
import os
from pandas_datareader import data as pdr
import pickle
import requests
from bs4 import BeautifulSoup
import bs4 as bs
#import fix_yahoo_finance as yf

###---User Input Section
start_date = '2010-01-01'
end_date = '2021-12-31'

index_selection = input('Input the index you want (i.e, "DOW","SPY", or "NASDAQ")')


###---Functions
def save_NASDAQ100_tickers():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100#Components"
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table containing the list of tickers
    ticker_table = soup.find("table", {"id": "constituents"})

    # Extract ticker symbols from the table
    tickers = []
    for row in ticker_table.findAll("tr")[1:]:
        ticker_cell = row.findAll("td")[1]
        ticker_symbol = ticker_cell.text.strip()
        tickers.append(ticker_symbol)

    print(tickers)
    with open("NASDAQ100tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    Ticker_Dataframe = pd.DataFrame(tickers,columns = ['Ticker_Name'])
    Ticker_Dataframe.to_excel("NASDAQ100 Current Ticker List.xlsx", index=False)
    return tickers

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

def save_dow_jones_tickers():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components"
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"class": "wikitable sortable"})
    rows = table.find_all("tr")

    tickers = []
    for row in rows[1:]:
        cells = row.find_all("td")
        ticker = cells[1].text.strip()
        tickers.append(ticker)
    with open("dowjones.pickle", "wb") as f:
        pickle.dump(tickers, f)
    Ticker_Dataframe = pd.DataFrame(tickers,columns = ['Ticker_Name'])
    Ticker_Dataframe.to_excel("DowJones Current Ticker List.xlsx", index=False)
    return tickers

save_dow_jones_tickers()
save_NASDAQ100_tickers()

# Define the function to get stock data
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Define the function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

# Define the function for mean-variance optimization
def mean_variance_optimization(mean_returns, cov_matrix, min_weight=0.05):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun': lambda x: x - min_weight})
    bounds = tuple((min_weight, 1) for asset in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])

    optimized_weights = minimize(portfolio_performance, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized_weights.x

# Define the function for backtesting
def backtest(tickers, start_date, end_date):
    data = get_stock_data(tickers, start_date, end_date)
    quarterly_rebalance_dates = pd.date_range(start_date, end_date, freq='QS')
    portfolio_weights = []

    for date in quarterly_rebalance_dates:
        start = date - pd.DateOffset(years=1)
        daily_returns = data.loc[start:date].pct_change().dropna()
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()

        optimized_weights = mean_variance_optimization(mean_returns, cov_matrix)
        portfolio_weights.append(optimized_weights)

    return quarterly_rebalance_dates, portfolio_weights

# Define the S&P500 tickers (replace this list with the actual tickers of the S&P500 companies)
if index_selection == 'DOW':
    tickers = save_dow_jones_tickers()

elif index_selection == 'NASDAQ':
    tickers = save_NASDAQ100_tickers()

else:
    tickers = save_sp500_tickers()


if __name__ == "__main__":

    rebalance_dates, weights = backtest(tickers, start_date, end_date)

    # Print results
    for date, weight in zip(rebalance_dates, weights):
        print(f"Rebalance date: {date}, Portfolio weights: {weight}")
