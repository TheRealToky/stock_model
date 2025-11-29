import pandas as pd
import yfinance as yf

"""
This file fetches all data from individual tickers and adds them to the database.


To do:
- Fetch the sp 500 list from the database
- Add sp 500 tickers into list
- add function result to database
"""

stock_list_df = pd.read_csv('data/stock_lists/stock_list.csv')
stock_change_df = pd.read_csv('data/stock_lists/stock_change_list.csv')

# Make this file run once every month for data verification purpose

def get_balancesheet(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    tick_bs = tick.balancesheet
    df_bs = pd.DataFrame(tick_bs)
    return df_bs


def get_historical_data(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    history = tick.history(period='max', interval='1d')
    df_history = pd.DataFrame(history)
    return df_history


def get_information(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    tick_info = tick.info  # ticker_symbol's general info
    # tick_info = json.dumps(tick_info, indent=4)
    return tick_info


def get_financials(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    finances = tick.financials
    df_financials = pd.DataFrame(finances)
    return df_financials


def get_cashflow (ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    tick_cashflow = tick.cashflow  # shows the cashflow data
    df_tick_cashflow = pd.DataFrame(tick_cashflow)
    return df_tick_cashflow


# bla, bla, bla something that fetches the sp500 list
# ticker_list = ["AAPL", "MSFT", "TSLA"]
ticker_list = ["AAPL"]

# change ticker_list with a dictionary with the ticker and the company Name

if __name__ == '__main__':
    for ticker in ticker_list:
        historical_data = get_historical_data(ticker)
        information =  get_information(ticker)
        financials = get_financials(ticker)
        cashflow = get_cashflow(ticker)
        balancesheet = get_balancesheet(ticker)

        print(get_information(ticker).keys())

    # bla bla bla, something that adds these information to the database
