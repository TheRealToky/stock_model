import pandas as pd
import yfinance as yf
import json

"""
This file fetches all data from individual tickers and adds them to the database.
It uses a class as a blueprint for each company
"""

# To do:
# Fetch the sp 500 list from the database
# Add sp 500 tickers into list
# add function result to database

# Make this file run once every month for data verification purpose

class Company:

    def __init__(self, ticker):
        # add a company name argument when you figure out how to do it
        # self.companyName = companyName
        self.tick = yf.Ticker(ticker)
        self.ticker = ticker

    def get_historical_data (self, ticker):
        self.tick = yf.Ticker(ticker)

        historical_data = self.tick.history(period='max', interval='1d')
        df_historical_data = pd.DataFrame(historical_data)
        return df_historical_data


    def get_information (self, ticker):
        tick = yf.Ticker(ticker)

        tick_info = tick.info  # ticker's general info
        tick_info = json.dumps(tick_info, indent=4)
        return tick_info


    def get_financials (self, ticker):
        tick = yf.Ticker(ticker)

        financials = tick.financials
        df_financials = pd.DataFrame(financials)
        return df_financials


    def get_cashflow (self, ticker):
        tick = yf.Ticker(ticker)

        tick_cashflow = tick.cashflow  # shows the cashflow data
        df_tick_cashflow = pd.DataFrame(tick_cashflow)
        return df_tick_cashflow


    def get_balancesheet (self, ticker):
        tick = yf.Ticker(ticker)

        tick_bs = tick.balancesheet
        df_bs = pd.DataFrame(tick_bs)
        return df_bs


# bla, bla, bla something that fetches the sp500 list
ticker_list = ["AAPL", "MSFT", "TSLA"]

# change ticker_list with a dictionary with the ticker and the company Name

if __name__ == '__main__':
    for ticker in ticker_list:
        # need to figure out how to get the company name
        # company = Company(ticker, companyName)
        company = Company(ticker)


        historical_data = company.get_historical_data(ticker)
        information =  company.get_information(ticker)
        financials = company.get_financials(ticker)
        cashflow = company.get_cashflow(ticker)
        balancesheet = company.get_balancesheet(ticker)

    # bla bla bla, something that adds these information to the database
