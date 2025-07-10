import yfinance as yf
import pandas as pd
import json

import bs4
import requests

"""
This module uses the yfinance library to fetch all the companies' information
"""

tick = yf.Ticker('AAPL')

historical_data = tick.history(period='1y', interval='1d') # the stock's
historical_data = tick.history(interval='1d') # the stock's
df_historical_data = pd.DataFrame(historical_data)
print(df_historical_data)


tick_info = tick.info # ticker's general info
tick_info = json.dumps(tick_info, indent=4)
print(tick_info)

df_info = pd.DataFrame(tick_info) *** Not working
print(df_info)


actions = tick.actions # dividends and stock splits
print(actions)


financials = tick.financials
df_financials = pd.DataFrame(financials)
print(df_financials.index)  # shows the index columns
print(df_financials['col1'])  # shows the first column aside of the index column
print(financials)


tick_cashflow = tick.cashflow  # shows the cashflow data
df_tc = pd.DataFrame(tick_cashflow)
print(df_tc)

tick_bs = tick.balancesheet
print(tick_bs)
