import yfinance as yf
import pandas as pd
import json

import bs4
import requests

tick = yf.Ticker('AAPL')

historical_data = tick.history(period='1y', interval='1d') # the stock's
print(historical_data)

tick_info = tick.info # ticker's general info
tick_info = json.dumps(tick_info, indent=4)
print(tick_info)

actions = tick.actions # dividends and stock splits
print(actions)

financials = tick.financials
df_financials = pd.DataFrame(financials)
print(df_financials.index)  # shows the index columns
# print(df_financials['col1'])  # shows the first column aside of the index column
# print(financials)

tick_cashflow = tick.cashflow  # shows the cashflow data
df_tc = pd.DataFrame(tick_cashflow)
print(df_tc.index)

tick_bs = tick.balancesheet
print(tick_bs)

