import yfinance as yf
import pandas as pd
import json

df = pd.read_csv("data/stock_lists/stock_by_market_cap.csv")

print(df.dtypes)
