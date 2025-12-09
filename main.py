import datetime as dt
import pandas as pd
from pathlib import Path

from IPython.core.display_functions import display

today = dt.datetime.today()

year = today.year
month = today.month
day = today.day
weekday = today.weekday()

print(f"Year: {year}")
print(f"Month: {month}")
print(f"Day: {day}")
print(f"Weekday: {weekday}")

companies_df = pd.read_csv(Path(f"./data/stock_lists/companies.csv"))

display(companies_df)