# This file parses the wikipidedia page on the list of the
# sp500 stock then fetches the stock tables and the stock change table

# Imports
# import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

# URL to parse
w_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#Selected_changes_to_the_list_of_S&P_500_components"

# Functions
def get_data(url):  # returns the whole website
 wiki_text = requests.get(url)
 soup = BeautifulSoup(wiki_text.text, 'html.parser')
 return soup


def get_html_data(soup, html_tag, class_name, id):  # returns the specific tag with class and id
 raw_html_data = soup.find(html_tag, attrs={'class': class_name, 'id': id})
 return raw_html_data


# Get stock table
raw_stock_table_class = "wikitable sortable sticky-header jquery-tablesorter"
stock_table_class = raw_stock_table_class.split()

stock_table_id = "constituents"

soup = get_data(w_url)
# raw_table = get_html_data(soup, 'table', stock_table_class, stock_table_id)
# print(raw_table.prettify())

# Filter data
## Get wikipedia stock tables
tables = pd.read_html(StringIO(str(soup)), flavor='lxml') # fetches the 2 tables
stock_list = tables[0]
stock_change_list = tables[1]

print("\n\n\n")
print(stock_list)
print("\n\n\n")
print(stock_change_list)

# bla, bla, bla something that add the list to a database

## Add stock tables to databases
# To do:
# Find way to store stock list into the database
