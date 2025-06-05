import json
import pandas as pd

from bs4 import BeautifulSoup
import requests

w_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#Selected_changes_to_the_list_of_S&P_500_components"

def get_data(url):
    wiki_text = requests.get(url)
    soup = BeautifulSoup(wiki_text.text, 'html.parser')
    return soup

def get_table(soup, class_name):
    table = soup.find('table', attrs={'class':class_name})
    return table

soup = get_data(w_url)
stock_table_class = "wikitable sortable sticky-header jquery-tablesorter"
stock_table_id = "constituents"

for i in stock_table_class:
    print(i)

# table = get_table(soup, stock_table_class)
# print(table)
