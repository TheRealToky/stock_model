# Imports
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO



url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#Selected_changes_to_the_list_of_S&P_500_components"

def get_html_data(website_soup, html_tag, class_name, target_id): # returns the specific tag with class and id
    raw_html_data = website_soup.find(html_tag , attrs={'class': class_name, 'id': target_id})
    return raw_html_data


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br"
}

# Tables properties, we didn't need them anyway
# raw_stock_table_class = "wikitable sortable mw-collapsible sticky-header jquery-tablesorter mw-made-collapsible"
# stock_table_class = raw_stock_table_class.split()
# stock_table_id = "constituents"
#
# raw_changes_class = "wikitable sortable jquery-tablesorter"
# changes_class = raw_changes_class.split()
# changes_id = "changes"

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

tables = pd.read_html(StringIO(str(soup)), flavor='lxml')

stock_list = tables[1]
stock_change_list = tables[2]

stock_list.to_csv("../../data/stock_lists/stock_list.csv")
stock_change_list.to_csv("../../data/stock_lists/stock_change_list.csv")