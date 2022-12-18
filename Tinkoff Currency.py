from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import requests
import xmltodict as xmltodict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
import json

pd.set_option('display.max_columns',10)
response = requests.get('https://www.tinkoff.ru/api/v1/currency_rates/')
todos = json.loads(response.content)
ts = int(todos['payload']['lastUpdate']['milliseconds']/1000)
#print(ts)
date = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#pprint(todos['payload']['rates'])
data = todos['payload']['rates']
print(date)
for item in data:
    item['fromCurrency'] = item['fromCurrency']['name']
    item['toCurrency'] = item['toCurrency']['name']

df = pd.DataFrame(todos['payload']['rates'])
print(df[df['category'] == 'DebitCardsTransfers'])
#   print(df)
# Создаем таблицу с нужными данными

#df = pd.DataFrame(tree['ValCurs']['Valute'])
#df['Date'] = tree['ValCurs']['@Date']
#df = df.set_index(['CharCode'])