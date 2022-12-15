import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime, timedelta
from pandas import ExcelWriter
import xmltodict as xmltodict

import numpy as np

#%matplotlib inline

from pprint import pprint
currences=['KRW','KGS','USD','CNY']

def currency_today(valute):
    # Делаем запрос к Центробанку
    url = 'http://www.cbr.ru/scripts/XML_daily.asp?date_req='
    date = datetime.now()
    date = date.strftime('%d/%m/%Y')
    response = requests.get(url + date)
    tree = xmltodict.parse(response.content)
    date = datetime.strptime(date, '%d/%m/%Y')
    #pprint(tree['ValCurs']['Valute'])
    # Создаем таблицу с нужными данными
    df = pd.DataFrame(tree['ValCurs']['Valute'])
    df['Date'] = tree['ValCurs']['@Date']
    df = df.set_index(['CharCode'])
    return df.loc[valute]

def convert_to_datetime(row):
    return datetime.strptime(row['Date'], '%d.%m.%Y')

#print(currency_today('USD'))
# Скачиваем файл с данными
def storage(name):
    cur_name = name
    file = name+'.xlsx'
    name = pd.read_excel(file, index_col='CharCode')
    name=name.append(currency_today(cur_name))
    #name = pd.concat([name, currency_today(cur_name)])
    name = name.drop_duplicates()
    name.to_excel(file)
    name['Value'] = name['Value'].str.replace(',', '.')
    name.Value = name.Value.apply(lambda x: float(x))
    name['Date'] = name.apply(convert_to_datetime, axis=1)
    return name

for currency in currences:
    storage(currency)


USD = storage('USD')
KRW = storage('KRW')
KGS = storage('KGS')
CNY = storage('CNY')

#print(KRW)
plt.plot(USD.Date, USD.Value, label="USD")
plt.plot(CNY.Date, CNY.Value, label="CNY")
plt.plot(KGS.Date, KGS.Value, label="KGS")
plt.plot(KRW.Date, KRW.Value, label="KRW")

plt.legend()
plt.show()



